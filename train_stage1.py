"""
Stage 1 Training: subject-relative micro-motion representation

Trains PTE + MoViNet-A2 on 30-second video epochs with cross-entropy plus
ISDL (a triplet loss sampled within each subject).

Usage:
    python train_stage1.py --config configs/stage1_train.yaml
"""
import os
import math
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.classification import Accuracy, F1Score
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import _LRScheduler

from models.movinet import build_movinet
from data.video_dataset import build_video_loader

# Set seeds
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


class WarmupCosineAnnealingLR(_LRScheduler):
    """Learning rate scheduler with warmup followed by cosine annealing."""
    def __init__(self, optimizer, total_epochs, warmup_epochs, base_lr, eta_min=1e-6, last_epoch=-1):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = float(self.last_epoch + 1) / float(self.warmup_epochs)
            return [self.base_lr * warmup_factor for _ in self.optimizer.param_groups]
        else:
            progress = float(self.last_epoch - self.warmup_epochs) / float(max(1, self.total_epochs - self.warmup_epochs))
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.eta_min + (self.base_lr - self.eta_min) * cosine_factor for _ in self.optimizer.param_groups]


class VideoDataModule(pl.LightningDataModule):
    """Data module for video classification."""
    def __init__(self, config, split_path):
        super().__init__()
        self.config = config
        self.split_path = split_path

    def setup(self, stage=None):
        # Load data splits
        def load_split(filename):
            path = os.path.join(self.split_path, filename)
            with open(path) as f:
                return [x.strip() for x in f.readlines()]

        self.train_list = load_split('train.txt')
        self.val_list = load_split('valid.txt')
        self.test_list = load_split('test.txt')

        print(f"Train: {len(self.train_list)}, Val: {len(self.val_list)}, Test: {len(self.test_list)}")

        # Build loaders
        self.train_loader = build_video_loader(
            data_path=self.config['data_path'],
            data_list=self.train_list,
            mode=self.config['mode'],
            batch_size=self.config['batch_size_t'],
            transforms_config=self.config['transforms_t'],
            sampler='weighted',
            num_workers=self.config['num_workers'],
            proportion=self.config['proportion_t'],
            num_classes=self.config['num_classes'],
            same_aug_for_contrastive=self.config.get('same_aug_for_contrastive', False)
        )

        self.val_loader = build_video_loader(
            data_path=self.config['data_path'],
            data_list=self.val_list,
            mode=self.config['mode'],
            batch_size=self.config['batch_size_v'],
            transforms_config=self.config['transforms_v'],
            sampler=None,
            num_workers=self.config['num_workers'],
            num_classes=self.config['num_classes']
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


class Stage1Model(pl.LightningModule):
    """Stage 1 model: PTE + MoViNet-A2 trained with cross-entropy + ISDL."""
    def __init__(self, model_config, learning_config, data_config, experiment_name):
        super().__init__()
        self.save_hyperparameters()

        self.lr = learning_config['lr']
        self.num_epochs = learning_config['num_epochs']
        self.warmup_epochs = learning_config.get('warmup_epochs', 20)
        self.scheduler_name = learning_config.get('scheduler_name', None)
        self.skip_val_epoch = learning_config.get('skip_val_epoch', 20)
        self.double_loss = data_config['mode'] == 'double_loss'

        # Build model
        self.model = build_movinet(
            causal=False,
            pretrained=model_config['pretrained'],
            num_classes=model_config['num_classes'],
            make_features=False,
            double_loss=self.double_loss
        )

        # Loss functions
        self.criterion = nn.CrossEntropyLoss()
        if self.double_loss:
            self.criterion_feature = nn.TripletMarginLoss(margin=1.0, p=2)

        # Metrics
        num_classes = model_config['num_classes']
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if self.double_loss:
            return self._double_loss_step(batch, is_train=True)
        else:
            frames, labels, _ = batch
            outputs, _ = self(frames)
            loss = self.criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            self.train_acc.update(preds, labels)
            self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
            self.log('train_acc', self.train_acc.compute(), on_epoch=True, sync_dist=True)
            return loss

    def _double_loss_step(self, batch, is_train=True):
        all_frames, all_labels = batch  # [B, 3, C, T, H, W], [B, 3]
        B, K, C, T, H, W = all_frames.size()

        all_frames = all_frames.reshape(-1, C, T, H, W)
        all_predictions, all_features = self(all_frames)
        all_features = all_features.reshape(B, K, -1)
        all_labels = all_labels.reshape(-1)

        # Losses
        feature_loss = self.criterion_feature(all_features[:, 0], all_features[:, 1], all_features[:, 2])
        class_loss = self.criterion(all_predictions, all_labels)
        total_loss = class_loss + feature_loss

        preds = all_predictions.argmax(dim=1)

        if is_train:
            self.train_acc.update(preds, all_labels)
            self.log('train_loss', class_loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
            self.log('train_feature_loss', feature_loss, on_step=True, on_epoch=True, sync_dist=True)
            self.log('train_total_loss', total_loss, on_step=True, on_epoch=True, sync_dist=True)
            return total_loss
        else:
            # Use only anchor for validation metrics
            preds = preds.reshape(B, K)[:, 0]
            labels = all_labels.reshape(B, K)[:, 0]
            self.all_preds.append(preds)
            self.all_labels.append(labels)
            self.log('val_loss', class_loss, on_epoch=True, sync_dist=True)
            return total_loss

    def on_train_epoch_start(self):
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        if self.current_epoch < self.skip_val_epoch:
            return

        if self.double_loss:
            return self._double_loss_step(batch, is_train=False)
        else:
            frames, labels, _ = batch
            outputs, _ = self(frames)
            loss = self.criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            self.all_preds.append(preds)
            self.all_labels.append(labels)
            self.log('val_loss', loss, on_epoch=True, sync_dist=True)

    def on_validation_epoch_start(self):
        self.val_acc.reset()
        self.val_f1.reset()
        self.all_preds = []
        self.all_labels = []

    def on_validation_epoch_end(self):
        if self.current_epoch < self.skip_val_epoch:
            self.log('val_f1', 0.0, sync_dist=True)
            return

        if len(self.all_preds) == 0:
            self.log('val_f1', 0.0, sync_dist=True)
            return

        all_preds = torch.cat(self.all_preds)
        all_labels = torch.cat(self.all_labels)

        f1 = self.val_f1(all_preds, all_labels)
        acc = self.val_acc(all_preds, all_labels)

        self.log('val_f1', f1, sync_dist=True)
        self.log('val_acc', acc, sync_dist=True)

        if self.trainer.is_global_zero:
            print(f"\n[Epoch {self.current_epoch}] Classification Report:")
            print(classification_report(all_labels.cpu().numpy(), all_preds.cpu().numpy(), digits=3))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        if self.scheduler_name == 'WarmupCosineAnnealingLR':
            scheduler = WarmupCosineAnnealingLR(
                optimizer,
                total_epochs=self.num_epochs,
                warmup_epochs=self.warmup_epochs,
                base_lr=self.lr,
                eta_min=1e-5
            )
            return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}]
        return optimizer


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train Stage 1 (PTE + MoViNet-A2, CE + ISDL)')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Experiment: {config['experiment_name']}")

    # Setup wandb
    if not config.get('wandb', False):
        os.environ["WANDB_MODE"] = "disabled"

    wandb_logger = WandbLogger(
        project=config.get('project_name', 'ViNUSS_stage1'),
        name=config['experiment_name'],
        config=config
    )

    # Data module
    data_module = VideoDataModule(
        config=config['data_config'],
        split_path=config['split_path']
    )

    # Model
    model = Stage1Model(
        model_config=config['model_config'],
        learning_config=config['learning_config'],
        data_config=config['data_config'],
        experiment_name=config['experiment_name']
    )

    # Callbacks
    dirpath = os.path.join(config['experiments_save_path'], config['experiment_name'])
    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1',
        mode='max',
        save_top_k=5,
        dirpath=dirpath,
        filename=config['experiment_name'] + '-{epoch:03d}-{val_f1:.3f}',
        verbose=True
    )

    early_stopping = pl.callbacks.EarlyStopping(monitor='val_f1', patience=50, mode='max')

    # Trainer
    trainer = pl.Trainer(
        devices=config['learning_config']['gpus'],
        max_epochs=config['learning_config']['num_epochs'],
        accelerator='gpu',
        logger=wandb_logger,
        precision='bf16-mixed',
        callbacks=[checkpoint_callback, early_stopping],
        strategy='ddp',
        accumulate_grad_batches=config['learning_config']['accumulate_grad_batches'],
        num_sanity_val_steps=0,
        check_val_every_n_epoch=config['learning_config']['check_val_every_n_epoch'],
        gradient_clip_val=1.0
    )

    # Train
    ckpt_path = config.get('previous_checkpoint')
    if ckpt_path:
        trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, datamodule=data_module)

    wandb_logger.experiment.finish()


if __name__ == '__main__':
    main()
