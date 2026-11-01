"""
Stage 2 Training: Sleep Dynamics Modeling (SDM)

Trains ShortMamba + bidirectional LongMamba on the Stage 1 features of
whole nights (one night per sample).

Usage:
    python train_stage2.py --config configs/stage2_train.yaml
"""
import os
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
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
import pandas as pd

from models.mamba import build_mamba
from data.feature_dataset import build_feature_loader

# Set seeds
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class FeatureDataModule(pl.LightningDataModule):
    """Data module for feature-based training."""
    def __init__(self, config, split_path, istest=False):
        super().__init__()
        self.config = config
        self.split_path = split_path
        self.istest = istest

    def setup(self, stage=None):
        def load_split(filename):
            path = os.path.join(self.split_path, filename)
            with open(path) as f:
                return [x.strip() for x in f.readlines()]

        self.train_list = load_split('train.txt')
        self.val_list = load_split('valid.txt')
        self.test_list = load_split('test.txt')

        print(f"Train: {len(self.train_list)}, Val: {len(self.val_list)}, Test: {len(self.test_list)}")

        if not self.istest:
            self.train_loader = build_feature_loader(
                data_path=self.config['data_path'],
                data_list=self.train_list,
                mode=self.config['mode'],
                batch_size=self.config['batch_size_t'],
                num_workers=self.config['num_workers']
            )
            self.val_loader = build_feature_loader(
                data_path=self.config['data_path'],
                data_list=self.val_list,
                mode=self.config['mode'],
                batch_size=self.config['batch_size_v'],
                num_workers=self.config['num_workers']
            )
        else:
            self.train_loader = []
            self.val_loader = []

        self.test_loader = build_feature_loader(
            data_path=self.config['data_path'],
            data_list=self.test_list,
            mode=self.config['mode'],
            batch_size=self.config['batch_size_v'],
            num_workers=self.config['num_workers']
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


class Stage2Model(pl.LightningModule):
    """Stage 2 model: SDM (ShortMamba + LongMamba)."""
    def __init__(self, model_config, learning_config, experiment_name, results_save_path):
        super().__init__()
        self.save_hyperparameters()

        self.experiment_name = experiment_name
        self.results_save_path = results_save_path
        self.learning_config = learning_config

        # Build model
        self.model = build_mamba(**model_config)

        # Loss
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

        # Metrics
        num_classes = model_config['num_classes']
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        features, labels, _ = batch
        outputs = self(features)
        outputs = outputs.permute(0, 2, 1)  # [B, num_classes, E]
        loss = self.criterion(outputs, labels)
        preds = outputs.argmax(dim=1)

        self.train_acc.update(preds, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('train_acc', self.train_acc.compute(), on_epoch=True, sync_dist=True)
        return loss

    def on_train_epoch_start(self):
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        features, labels, _ = batch
        outputs = self(features)
        outputs = outputs.permute(0, 2, 1)
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
        if len(self.all_preds) == 0:
            self.log('val_f1', 0.0, sync_dist=True)
            return

        all_preds = [p for batch in self.all_preds for p in batch]
        all_labels = [l for batch in self.all_labels for l in batch]
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        f1 = self.val_f1(all_preds, all_labels)
        acc = self.val_acc(all_preds, all_labels)

        self.log('val_f1', f1, sync_dist=True)
        self.log('val_acc', acc, sync_dist=True)

        if self.trainer.is_global_zero:
            print(f"\n[Epoch {self.current_epoch}] Validation Report:")
            print(classification_report(all_labels.cpu().numpy(), all_preds.cpu().numpy(), digits=3, zero_division=0))

    def test_step(self, batch, batch_idx):
        features, labels, subject_name = batch
        outputs = self(features)
        outputs = outputs.permute(0, 2, 1)
        preds = outputs.argmax(dim=1)

        self.test_preds.append(preds)
        self.test_labels.append(labels)
        self.test_subjects.append(subject_name)

    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_labels = []
        self.test_subjects = []

    def on_test_epoch_end(self):
        # Gather from all GPUs
        if dist.is_initialized():
            gathered_preds = [None] * dist.get_world_size()
            gathered_labels = [None] * dist.get_world_size()
            gathered_subjects = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_preds, self.test_preds)
            dist.all_gather_object(gathered_labels, self.test_labels)
            dist.all_gather_object(gathered_subjects, self.test_subjects)
            gathered_preds = [x for sublist in gathered_preds for x in sublist]
            gathered_labels = [x for sublist in gathered_labels for x in sublist]
            gathered_subjects = [x for sublist in gathered_subjects for x in sublist]
        else:
            gathered_preds = self.test_preds
            gathered_labels = self.test_labels
            gathered_subjects = self.test_subjects

        if self.trainer.is_global_zero:
            # Save results
            rows = []
            for i in range(len(gathered_preds)):
                subject = gathered_subjects[i][0] if isinstance(gathered_subjects[i], (tuple, list)) else gathered_subjects[i]
                subject = str(subject).replace('.h5', '')
                preds = gathered_preds[i].cpu().numpy()[0]
                labels = gathered_labels[i].cpu().numpy()[0]
                for e in range(len(preds)):
                    rows.append({
                        'subject': subject,
                        'epoch': e,
                        'prediction': preds[e],
                        'label': labels[e]
                    })

            df = pd.DataFrame(rows).drop_duplicates()
            save_path = os.path.join(self.results_save_path, self.experiment_name)
            os.makedirs(save_path, exist_ok=True)
            df.to_csv(os.path.join(save_path, 'test_results.csv'), index=False)
            print(f"Test results saved to {save_path}/test_results.csv")

            # Print report
            all_preds = df['prediction'].values
            all_labels = df['label'].values
            print("\nTest Classification Report:")
            print(classification_report(all_labels, all_preds, digits=3, zero_division=0))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_config['lr'])
        scheduler_name = self.learning_config.get('scheduler_name', False)
        if scheduler_name == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.learning_config['num_epochs'],
                eta_min=1e-6
            )
            return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}]
        return [optimizer]


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train Stage 2 (SDM)')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--test', action='store_true', help='Run test only')
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Experiment: {config['experiment_name']}")

    istest = args.test or config.get('istest', False)

    # Setup wandb
    if not config.get('wandb', False) or istest:
        os.environ["WANDB_MODE"] = "disabled"

    wandb_logger = WandbLogger(
        project=config.get('project_name', 'ViNUSS_stage2'),
        name=config['experiment_name'],
        config=config
    )

    # Data module
    data_module = FeatureDataModule(
        config=config['data_config'],
        split_path=config['split_path'],
        istest=istest
    )

    # Model
    model = Stage2Model(
        model_config=config['model_config'],
        learning_config=config['learning_config'],
        experiment_name=config['experiment_name'],
        results_save_path=config['experiments_save_path']
    )

    # Callbacks
    dirpath = os.path.join(config['experiments_save_path'], config['experiment_name'])
    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1',
        mode='max',
        save_top_k=3,
        dirpath=dirpath,
        filename=config['experiment_name'] + '-{epoch:03d}-{val_f1:.3f}',
        verbose=True
    )

    early_stopping = pl.callbacks.EarlyStopping(monitor='val_f1', patience=100, mode='max')

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

    if istest:
        ckpt_path = config.get('previous_checkpoint')
        if not ckpt_path:
            raise ValueError("Test mode requires previous_checkpoint")
        trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)
    else:
        ckpt_path = config.get('previous_checkpoint')
        if ckpt_path:
            trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
        else:
            trainer.fit(model, datamodule=data_module)

    wandb_logger.experiment.finish()


if __name__ == '__main__':
    main()
