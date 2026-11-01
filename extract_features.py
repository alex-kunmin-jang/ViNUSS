"""
Feature Extraction: Extract MoViNet features for Stage 2

Loads a trained Stage 1 model and extracts features from all video data,
saving them as h5 files for Stage 2 training.

Usage:
    python extract_features.py --checkpoint checkpoints/stage1.ckpt --data_path /path/to/kvss_h5/224x224 --output_path ./features
"""
import os
import h5py
import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.movinet import build_movinet
from data.video_dataset import SleepVideoDataset
from data.augmentations import get_transforms


def extract_features(
    checkpoint_path: str,
    data_path: str,
    output_path: str,
    split_path: str = './data/splits',
    batch_size: int = 4,
    device: str = 'cuda'
):
    """
    Extract features from video data using trained MoViNet model.

    Args:
        checkpoint_path: Path to trained Stage 1 checkpoint
        data_path: Path to video h5 files (224x224)
        output_path: Path to save extracted features
        split_path: Path to data split files
        batch_size: Batch size for feature extraction
        device: Device to run on
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    hparams = checkpoint['hyper_parameters']
    model_config = hparams['model_config']
    state_dict = checkpoint['state_dict']

    # Build model
    model = build_movinet(
        causal=False,
        pretrained=False,
        num_classes=model_config['num_classes'],
        make_features=True,  # Enable feature extraction
        double_loss=False,
        tf_like=model_config.get('pretrained', True)
    )

    # Load weights (remove 'model.' prefix if present)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    # strict=False only for the ISDL projection head, which is not needed for feature extraction
    result = model.load_state_dict(new_state_dict, strict=False)
    assert not result.missing_keys, f"Missing keys in Stage 1 checkpoint: {result.missing_keys}"
    assert all(k.startswith('projection_head') for k in result.unexpected_keys), \
        f"Unexpected keys in Stage 1 checkpoint: {result.unexpected_keys}"

    model = model.to(device)
    model.eval()
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load all data lists
    all_data_list = []
    for split_file in ['train.txt', 'valid.txt', 'test.txt']:
        split_file_path = os.path.join(split_path, split_file)
        if os.path.exists(split_file_path):
            with open(split_file_path) as f:
                all_data_list.extend([x.strip() for x in f.readlines()])

    all_data_list = list(set(all_data_list))  # Remove duplicates
    print(f"Total subjects to process: {len(all_data_list)}")

    # Transforms (only normalization, no augmentation)
    transforms_config = {
        'normalize_stat': {'mean': 0.608, 'std': 0.144},
        'TensorRandomBrightness': 0,
        'TensorRandomContrast': 0,
        'TensorRandomFlip': 0,
        'TensorRandomRotation': 0,
        'TensorRandomScaling': [1.0, 1.0]
    }
    transforms = get_transforms(transforms_config)

    os.makedirs(output_path, exist_ok=True)

    # Process each subject
    for subject_name in tqdm(all_data_list, desc="Extracting features"):
        subject_output_path = os.path.join(output_path, subject_name)

        # Skip if already processed
        if os.path.exists(subject_output_path):
            print(f"Skipping {subject_name} (already exists)")
            continue

        # Create dataset for single subject
        dataset = SleepVideoDataset(
            data_path=data_path,
            data_list=[subject_name],
            mode='epochs',
            transforms=transforms,
            num_classes=5
        )

        if len(dataset) == 0:
            print(f"Skipping {subject_name} (no data)")
            continue

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        all_features = []
        all_labels = []

        with torch.no_grad():
            for frames, labels, _ in loader:
                frames = frames.to(device)
                _, features = model(frames)
                # [B, 640, 50, 1, 1]
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())

        # Concatenate all features and labels
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Save to h5 file
        with h5py.File(subject_output_path, 'w') as f:
            f.create_dataset('features', data=all_features, compression='gzip')
            f.create_dataset('labels', data=all_labels)

        print(f"Saved {subject_name}: features {all_features.shape}, labels {all_labels.shape}")

    print(f"\nFeature extraction complete. Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract features from Stage 1 model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to Stage 1 checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to video h5 files')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save features')
    parser.add_argument('--split_path', type=str, default='./data/splits', help='Path to split files')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()

    extract_features(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        output_path=args.output_path,
        split_path=args.split_path,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == '__main__':
    main()
