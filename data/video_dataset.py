"""
Video dataset for Stage 1 training (PTE + MoViNet-A2, CE + ISDL).

Data format: h5 files with keys ['fps', 'frames', 'labels', 'num_epochs', 'padded_size']
- frames: (num_epochs, 150, 224, 224)
- labels: (num_epochs,)
- Fixed: 150 frames per epoch, 224x224 resolution

Two modes:
- epochs: Single epoch per sample
- double_loss: Triplet (anchor, positive, negative) for ISDL
"""
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, RandomSampler
from collections import Counter
from typing import Optional, List, Dict, Tuple

from .augmentations import get_transforms


class SleepVideoDataset(Dataset):
    """
    Sleep video dataset for Stage 1 training.

    Args:
        data_path: Path to directory containing h5 files
        data_list: List of h5 filenames to include
        mode: 'epochs' for single epoch or 'double_loss' for triplet sampling
        transforms: List of transform functions
        num_classes: Number of sleep stage classes
        same_aug_for_contrastive: Apply same augmentation to all triplet samples
    """
    def __init__(
        self,
        data_path: str,
        data_list: List[str],
        mode: str = 'epochs',
        transforms: Optional[List] = None,
        num_classes: int = 5,
        same_aug_for_contrastive: bool = False
    ):
        self.data_path = data_path
        self.transforms = transforms
        self.mode = mode
        self.num_classes = num_classes
        self.same_aug_for_contrastive = same_aug_for_contrastive

        # Fixed dimensions
        self.num_frames_per_epoch = 150
        self.height = 224
        self.width = 224

        if data_list is None:
            raise ValueError("data_list cannot be None")

        self.data_list = data_list
        self.total_num_epochs = 0
        self.total_num_subjects = len(data_list)

        # Build index maps
        self.subject_epoch_dict: Dict[str, int] = {}
        self.subject_label_loc: Dict[str, Dict[int, List[int]]] = {}
        self.all_label_dict: Dict[str, Dict[int, int]] = {}
        self.all_labels: List[int] = []

        for subject_name in data_list:
            subject_path = os.path.join(data_path, subject_name)
            with h5py.File(subject_path, 'r') as f:
                num_epochs = f['num_epochs'][()]
                labels = list(f['labels'][:])

                # Handle incomplete data
                if len(labels) != num_epochs:
                    num_epochs = len(labels)

                self.total_num_epochs += num_epochs
                self.subject_epoch_dict[subject_name] = num_epochs
                self.all_labels.extend(labels)

                # Map labels to epoch indices
                label_loc_map = {lbl: [] for lbl in range(num_classes)}
                label_dict = {}
                for i, lbl in enumerate(labels):
                    label_loc_map[lbl].append(i)
                    label_dict[i] = lbl

                self.subject_label_loc[subject_name] = label_loc_map
                self.all_label_dict[subject_name] = label_dict

        self.index_map = self._create_index()

    def _create_index(self) -> List[Tuple[str, int]]:
        """Create (subject_name, epoch) index mapping."""
        index_map = []
        for subject_name, num_epochs in self.subject_epoch_dict.items():
            for epoch in range(num_epochs):
                index_map.append((subject_name, epoch))
        return index_map

    def __len__(self) -> int:
        return self.total_num_epochs

    def __repr__(self) -> str:
        return f"SleepVideoDataset(mode={self.mode}, subjects={self.total_num_subjects}, epochs={self.total_num_epochs})"

    def __getitem__(self, idx: int):
        if self.mode == 'epochs':
            return self._get_item_epochs(idx)
        elif self.mode == 'double_loss':
            return self._get_item_double_loss(idx)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _get_item_epochs(self, idx: int):
        """Get single epoch sample."""
        subject_name, epoch = self.index_map[idx]
        subject_path = os.path.join(self.data_path, subject_name)

        with h5py.File(subject_path, 'r', swmr=True) as f:
            frames = f['frames'][epoch]
            label = f['labels'][epoch]

        frames = torch.tensor(frames, dtype=torch.float32)

        if self.transforms:
            for transform in self.transforms:
                frames = transform(frames)

        frames = frames.unsqueeze(0)  # Add channel dimension
        return frames, label, (subject_name, epoch)

    def _get_item_double_loss(self, idx: int):
        """Get triplet sample (anchor, positive, negative) for ISDL."""
        anchor_subject, anchor_epoch = self.index_map[idx]
        anchor_label = self.all_label_dict[anchor_subject][anchor_epoch]

        # Find valid negative labels
        label_loc_map = self.subject_label_loc[anchor_subject]
        negative_labels = [i for i in range(self.num_classes) if i != anchor_label]
        valid_negative_labels = [lbl for lbl in negative_labels if len(label_loc_map[lbl]) > 0]
        negative_label = np.random.choice(valid_negative_labels)

        # Choose positive epoch (same label, different epoch)
        positive_candidates = [ep for ep in label_loc_map[anchor_label] if ep != anchor_epoch]
        positive_epoch = np.random.choice(positive_candidates)

        # Choose negative epoch
        negative_epoch = np.random.choice(label_loc_map[negative_label])

        # Sort epochs for efficient h5 reading
        all_epochs = [anchor_epoch, positive_epoch, negative_epoch]
        all_epochs.sort()
        negative_epoch_index = all_epochs.index(negative_epoch)

        # Load frames
        subject_path = os.path.join(self.data_path, anchor_subject)
        with h5py.File(subject_path, 'r', swmr=True) as f:
            all_frames = f['frames'][all_epochs]  # (3, 150, 224, 224)

        # Reorder: anchor, positive, negative (negative was sorted)
        last_index = all_frames.shape[0] - 1
        all_frames[[negative_epoch_index, last_index]] = all_frames[[last_index, negative_epoch_index]]

        all_frames = torch.tensor(all_frames, dtype=torch.float32)

        # Apply transforms
        if self.transforms:
            for transform in self.transforms:
                if self.same_aug_for_contrastive:
                    all_frames = all_frames.view(-1, self.height, self.width)
                    all_frames = transform(all_frames)
                    all_frames = all_frames.view(3, self.num_frames_per_epoch, self.height, self.width)
                else:
                    all_frames = torch.stack([transform(ff) for ff in all_frames])

        all_frames = all_frames.unsqueeze(1)  # (3, 1, 150, 224, 224)
        all_labels = np.array([anchor_label, anchor_label, negative_label])

        return all_frames, all_labels

    def get_all_labels(self) -> List[int]:
        """Return all labels for computing class weights."""
        return self.all_labels


def build_video_loader(
    data_path: str,
    data_list: List[str],
    mode: str = 'epochs',
    batch_size: int = 2,
    transforms_config: Optional[dict] = None,
    sampler: str = 'weighted',
    num_workers: int = 8,
    proportion: float = 1.0,
    num_classes: int = 5,
    same_aug_for_contrastive: bool = False
) -> DataLoader:
    """
    Build video data loader for Stage 1 training.

    Args:
        data_path: Path to h5 data directory
        data_list: List of h5 filenames
        mode: 'epochs' or 'double_loss'
        batch_size: Batch size
        transforms_config: Transform configuration dict
        sampler: 'weighted', 'random', or None
        num_workers: Number of data loading workers
        proportion: Fraction of data to use per epoch
        num_classes: Number of classes
        same_aug_for_contrastive: Apply same augmentation to triplet samples

    Returns:
        DataLoader instance
    """
    transforms = get_transforms(transforms_config) if transforms_config else None

    dataset = SleepVideoDataset(
        data_path=data_path,
        data_list=data_list,
        mode=mode,
        transforms=transforms,
        num_classes=num_classes,
        same_aug_for_contrastive=same_aug_for_contrastive
    )
    print(f"Dataset: {dataset}")

    if sampler == 'weighted':
        all_labels = dataset.get_all_labels()
        class_count = Counter(all_labels)
        class_weights = 1.0 / torch.tensor(
            [class_count[i] for i in range(num_classes)], dtype=torch.float
        )
        sample_weights = torch.tensor(
            [class_weights[label] for label in all_labels], dtype=torch.float
        )
        sampler_obj = WeightedRandomSampler(
            sample_weights,
            int(len(all_labels) * proportion),
            replacement=True
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler_obj,
            num_workers=num_workers,
            prefetch_factor=2,
            persistent_workers=True
        )
    elif sampler == 'random':
        sampler_obj = RandomSampler(
            dataset,
            replacement=True,
            num_samples=int(len(dataset) * proportion)
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler_obj,
            num_workers=num_workers,
            prefetch_factor=2,
            persistent_workers=True
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=2,
            persistent_workers=True
        )

    return loader
