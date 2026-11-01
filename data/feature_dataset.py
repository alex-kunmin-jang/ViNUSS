"""
Feature dataset for Stage 2 training (SDM).

Data format: h5 files with keys ['features', 'labels']
- features: (epoch, feature_dim=640, time_stamp=50, height=1, width=1)
- labels: (epoch,)

Mode:
- subject: Load entire subject's features at once for temporal modeling
"""
import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from typing import Optional, List, Tuple


class FeatureDataset(Dataset):
    """
    One night (all epochs of one subject) per sample.

    Args:
        data_path: Path to directory containing feature h5 files
        data_list: List of h5 filenames to include
        mode: 'subject' (load all epochs per subject)
        num_classes: Number of sleep stage classes
    """
    def __init__(
        self,
        data_path: str,
        data_list: List[str],
        mode: str = 'subject',
        num_classes: int = 5,
        **kwargs
    ):
        self.data_path = data_path
        self.data_list = data_list
        self.mode = mode
        self.num_classes = num_classes

        if data_list is None:
            raise ValueError("data_list cannot be None")

        self.total_num_epochs = 0
        self.total_num_subjects = len(data_list)
        self.subject_epoch_dict = {}

        print("Loading feature metadata...")
        for subject_name in data_list:
            subject_path = os.path.join(data_path, subject_name)
            with h5py.File(subject_path, 'r', swmr=True) as f:
                num_epochs = f['features'].shape[0]
                self.total_num_epochs += num_epochs
                self.subject_epoch_dict[subject_name] = num_epochs

        # Get feature shape from last subject
        with h5py.File(subject_path, 'r', swmr=True) as f:
            shape = f['features'].shape
            shape_len = len(shape)

            self.feature_dim = shape[1] if shape_len > 1 else None
            self.time_stamp = shape[2] if shape_len > 2 else None
            self.height = shape[3] if shape_len > 3 else None
            self.width = shape[4] if shape_len > 4 else None

        # Print shape info
        shape_str = f"(epoch, feature_dim={self.feature_dim}"
        if self.time_stamp is not None:
            shape_str += f", time_stamp={self.time_stamp}"
        if self.height is not None:
            shape_str += f", height={self.height}"
        if self.width is not None:
            shape_str += f", width={self.width}"
        shape_str += ")"
        print(f"Feature Shape: {shape_str}")

        self.index_map = self._create_index()

    def _create_index(self) -> List[Tuple[str, int]]:
        """Create (subject_name, num_epochs) index mapping."""
        index_map = []
        for subject_name, num_epochs in self.subject_epoch_dict.items():
            index_map.append((subject_name, num_epochs))
        return index_map

    def __len__(self) -> int:
        return self.total_num_subjects

    def __repr__(self) -> str:
        return (
            f"FeatureDataset(subjects={self.total_num_subjects}, "
            f"epochs={self.total_num_epochs}, mode={self.mode}, "
            f"feature_dim={self.feature_dim})"
        )

    def __getitem__(self, idx: int):
        if self.mode != 'subject':
            raise NotImplementedError(f"Mode '{self.mode}' not implemented")

        subject_name, num_epochs = self.index_map[idx]
        subject_path = os.path.join(self.data_path, subject_name)

        with h5py.File(subject_path, 'r', swmr=True) as f:
            features = f['features'][:]  # (epoch, 640, 50, 1, 1)
            labels = f['labels'][:]      # (epoch,)

        features = torch.tensor(features, dtype=torch.float32)
        return features, labels, subject_name


def build_feature_loader(
    data_path: str,
    data_list: List[str],
    mode: str = 'subject',
    batch_size: int = 1,
    shuffle: bool = False,
    sampler: Optional[str] = None,
    num_classes: int = 5,
    proportion: float = 1.0,
    num_workers: int = 4
) -> DataLoader:
    """
    Build feature data loader for Stage 2 training.

    Args:
        data_path: Path to feature h5 directory
        data_list: List of h5 filenames
        mode: 'subject' (load all epochs per subject)
        batch_size: Batch size (typically 1 for subject-level loading)
        shuffle: Whether to shuffle data
        sampler: 'random' or None
        num_classes: Number of classes
        proportion: Fraction of subjects to use
        num_workers: Number of data loading workers

    Returns:
        DataLoader instance
    """
    if mode != 'subject':
        raise ValueError(f"Mode must be 'subject', got '{mode}'")

    dataset = FeatureDataset(
        data_path=data_path,
        data_list=data_list,
        mode=mode,
        num_classes=num_classes
    )
    print(f"Dataset: {dataset}")

    if sampler == 'random':
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
            prefetch_factor=2
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=2
        )

    return loader
