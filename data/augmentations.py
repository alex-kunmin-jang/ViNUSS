"""
Video data augmentation transforms for sleep video classification.

All transforms operate on tensors of shape (frames, height, width).
"""
import torch
import random
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class TensorRandomFlip:
    """Randomly flip frames horizontally or vertically."""
    def __init__(self, horizontal: bool = True, vertical: bool = False, p: float = 0.5):
        self.horizontal = horizontal
        self.vertical = vertical
        self.p = p

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p and self.horizontal:
            frames = torch.flip(frames, dims=[2])
        if random.random() < self.p and self.vertical:
            frames = torch.flip(frames, dims=[1])
        return frames


class TensorNormalize:
    """Normalize frames with given mean and std."""
    def __init__(self, mean: float = 0.608, std: float = 0.144):
        self.mean = mean
        self.std = std

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        return (frames / 255.0 - self.mean) / self.std


class RandomBrightnessContrast:
    """Randomly adjust brightness and contrast."""
    def __init__(self, brightness: float = 0.2, contrast: float = 0.2):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)

        frames = frames * brightness_factor
        frames = (frames - frames.mean()) * contrast_factor + frames.mean()
        frames = torch.clamp(frames, 0, 255.0)
        return frames


class TensorRandomScaling:
    """Randomly scale frames within a range, maintaining output size."""
    def __init__(self, scale_range: tuple = (0.8, 1.2)):
        assert len(scale_range) == 2 and scale_range[0] > 0 and scale_range[1] > 0
        self.scale_range = scale_range

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        num_frames, height, width = frames.shape
        scale = random.uniform(*self.scale_range)

        new_height = int(height * scale)
        new_width = int(width * scale)

        frames = frames.unsqueeze(1).float()
        resized = F.interpolate(
            frames,
            size=(new_height, new_width),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)

        if scale > 1.0:  # Center crop
            start_y = (new_height - height) // 2
            start_x = (new_width - width) // 2
            output = resized[:, start_y:start_y + height, start_x:start_x + width]
        else:  # Pad
            pad_y = (height - new_height) // 2
            pad_x = (width - new_width) // 2
            padding = (pad_x, width - new_width - pad_x, pad_y, height - new_height - pad_y)
            output = F.pad(resized, padding, mode='constant', value=0)

        return output.squeeze(1).type_as(frames)


class TensorRandomRotation:
    """Randomly rotate frames within a degree range."""
    def __init__(self, degrees: float = 30):
        if isinstance(degrees, (int, float)):
            self.degrees = (-abs(degrees), abs(degrees))
        else:
            self.degrees = degrees

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        angle = random.uniform(self.degrees[0], self.degrees[1])
        frames = frames.unsqueeze(1)  # [N, 1, H, W]
        rotated = TF.rotate(frames, angle)
        return rotated.squeeze(1)


def get_transforms(config: dict) -> list:
    """
    Build transform list from config dictionary.

    Args:
        config: Dictionary with transform parameters:
            - normalize_stat: {'mean': float, 'std': float}
            - TensorRandomBrightness: float (0 to disable)
            - TensorRandomContrast: float (0 to disable)
            - TensorRandomFlip: float (0 to disable)
            - TensorRandomRotation: float (0 to disable)
            - TensorRandomScaling: tuple or [1.0, 1.0] to disable

    Returns:
        List of transform objects
    """
    transforms = []

    # Brightness and contrast (applied before normalization)
    if config.get('TensorRandomBrightness', 0) != 0 or config.get('TensorRandomContrast', 0) != 0:
        transforms.append(RandomBrightnessContrast(
            brightness=config.get('TensorRandomBrightness', 0),
            contrast=config.get('TensorRandomContrast', 0)
        ))

    # Normalization
    normalize_stat = config.get('normalize_stat', {'mean': 0.608, 'std': 0.144})
    transforms.append(TensorNormalize(
        mean=normalize_stat['mean'],
        std=normalize_stat['std']
    ))

    # Flip
    if config.get('TensorRandomFlip', 0) != 0:
        transforms.append(TensorRandomFlip(p=config['TensorRandomFlip']))

    # Rotation
    if config.get('TensorRandomRotation', 0) != 0:
        transforms.append(TensorRandomRotation(degrees=config['TensorRandomRotation']))

    # Scaling
    scale = config.get('TensorRandomScaling', [1.0, 1.0])
    if isinstance(scale, list):
        scale = tuple(scale)
    if scale != (1.0, 1.0):
        transforms.append(TensorRandomScaling(scale_range=scale))

    return transforms
