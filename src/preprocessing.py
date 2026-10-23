"""Deterministic image preprocessing and tensor transforms driven by model_config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.v2 as transforms
from torchvision.transforms import InterpolationMode
import yaml


class ApplyCLAHE:
    """Config-driven CLAHE transform for PIL / numpy images (L-channel in LAB)."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        np_img = np.array(img)
        # Convert RGB to LAB
        lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return Image.fromarray(final_rgb)


def get_transforms(config: Dict[str, Any], is_training: bool = False) -> transforms.Compose:
    """Build torchvision transform pipeline strictly adhering to configs/model_config.yaml.

    Args:
        config: Parsed model_config.yaml dictionary.
        is_training: If True, include training augmentations (flip, rotation, jitter).

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    input_cfg = config.get("input", {})
    size = int(input_cfg.get("size", 224))
    norm_cfg = input_cfg.get("normalization", {})
    mean = norm_cfg.get("mean", [0.485, 0.456, 0.406])
    std = norm_cfg.get("std", [0.229, 0.224, 0.225])

    interp_name = str(input_cfg.get("interpolation", "bilinear")).lower()
    interp_mode = InterpolationMode.BILINEAR if interp_name == "bilinear" else InterpolationMode.NEAREST

    clahe_cfg = input_cfg.get("clahe", {})
    if isinstance(clahe_cfg, bool):
        clahe_enabled = clahe_cfg
        clip_limit = 2.0
        grid = (8, 8)
    elif isinstance(clahe_cfg, dict):
        clahe_enabled = bool(clahe_cfg.get("enabled", False))
        clip_limit = float(clahe_cfg.get("clip_limit", 2.0))
        grid = tuple(clahe_cfg.get("tile_grid_size", [8, 8]))
    else:
        clahe_enabled = False
        clip_limit = 2.0
        grid = (8, 8)

    transform_list: list = []

    # Optional CLAHE (must be identical in train and inference if enabled)
    if clahe_enabled:
        transform_list.append(ApplyCLAHE(clip_limit=clip_limit, tile_grid_size=grid))

    # Spatial resize
    transform_list.append(transforms.Resize((size, size), interpolation=interp_mode, antialias=True))

    if is_training:
        aug_cfg = config.get("training", {}).get("augmentation", {})
        hflip_p = float(aug_cfg.get("random_horizontal_flip", 0.5))
        vflip_p = float(aug_cfg.get("random_vertical_flip", 0.5))
        rot_deg = float(aug_cfg.get("random_rotation_degrees", 15.0))

        if hflip_p > 0:
            transform_list.append(transforms.RandomHorizontalFlip(p=hflip_p))
        if vflip_p > 0:
            transform_list.append(transforms.RandomVerticalFlip(p=vflip_p))
        if rot_deg > 0:
            transform_list.append(transforms.RandomRotation(degrees=rot_deg))

    # Convert to Tensor and normalize
    transform_list.append(transforms.ToImage())
    transform_list.append(transforms.ToDtype(torch.float32, scale=True))
    transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)


def preprocess_image(
    image: Union[Image.Image, np.ndarray, Path, str],
    config: Dict[str, Any],
) -> torch.Tensor:
    """Preprocess a single retinal fundus image for inference, returning a (1, 3, 224, 224) float32 tensor."""
    if isinstance(image, (str, Path)):
        p = Path(image)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        pil_img = Image.open(p).convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        pil_img = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    transform = get_transforms(config, is_training=False)
    tensor = transform(pil_img)  # Shape: (3, 224, 224)
    return tensor.unsqueeze(0)  # Shape: (1, 3, 224, 224)


# Re-export RefugeDataset for direct access from src.preprocessing
from src.data.dataset import RefugeDataset

__all__ = [
    "ApplyCLAHE",
    "get_transforms",
    "preprocess_image",
    "RefugeDataset",
]
