"""PyTorch Dataset implementation for loading REFUGE retinal fundus images."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
import torch
from torch.utils.data import Dataset


class RefugeDataset(Dataset):
    """PyTorch Dataset loading REFUGE fundus photographs from split CSV manifests."""

    def __init__(
        self,
        csv_path: Union[str, Path],
        raw_dir: Union[str, Path] = Path("data/raw/refuge"),
        transform: Optional[Callable] = None,
        config: Optional[Dict[str, Any]] = None,
        is_training: bool = False,
    ):
        self.csv_path = Path(csv_path).resolve()
        self.raw_dir = Path(raw_dir).resolve()
        self.is_training = is_training

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Split CSV manifest not found: {self.csv_path}")
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw dataset directory not found: {self.raw_dir}")

        if transform is not None:
            self.transform = transform
        elif config is not None:
            from src.preprocessing import get_transforms
            self.transform = get_transforms(config, is_training=self.is_training)
        else:
            raise ValueError("Must provide either 'transform' callable or 'config' dictionary.")

        self.records: List[Dict[str, str]] = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.records = list(reader)

        if not self.records:
            raise ValueError(f"No records found in manifest: {self.csv_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[idx]
        rel_path = record["image_path"]
        full_path = self.raw_dir / rel_path

        if not full_path.exists():
            raise FileNotFoundError(f"Image not found on disk: {full_path}")

        with Image.open(full_path) as pil_img:
            img_rgb = pil_img.convert("RGB")

        tensor_img = self.transform(img_rgb)
        label_float = float(record["label_binary"])
        label_tensor = torch.tensor([label_float], dtype=torch.float32)

        return tensor_img, label_tensor, rel_path
