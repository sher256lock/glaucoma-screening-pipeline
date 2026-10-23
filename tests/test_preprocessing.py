"""Unit tests verifying single-source image preprocessing and tensor transforms."""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from src.preprocessing import get_transforms, preprocess_image, ApplyCLAHE
from src.data.dataset import RefugeDataset

CONFIG_PATH = Path("configs/model_config.yaml")


@pytest.fixture(scope="module")
def config():
    assert CONFIG_PATH.exists()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def dummy_rgb_image():
    """Create a temporary 512x512 RGB synthetic PIL image."""
    arr = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_validation_transform_shape_and_dtype(config, dummy_rgb_image):
    """Verify inference transform produces strictly (3, 224, 224) float32 tensor."""
    tf = get_transforms(config, is_training=False)
    tensor = tf(dummy_rgb_image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224), f"Expected shape (3, 224, 224), got {tensor.shape}"
    assert tensor.dtype == torch.float32, f"Expected float32, got {tensor.dtype}"


def test_deterministic_inference_consistency(config, dummy_rgb_image):
    """Verify that inference transforms are strictly deterministic across consecutive calls."""
    tf = get_transforms(config, is_training=False)
    t1 = tf(dummy_rgb_image)
    t2 = tf(dummy_rgb_image)

    assert torch.equal(t1, t2), "Inference preprocessing must be strictly deterministic across calls."


def test_training_transform_shape_and_dtype(config, dummy_rgb_image):
    """Verify training transform produces (3, 224, 224) float32 tensor."""
    tf = get_transforms(config, is_training=True)
    tensor = tf(dummy_rgb_image)

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_preprocess_image_single_item(config, dummy_rgb_image):
    """Verify preprocess_image adds batch dimension for model forward pass."""
    batch_tensor = preprocess_image(dummy_rgb_image, config)
    assert batch_tensor.shape == (1, 3, 224, 224)
    assert batch_tensor.dtype == torch.float32


def test_clahe_transform(dummy_rgb_image):
    """Verify ApplyCLAHE operates on RGB images returning PIL.Image."""
    clahe_op = ApplyCLAHE(clip_limit=2.0, tile_grid_size=(8, 8))
    out = clahe_op(dummy_rgb_image)
    assert isinstance(out, Image.Image)
    assert out.size == dummy_rgb_image.size


def test_dataset_item_schema(config):
    """Verify RefugeDataset emits (image_tensor, label_tensor, rel_path)."""
    train_csv = Path("data/processed/splits/train.csv")
    if not train_csv.exists():
        pytest.skip("train.csv not yet generated.")

    dataset = RefugeDataset(train_csv, config=config, is_training=False)
    assert len(dataset) == 400

    img_tensor, label_tensor, rel_path = dataset[0]
    assert img_tensor.shape == (3, 224, 224)
    assert img_tensor.dtype == torch.float32
    assert label_tensor.shape == (1,)
    assert label_tensor.dtype == torch.float32
    assert isinstance(rel_path, str)
    assert label_tensor.item() in {0.0, 1.0}
