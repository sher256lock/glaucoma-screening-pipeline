"""Unit tests verifying checkpoint weights loading, forward pass, and metadata schema contract."""

import json
from pathlib import Path
import pytest
import torch
import yaml

from src.model import build_model, configure_phase_a, configure_phase_b, count_parameters

CONFIG_PATH = Path("configs/model_config.yaml")
WEIGHTS_PATH = Path("artifacts/models/resnet18_refuge.pth")
META_PATH = Path("artifacts/models/resnet18_refuge.meta.json")


@pytest.fixture(scope="module")
def config():
    assert CONFIG_PATH.exists()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestModelArchitectureAndFreezing:
    """Verify ResNet-18 layer configurations and Phase A / B freeze policies."""

    def test_build_model_structure(self, config):
        """Verify model head replacement and output logit dimension."""
        model = build_model(config)
        assert hasattr(model, "fc")
        assert len(model.fc) == 2  # Dropout(0.3) -> Linear(512, 1)
        assert isinstance(model.fc[0], torch.nn.Dropout)
        assert isinstance(model.fc[1], torch.nn.Linear)
        assert model.fc[1].in_features == 512
        assert model.fc[1].out_features == 1

    def test_configure_phase_a_freezing(self, config):
        """Verify Phase A unfreezes only classification head (513 params)."""
        model = build_model(config)
        configure_phase_a(model)

        counts = count_parameters(model)
        assert counts["trainable"] == 513, f"Phase A expected 513 trainable params, got {counts['trainable']}"
        assert not model.conv1.weight.requires_grad
        assert not model.layer1[0].conv1.weight.requires_grad
        assert not model.layer4[0].conv1.weight.requires_grad
        assert model.fc[1].weight.requires_grad
        assert model.fc[1].bias.requires_grad

    def test_configure_phase_b_freezing(self, config):
        """Verify Phase B unfreezes conv weights in layer3, layer4, and fc, keeping bn and layers 1-2 frozen."""
        model = build_model(config)
        configure_phase_b(model)

        assert not model.conv1.weight.requires_grad
        assert not model.layer1[0].conv1.weight.requires_grad
        assert not model.layer2[0].conv1.weight.requires_grad

        # layer3 conv weights unfrozen, bn frozen
        assert model.layer3[0].conv1.weight.requires_grad
        assert not model.layer3[0].bn1.weight.requires_grad

        # layer4 conv weights unfrozen, bn frozen
        assert model.layer4[0].conv1.weight.requires_grad
        assert not model.layer4[0].bn1.weight.requires_grad

        assert model.fc[1].weight.requires_grad
        assert model.fc[1].bias.requires_grad


class TestCheckpointSerializationAndInference:
    """Verify saved checkpoint weights and metadata contracts."""

    def test_checkpoint_weights_load_and_forward(self, config):
        """Verify resnet18_refuge.pth loads cleanly on CPU and runs forward pass."""
        if not WEIGHTS_PATH.exists():
            pytest.skip("Model weights not yet trained (run train.py first).")

        model = build_model(config)
        state_dict = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        dummy_batch = torch.randn(2, 3, 224, 224, dtype=torch.float32)
        with torch.no_grad():
            logits = model(dummy_batch)

        assert logits.shape == (2, 1), f"Expected forward shape (2, 1), got {logits.shape}"
        assert not torch.isnan(logits).any()

    def test_metadata_contract_schema(self):
        """Verify resnet18_refuge.meta.json adheres strictly to architectural contracts."""
        if not META_PATH.exists():
            pytest.skip("Model metadata not yet generated (run train.py first).")

        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)

        required_keys = {
            "arch",
            "framework",
            "pytorch_version",
            "input_size",
            "channels",
            "normalization",
            "clahe",
            "dataset",
            "dataset_version",
            "usable_n",
            "split_protocol",
            "pos_weight",
            "threshold",
            "threshold_procedure",
            "val_auroc",
            "val_sensitivity_at_threshold",
            "val_specificity_at_threshold",
            "git_sha",
            "config_hash",
            "best_epoch",
            "training_phase",
            "created_at",
        }
        missing = required_keys - set(meta.keys())
        assert not missing, f"Missing required metadata keys: {missing}"

        assert meta["arch"] == "resnet18"
        assert meta["framework"] == "pytorch"
        assert meta["input_size"] == 224
        assert meta["channels"] == 3
        assert meta["dataset"] == "REFUGE"
        assert meta["pos_weight"] == 9.0
        assert meta["clahe"] is True
        assert meta["val_auroc"] > 0.85, f"Validation AUROC ({meta['val_auroc']}) below required target (>0.85)"
