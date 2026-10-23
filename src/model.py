"""ResNet-18 model architecture factory and layer freezing controls."""

from __future__ import annotations

import logging
from typing import Any, Dict
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def build_model(config: Dict[str, Any]) -> nn.Module:
    """Build canonical ResNet-18 model with custom binary classification head.

    Args:
        config: Parsed model_config.yaml dictionary.

    Returns:
        Configured PyTorch ResNet-18 nn.Module with (Dropout -> Linear(512, 1)) head.
    """
    model_cfg = config.get("model", {})
    arch = model_cfg.get("arch", "resnet18")
    if arch != "resnet18":
        raise ValueError(f"Strict constraint: model.arch must be 'resnet18', got '{arch}'")

    head_cfg = model_cfg.get("head", {})
    dropout_p = float(head_cfg.get("dropout", 0.3))
    in_features = int(head_cfg.get("in_features", 512))
    out_features = int(head_cfg.get("out_features", 1))

    # Load official ImageNet-1K pretrained weights
    logger.info("Instantiating torchvision.models.resnet18(weights=IMAGENET1K_V1)")
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)

    # Replace classification head with single raw logit output
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features=in_features, out_features=out_features),
    )

    return model


def configure_phase_a(model: nn.Module) -> None:
    """Configure Phase A: Freeze all backbone layers, warm up classification head only."""
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    # Lock all BatchNorm modules permanently in eval mode with no grad
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Phase A configured: fc unfrozen, backbone frozen. Trainable params: %d", trainable_count)


def configure_phase_b(model: nn.Module) -> None:
    """Configure Phase B: Unfreeze conv weights in layer3, layer4, and fc; keep all BatchNorm and layers 1-2 frozen."""
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only Conv2d weights inside layer3
    for m in model.layer3.modules():
        if isinstance(m, nn.Conv2d):
            for p in m.parameters():
                p.requires_grad = True

    # Unfreeze only Conv2d weights inside layer4
    for m in model.layer4.modules():
        if isinstance(m, nn.Conv2d):
            for p in m.parameters():
                p.requires_grad = True

    # Unfreeze classification head
    for param in model.fc.parameters():
        param.requires_grad = True

    # Lock all BatchNorm2d modules unconditionally in eval mode with no gradients
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Phase B configured: layer3 conv + layer4 conv + fc unfrozen, BatchNorm locked. Trainable params: %d",
        trainable_count,
    )


def get_phase_b_param_groups(
    model: nn.Module,
    lr_layer3: float = 1.0e-5,
    lr_layer4: float = 3.0e-5,
    lr_fc: float = 5.0e-5,
    weight_decay: float = 1.0e-3,
) -> list[dict[str, Any]]:
    """Return differential parameter groups for Phase B fine-tuning."""
    layer3_conv_params = [
        p for m in model.layer3.modules() if isinstance(m, nn.Conv2d)
        for p in m.parameters() if p.requires_grad
    ]
    layer4_conv_params = [
        p for m in model.layer4.modules() if isinstance(m, nn.Conv2d)
        for p in m.parameters() if p.requires_grad
    ]
    fc_params = [p for p in model.fc.parameters() if p.requires_grad]

    param_groups = [
        {"params": layer3_conv_params, "lr": lr_layer3, "weight_decay": weight_decay},
        {"params": layer4_conv_params, "lr": lr_layer4, "weight_decay": weight_decay},
        {"params": fc_params, "lr": lr_fc, "weight_decay": weight_decay},
    ]
    return param_groups


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Return dictionary with total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}

