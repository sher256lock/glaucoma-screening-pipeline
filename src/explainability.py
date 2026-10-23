"""PyTorch forward/backward hook implementation of Grad-CAM targeting model.layer4[-1]."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def classify_outcomes(
    y_true: Union[np.ndarray, list],
    y_score: Union[np.ndarray, list],
    threshold: float,
) -> Dict[str, np.ndarray]:
    """Classify binary predictions into outcome categories at a given threshold.

    Args:
        y_true: Binary ground-truth labels {0, 1}.
        y_score: Continuous Glaucoma Suspicion Scores in [0, 1].
        threshold: Decision threshold for positive prediction.

    Returns:
        Dictionary with boolean masks and integer indices for TP, FP, TN, and FN.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64).ravel()
    y_score_arr = np.asarray(y_score, dtype=np.float32).ravel()
    y_pred = (y_score_arr >= threshold).astype(np.int64)

    tp_mask = (y_true_arr == 1) & (y_pred == 1)
    fp_mask = (y_true_arr == 0) & (y_pred == 1)
    tn_mask = (y_true_arr == 0) & (y_pred == 0)
    fn_mask = (y_true_arr == 1) & (y_pred == 0)

    return {
        "tp": tp_mask,
        "fp": fp_mask,
        "tn": tn_mask,
        "fn": fn_mask,
        "tp_indices": np.where(tp_mask)[0],
        "fp_indices": np.where(fp_mask)[0],
        "tn_indices": np.where(tn_mask)[0],
        "fn_indices": np.where(fn_mask)[0],
    }


def generate_gradcam_overlay(
    model: nn.Module,
    input_tensor: torch.Tensor,
    original_rgb_uint8: np.ndarray,
    target_layer: Optional[nn.Module] = None,
    alpha_overlay: float = 0.40,
    upsample_size: Tuple[int, int] = (224, 224),
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Grad-CAM activation heatmap and blended RGB overlay.

    Hooks the target layer (defaulting to model.layer4[-1]), computes channel
    importance weights via global average pooling of gradients, applies ReLU,
    normalizes to [0, 1], and bilinearly interpolates to upsample_size.
    Hooks and temporary gradient states are cleanly removed in a finally block.

    Args:
        model: ResNet-18 PyTorch nn.Module.
        input_tensor: Preprocessed image tensor of shape (1, 3, 224, 224) or (3, 224, 224).
        original_rgb_uint8: Unnormalized RGB image numpy array (H, W, 3) in uint8 [0, 255].
        target_layer: PyTorch module to hook (defaults to model.layer4[-1]).
        alpha_overlay: Weight of the heatmap overlay in blending (1 - alpha for original).
        upsample_size: Output spatial resolution (H, W).

    Returns:
        Tuple of (heatmap, overlay):
            - heatmap: np.ndarray float32 of shape (224, 224) normalized to [0.0, 1.0].
            - overlay: np.ndarray uint8 of shape (224, 224, 3) with JET colormap blended onto original.
    """
    if target_layer is None:
        if not hasattr(model, "layer4") or len(model.layer4) == 0:
            raise ValueError("Model does not have a valid 'layer4' attribute.")
        target_layer = model.layer4[-1]

    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)
    elif input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
        raise ValueError(f"Expected input_tensor shape (1, C, H, W) or (C, H, W), got {input_tensor.shape}")

    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Save original requires_grad state to restore in finally block
    orig_states = {p: p.requires_grad for p in model.parameters()}

    # Ensure target_layer and downstream layers can backpropagate gradients
    for p in target_layer.parameters():
        p.requires_grad = True
    if hasattr(model, "fc"):
        for p in model.fc.parameters():
            p.requires_grad = True

    activations = []
    gradients = []

    def forward_hook(module: nn.Module, inp: Any, out: torch.Tensor) -> None:
        activations.append(out)

    def backward_hook(module: nn.Module, grad_inp: Any, grad_out: Tuple[torch.Tensor, ...]) -> None:
        gradients.append(grad_out[0])

    h_fwd = target_layer.register_forward_hook(forward_hook)
    h_bwd = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad()
        with torch.enable_grad():
            output = model(input_tensor)
            if output.shape != (1, 1):
                output = output.view(1, 1)
            # Backward pass on the raw logit
            output.backward(torch.ones_like(output))

        if not activations or not gradients:
            raise RuntimeError("Failed to capture activations or gradients from target layer.")

        act = activations[0]  # Shape: (1, 512, H_act, W_act)
        grad = gradients[0]   # Shape: (1, 512, H_act, W_act)

        # Global average pooling of gradients: alpha_k = (1 / (H*W)) sum_{i,j} grad_{i,j}
        weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # (1, 512, 1, 1)

        # Linear combination of feature maps weighted by alpha_k
        cam = torch.sum(weights * act, dim=1, keepdim=True)   # (1, 1, H_act, W_act)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if (cam_max - cam_min) > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        # Bilinear interpolation upsampling
        cam_upsampled = F.interpolate(
            cam,
            size=upsample_size,
            mode="bilinear",
            align_corners=False,
        )
        heatmap = cam_upsampled.squeeze().detach().cpu().numpy().astype(np.float32)

        # Ensure original image is correctly sized and formatted
        h_out, w_out = upsample_size
        if original_rgb_uint8.shape[:2] != (h_out, w_out):
            orig_resized = cv2.resize(original_rgb_uint8, (w_out, h_out), interpolation=cv2.INTER_LINEAR)
        else:
            orig_resized = original_rgb_uint8.copy()

        # Colormap mapping (JET)
        heatmap_uint8 = np.uint8(255.0 * np.clip(heatmap, 0.0, 1.0))
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Alpha blend: (1 - alpha) * original + alpha * heatmap
        alpha = float(alpha_overlay)
        overlay = cv2.addWeighted(orig_resized, 1.0 - alpha, heatmap_rgb, alpha, 0)

        return heatmap, overlay

    finally:
        h_fwd.remove()
        h_bwd.remove()
        for p, state in orig_states.items():
            p.requires_grad = state
        model.zero_grad()
