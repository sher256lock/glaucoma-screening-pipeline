"""Pure CPU inference engine enforcing single-image batching, thread pinning, and threshold loading."""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import psutil
import torch
import torch.nn as nn
import yaml

from src.explainability import generate_gradcam_overlay
from src.model import build_model
from src.preprocessing import get_transforms
from src.quality_checks import run_quality_checks

logger = logging.getLogger(__name__)


# ==============================================================================
# Structured Exceptions
# ==============================================================================

class InferenceError(Exception):
    """Base exception for all inference engine errors."""
    pass


class ModelCheckpointError(InferenceError):
    """Raised when checkpoint weights or metadata cannot be located or loaded."""
    pass


class MetadataContractError(InferenceError):
    """Raised when metadata schema or threshold violates architectural contracts."""
    pass


class CorruptImageError(InferenceError):
    """Raised when uploaded image bytes cannot be decoded into a valid image."""
    pass


# ==============================================================================
# GlaucomaInferenceEngine
# ==============================================================================

class GlaucomaInferenceEngine:
    """Production CPU inference engine for binary Glaucoma screening using ResNet-18."""

    def __init__(
        self,
        config_path: Union[str, Path] = "configs/model_config.yaml",
        weights_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
    ):
        self.config_path = Path(config_path).resolve()
        if not self.config_path.exists():
            raise ModelCheckpointError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Determine and validate checkpoint paths
        ckpt_cfg = self.config.get("checkpoint", {})
        self.weights_path = Path(weights_path or ckpt_cfg.get("weights_path", "artifacts/models/resnet18_refuge.pth")).resolve()
        self.metadata_path = Path(metadata_path or ckpt_cfg.get("metadata_path", "artifacts/models/resnet18_refuge.meta.json")).resolve()

        if not self.weights_path.exists():
            raise ModelCheckpointError(f"Model weights file not found on disk: {self.weights_path}")
        if not self.metadata_path.exists():
            raise ModelCheckpointError(f"Metadata contract file not found on disk: {self.metadata_path}")

        # Pin physical CPU threads for deterministic low-latency intra-op inference
        threads_cfg = self.config.get("inference", {}).get("threads", "physical_cores")
        if threads_cfg == "physical_cores":
            physical_threads = psutil.cpu_count(logical=False) or 4
        else:
            physical_threads = int(threads_cfg)
        torch.set_num_threads(physical_threads)
        self.threads = physical_threads

        # Load and validate metadata contract
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        except Exception as e:
            raise MetadataContractError(f"Failed to parse metadata JSON at {self.metadata_path}: {e}") from e

        self.threshold = self.metadata.get("threshold")
        if self.threshold is None or not isinstance(self.threshold, (int, float)):
            raise MetadataContractError(
                f"Frozen decision threshold missing or invalid in {self.metadata_path}. Got: {self.threshold}"
            )
        self.threshold = float(self.threshold)
        if not (0.0 < self.threshold < 1.0):
            raise MetadataContractError(f"Threshold must be strictly in (0.0, 1.0), got {self.threshold}")

        self.threshold_procedure = str(self.metadata.get("threshold_procedure", "youden_j_on_val"))
        self.model_version = f"{self.config.get('project', {}).get('name', 'resnet18')}-{self.metadata.get('git_sha', 'v1')}"
        self.disclaimer = str(self.config.get("project", {}).get(
            "disclaimer",
            "RESEARCH PROTOTYPE ONLY. Glaucoma Suspicion Score is not a calibrated probability and is not for clinical diagnosis."
        ))

        # Instantiate CPU ResNet-18 and load weights
        self.device = torch.device("cpu")
        try:
            self.model = build_model(self.config).to(self.device)
            state_dict = torch.load(self.weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        except Exception as e:
            raise ModelCheckpointError(f"Failed to load ResNet-18 weights into architecture: {e}") from e

        # Build inference transform pipeline (single-source CLAHE + resize + ImageNet normalization)
        self.transform = get_transforms(self.config, is_training=False)
        logger.info(
            f"GlaucomaInferenceEngine initialized successfully on CPU ({self.threads} threads). "
            f"Threshold: {self.threshold:.4f} ({self.threshold_procedure})"
        )

    def predict(
        self,
        image_bytes: bytes,
        include_gradcam: bool = True,
    ) -> Dict[str, Any]:
        """Perform end-to-end in-memory inference from raw image bytes.

        Args:
            image_bytes: Raw bytes of uploaded fundus photograph (JPEG, PNG, etc.).
            include_gradcam: Whether to generate and return base64 Grad-CAM overlay.

        Returns:
            Dictionary matching FastAPI prediction response contract.
        """
        if not image_bytes or len(image_bytes) == 0:
            raise CorruptImageError("Empty image bytes payload provided.")

        t_start = time.perf_counter()
        reference_id = f"req_{uuid.uuid4().hex[:8]}"

        # 1. Decode bytes in memory
        try:
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img_cv2 is None:
                raise CorruptImageError("Failed to decode image bytes into valid OpenCV image array.")
        except Exception as e:
            raise CorruptImageError(f"Image decode failed: {e}") from e

        # Convert OpenCV BGR to RGB for PIL and Grad-CAM
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # 2. Execute non-blocking quality checks
        quality_warnings = run_quality_checks(img_cv2, self.config)

        # 3. Apply single-source preprocessing
        try:
            tensor_img = self.transform(pil_img)  # Shape: (3, 224, 224)
            batch_tensor = tensor_img.unsqueeze(0).to(self.device)  # Shape: (1, 3, 224, 224)
        except Exception as e:
            raise InferenceError(f"Preprocessing transform failed: {e}") from e

        # 4. Pure CPU forward pass
        with torch.no_grad():
            logit = self.model(batch_tensor)
            suspicion_score = float(torch.sigmoid(logit).squeeze().item())

        predicted_class = "Glaucoma" if suspicion_score >= self.threshold else "Normal"

        # 5. Optional Grad-CAM overlay generation
        gradcam_b64 = ""
        if include_gradcam:
            try:
                heatmap, overlay = generate_gradcam_overlay(
                    model=self.model,
                    input_tensor=batch_tensor,
                    original_rgb_uint8=img_rgb,
                    alpha_overlay=float(self.config.get("gradcam", {}).get("alpha_overlay", 0.40)),
                )
                # Encode RGB overlay to PNG bytes in memory
                overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                success, buf = cv2.imencode(".png", overlay_bgr)
                if success:
                    gradcam_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            except Exception as e:
                logger.warning(f"Grad-CAM generation failed for {reference_id}: {e}")
                quality_warnings.append(f"Grad-CAM generation failed: {e}")

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "reference_id": reference_id,
            "predicted_class": predicted_class,
            "class": predicted_class,  # Compatibility alias
            "suspicion_score": round(suspicion_score, 4),
            "threshold_used": round(self.threshold, 4),
            "threshold_procedure": self.threshold_procedure,
            "latency_ms": round(latency_ms, 1),
            "quality_warnings": quality_warnings,
            "gradcam_overlay_base64": gradcam_b64,
            "gradcam_base64": gradcam_b64,  # Compatibility alias
            "model_version": self.model_version,
            "disclaimer": self.disclaimer,
        }


# Alias for orchestration compatibility
GlaucomaPredictor = GlaucomaInferenceEngine
