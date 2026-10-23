"""OpenCV heuristic quality checks: Laplacian variance blur warning and mean luminance warnings."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def check_blur(
    image_cv2: np.ndarray,
    blur_threshold: float = 50.0,
) -> List[str]:
    """Check for severe image blur using Laplacian variance.

    Args:
        image_cv2: Image as numpy array (BGR, RGB, or Grayscale).
        blur_threshold: Minimum acceptable Laplacian variance.

    Returns:
        List of human-readable warning strings (empty if passing).
    """
    if image_cv2 is None or image_cv2.size == 0:
        return ["Empty image passed to blur check."]

    if image_cv2.ndim == 3:
        gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_cv2

    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    warnings = []
    if laplacian_var < blur_threshold:
        warnings.append(
            f"Blurry fundus image detected (Laplacian variance: {laplacian_var:.1f} < {blur_threshold:.1f}). "
            "Interpret results with caution."
        )
    return warnings


def check_illumination(
    image_cv2: np.ndarray,
    min_brightness: float = 40.0,
    max_brightness: float = 220.0,
) -> List[str]:
    """Check for severe under-exposure or over-exposure using mean pixel intensity.

    Args:
        image_cv2: Image as numpy array (BGR, RGB, or Grayscale).
        min_brightness: Minimum acceptable mean intensity.
        max_brightness: Maximum acceptable mean intensity.

    Returns:
        List of human-readable warning strings (empty if passing).
    """
    if image_cv2 is None or image_cv2.size == 0:
        return ["Empty image passed to illumination check."]

    if image_cv2.ndim == 3:
        gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_cv2

    mean_brightness = float(np.mean(gray))
    warnings = []
    if mean_brightness < min_brightness:
        warnings.append(
            f"Low illumination detected (mean intensity: {mean_brightness:.1f} < {min_brightness:.1f}). "
            "Verify fundus exposure."
        )
    elif mean_brightness > max_brightness:
        warnings.append(
            f"Severe over-exposure detected (mean intensity: {mean_brightness:.1f} > {max_brightness:.1f}). "
            "Image may be washed out."
        )
    return warnings


def run_quality_checks(
    image_cv2: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Execute all non-blocking quality heuristics against model_config.yaml constraints.

    Args:
        image_cv2: Image as numpy array (BGR, RGB, or Grayscale).
        config: Optional parsed model_config.yaml dictionary.

    Returns:
        List of warning strings (empty if all quality checks pass).
    """
    qc_cfg = (config or {}).get("quality_checks", {})
    min_w = int(qc_cfg.get("min_width", 64))
    min_h = int(qc_cfg.get("min_height", 64))
    blur_min = float(qc_cfg.get("blur_laplacian_min", 50.0))
    bright_min = float(qc_cfg.get("brightness_min", 40.0))
    bright_max = float(qc_cfg.get("brightness_max", 220.0))

    warnings: List[str] = []

    if image_cv2 is None or image_cv2.size == 0:
        return ["Invalid or empty image provided."]

    h, w = image_cv2.shape[:2]
    if w < min_w or h < min_h:
        warnings.append(
            f"Low resolution image detected ({w}x{h} < {min_w}x{min_h}). "
            "Downsampling artifacts may occur."
        )

    warnings.extend(check_blur(image_cv2, blur_threshold=blur_min))
    warnings.extend(check_illumination(image_cv2, min_brightness=bright_min, max_brightness=bright_max))

    return warnings
