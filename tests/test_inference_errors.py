"""Unit tests covering inference engine error handling, corrupt image bytes, and checkpoint integrity."""

import io
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from src.inference import (
    CorruptImageError,
    GlaucomaInferenceEngine,
    InferenceError,
    MetadataContractError,
    ModelCheckpointError,
)
from src.quality_checks import check_blur, check_illumination, run_quality_checks


@pytest.fixture(scope="module")
def engine():
    """Instantiate a valid GlaucomaInferenceEngine using project checkpoints."""
    return GlaucomaInferenceEngine()


@pytest.fixture(scope="module")
def sample_valid_image_bytes():
    """Generate in-memory valid fundus-like JPEG bytes."""
    img = Image.new("RGB", (224, 224), color=(120, 80, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestEngineInstantiationFailures:
    """Verify engine instantiation detects missing or malformed configuration and checkpoints."""

    def test_missing_weights_file_raises_checkpoint_error(self, tmp_path):
        """Simulate missing .pth weights file."""
        fake_weights = tmp_path / "nonexistent_model.pth"
        with pytest.raises(ModelCheckpointError) as exc_info:
            GlaucomaInferenceEngine(weights_path=fake_weights)
        assert "not found" in str(exc_info.value).lower()

    def test_missing_metadata_file_raises_checkpoint_error(self, tmp_path):
        """Simulate missing .meta.json file."""
        fake_meta = tmp_path / "nonexistent_meta.json"
        with pytest.raises(ModelCheckpointError) as exc_info:
            GlaucomaInferenceEngine(metadata_path=fake_meta)
        assert "not found" in str(exc_info.value).lower()

    def test_missing_config_file_raises_checkpoint_error(self, tmp_path):
        """Simulate missing config YAML file."""
        fake_cfg = tmp_path / "nonexistent_config.yaml"
        with pytest.raises(ModelCheckpointError) as exc_info:
            GlaucomaInferenceEngine(config_path=fake_cfg)
        assert "not found" in str(exc_info.value).lower()


class TestCorruptAndInvalidInputs:
    """Verify predict() handles malformed, corrupt, or empty image bytes gracefully."""

    def test_empty_bytes_raises_corrupt_image_error(self, engine):
        """Passing empty bytes must raise CorruptImageError."""
        with pytest.raises(CorruptImageError) as exc_info:
            engine.predict(b"")
        assert "empty" in str(exc_info.value).lower()

    def test_random_corrupt_bytes_raises_corrupt_image_error(self, engine):
        """Passing random binary noise must raise CorruptImageError."""
        corrupt_bytes = b"This is not a valid JPEG or PNG header \x00\xff\xfe\x00\x12"
        with pytest.raises(CorruptImageError) as exc_info:
            engine.predict(corrupt_bytes)
        assert "decode" in str(exc_info.value).lower()

    def test_truncated_bytes_raises_corrupt_image_error(self, engine):
        """Passing truncated incomplete JPEG bytes must raise CorruptImageError."""
        truncated_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        with pytest.raises(CorruptImageError):
            engine.predict(truncated_bytes)


class TestSuccessfulInferenceFlow:
    """Verify correct prediction payload schema and score bounds."""

    def test_valid_image_inference_returns_contract_schema(self, engine, sample_valid_image_bytes):
        """Inference on valid bytes must return full contract schema with score in [0, 1]."""
        res = engine.predict(sample_valid_image_bytes, include_gradcam=True)

        assert "reference_id" in res and res["reference_id"].startswith("req_")
        assert res["predicted_class"] in ["Glaucoma", "Normal"]
        assert "suspicion_score" in res
        assert isinstance(res["suspicion_score"], float)
        assert 0.0 <= res["suspicion_score"] <= 1.0

        assert "threshold_used" in res
        assert res["threshold_used"] == engine.threshold

        assert "latency_ms" in res
        assert res["latency_ms"] > 0

        assert "quality_warnings" in res
        assert isinstance(res["quality_warnings"], list)

        assert "gradcam_overlay_base64" in res
        assert len(res["gradcam_overlay_base64"]) > 0

        assert "disclaimer" in res
        assert "RESEARCH PROTOTYPE ONLY" in res["disclaimer"]
        assert "Glaucoma Suspicion Score" in res["disclaimer"]


class TestQualityChecksHeuristics:
    """Verify OpenCV blur and illumination heuristics."""

    def test_severely_blurred_image_triggers_warning(self):
        """Homogeneous constant image has near-zero Laplacian variance and must trigger blur warning."""
        flat_img = np.full((224, 224, 3), 128, dtype=np.uint8)
        warnings = check_blur(flat_img, blur_threshold=50.0)
        assert len(warnings) > 0
        assert "blurry" in warnings[0].lower()

    def test_severely_dark_image_triggers_illumination_warning(self):
        """Severely underexposed image must trigger low illumination warning."""
        dark_img = np.full((224, 224, 3), 10, dtype=np.uint8)
        warnings = check_illumination(dark_img, min_brightness=40.0, max_brightness=220.0)
        assert len(warnings) > 0
        assert "low illumination" in warnings[0].lower()

    def test_severely_bright_image_triggers_illumination_warning(self):
        """Severely overexposed image must trigger over-exposure warning."""
        bright_img = np.full((224, 224, 3), 245, dtype=np.uint8)
        warnings = check_illumination(bright_img, min_brightness=40.0, max_brightness=220.0)
        assert len(warnings) > 0
        assert "over-exposure" in warnings[0].lower()
