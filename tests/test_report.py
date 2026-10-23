"""Unit tests verifying in-memory clinical PDF report generation and endpoint delivery."""

import io
from pathlib import Path
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.report import generate_clinical_pdf


@pytest.fixture(scope="module")
def sample_prediction():
    """Sample GlaucomaInferenceEngine output dictionary."""
    return {
        "reference_id": "req_test_8f92a1c4",
        "predicted_class": "Glaucoma",
        "suspicion_score": 0.8142,
        "threshold_used": 0.2921,
        "threshold_procedure": "youden_j_on_val",
        "latency_ms": 32.5,
        "quality_warnings": [
            "Blurry fundus image detected (Laplacian variance: 42.1 < 50.0). Interpret results with caution."
        ],
        "gradcam_overlay_base64": "",
        "model_version": "resnet18-clean",
        "disclaimer": "RESEARCH PROTOTYPE ONLY. Glaucoma Suspicion Score is not a calibrated probability and is not for clinical diagnosis.",
    }


@pytest.fixture(scope="module")
def sample_fundus_bytes():
    """Create in-memory synthetic fundus image bytes."""
    img = Image.new("RGB", (224, 224), color=(180, 80, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestClinicalPDFReport:
    """Verify in-memory PDF generation, magic byte contracts, and PHI compliance."""

    def test_generate_clinical_pdf_magic_bytes_and_stream(
        self,
        sample_prediction,
        sample_fundus_bytes,
        tmp_path,
        monkeypatch,
    ):
        """Verify generate_clinical_pdf returns valid PDF bytes in BytesIO without touching disk."""
        # Ensure no file writes occur in current working directory
        cwd_before = set(Path.cwd().glob("*"))

        pdf_stream = generate_clinical_pdf(
            prediction_result=sample_prediction,
            original_image=sample_fundus_bytes,
        )

        cwd_after = set(Path.cwd().glob("*"))
        assert cwd_before == cwd_after, "generate_clinical_pdf must not write temporary files to disk!"

        assert isinstance(pdf_stream, io.BytesIO)
        pdf_bytes = pdf_stream.getvalue()

        # Check %PDF magic bytes
        assert pdf_bytes.startswith(b"%PDF"), f"Expected %PDF magic header, got: {pdf_bytes[:10]}"
        assert len(pdf_bytes) > 2000, f"Expected non-trivial PDF size, got {len(pdf_bytes)} bytes"

        # Check EOF trailer marker
        assert b"%%EOF" in pdf_bytes[-1024:], "Expected valid %%EOF trailer in PDF structure"

    def test_api_report_endpoint_returns_streaming_pdf(self, sample_fundus_bytes):
        """POST /api/v1/report must stream application/pdf with %PDF header in-memory."""
        with TestClient(app) as client:
            files = {"file": ("fundus_test.jpg", sample_fundus_bytes, "image/jpeg")}
            resp = client.post("/api/v1/report", files=files)

            assert resp.status_code == 200
            assert resp.headers.get("content-type") == "application/pdf"
            assert "attachment;" in resp.headers.get("content-disposition", "")
            assert resp.content.startswith(b"%PDF")
