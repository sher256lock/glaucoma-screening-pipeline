"""Unit tests verifying FastAPI endpoints, lifecycles, and error responses."""

import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with lifespan context entered."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def sample_jpeg_bytes():
    """Create sample in-memory JPEG bytes."""
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestAPIEndpoints:
    """Verify health, prediction, evaluation, and static UI routes."""

    def test_health_check_endpoint(self, client):
        """GET /health must return 200 OK with model status and threshold."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["device"] == "cpu"
        assert 0.0 < data["threshold"] < 1.0
        assert "disclaimer" in data

    def test_evaluation_endpoint(self, client):
        """GET /api/v1/evaluation must return 200 OK with test cohort metrics."""
        resp = client.get("/api/v1/evaluation")
        assert resp.status_code == 200
        data = resp.json()
        assert data["test_n"] == 400
        assert "metrics" in data
        assert "confusion_matrix" in data

    def test_predict_endpoint_success(self, client, sample_jpeg_bytes):
        """POST /api/v1/predict with valid image must return 200 OK with prediction contract."""
        files = {"file": ("test_fundus.jpg", sample_jpeg_bytes, "image/jpeg")}
        resp = client.post("/api/v1/predict", files=files)
        assert resp.status_code == 200
        data = resp.json()

        assert "reference_id" in data
        assert data["predicted_class"] in ["Glaucoma", "Normal"]
        assert 0.0 <= data["suspicion_score"] <= 1.0
        assert 0.0 < data["threshold_used"] < 1.0
        assert data["latency_ms"] > 0
        assert isinstance(data["quality_warnings"], list)
        assert len(data["gradcam_overlay_base64"]) > 0
        assert "RESEARCH PROTOTYPE ONLY" in data["disclaimer"]

    def test_predict_endpoint_corrupt_bytes(self, client):
        """POST /api/v1/predict with corrupt image bytes must return 400 Bad Request."""
        files = {"file": ("corrupt.jpg", b"not-a-real-jpeg", "image/jpeg")}
        resp = client.post("/api/v1/predict", files=files)
        assert resp.status_code == 400
        assert "corrupt" in resp.json()["detail"].lower() or "decode" in resp.json()["detail"].lower()

    def test_predict_endpoint_disallowed_extension(self, client, sample_jpeg_bytes):
        """POST /api/v1/predict with disallowed file extension must return 400 Bad Request."""
        files = {"file": ("malicious.exe", sample_jpeg_bytes, "application/octet-stream")}
        resp = client.post("/api/v1/predict", files=files)
        assert resp.status_code == 400
        assert "unsupported" in resp.json()["detail"].lower()

    def test_root_serves_static_ui(self, client):
        """GET / must return 200 OK with HTML content."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "REFUGE Glaucoma Binary Screening" in resp.text
