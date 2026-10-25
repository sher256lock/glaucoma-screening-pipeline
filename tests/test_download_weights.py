"""Unit tests verifying model weights download, atomic writes, SHA-256 validation, and baseline fallback."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import urllib.error
import pytest
import torch

from src.download_weights import (
    ChecksumMismatchError,
    WeightDownloadError,
    calculate_sha256,
    download_weights,
    ensure_weights,
    generate_baseline_weights,
)


class TestCalculateSha256:
    """Verify SHA-256 computation."""

    def test_calculate_sha256_known_content(self, tmp_path):
        sample_file = tmp_path / "sample.txt"
        sample_file.write_bytes(b"hello world")
        # SHA-256 of b"hello world" is b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9
        expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        assert calculate_sha256(sample_file) == expected


class TestDownloadWeights:
    """Verify download logic, atomic replacement, and checksum validation."""

    def test_download_success_with_valid_checksum(self, tmp_path):
        dest = tmp_path / "models" / "downloaded.pth"
        payload = b"dummy-pytorch-weights-bytes"
        expected_hash = "9e3fa848a1078a188aff59967491c521de31530994b51c5cf7e2547bcc36ee67"

        # Mock urllib.request.urlopen context manager
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.headers = {"Content-Length": str(len(payload))}
        mock_resp.read.side_effect = [payload, b""]
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result_path = download_weights(
                url="https://example.com/resnet18.pth",
                destination=dest,
                expected_sha256=expected_hash,
            )

        assert result_path.exists()
        assert result_path.read_bytes() == payload
        # Ensure temporary file is cleaned up / replaced
        assert not dest.with_suffix(".pth.tmp").exists()

    def test_download_checksum_mismatch_raises_and_cleans_tmp(self, tmp_path):
        dest = tmp_path / "models" / "corrupt.pth"
        payload = b"corrupted-payload"

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.headers = {"Content-Length": str(len(payload))}
        mock_resp.read.side_effect = [payload, b""]
        mock_resp.__enter__.return_value = mock_resp

        wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000"

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(ChecksumMismatchError) as exc_info:
                download_weights(
                    url="https://example.com/corrupt.pth",
                    destination=dest,
                    expected_sha256=wrong_hash,
                )

        assert "integrity check failed" in str(exc_info.value).lower()
        assert not dest.exists()
        assert not dest.with_suffix(".pth.tmp").exists()

    def test_download_network_error_raises_weight_download_error(self, tmp_path):
        dest = tmp_path / "models" / "network_err.pth"

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            with pytest.raises(WeightDownloadError) as exc_info:
                download_weights(
                    url="https://unreachable.domain/model.pth",
                    destination=dest,
                )

        assert "network error" in str(exc_info.value).lower()
        assert not dest.exists()


class TestBaselineGenerationAndEnsure:
    """Verify offline baseline checkpoint creation and ensure_weights workflow."""

    def test_generate_baseline_weights_creates_valid_state_dict(self, tmp_path):
        dest = tmp_path / "baseline.pth"
        result_path = generate_baseline_weights(destination=dest)

        assert result_path.exists()
        state_dict = torch.load(result_path, map_location="cpu", weights_only=True)
        assert isinstance(state_dict, dict)
        assert "fc.1.weight" in state_dict
        assert state_dict["fc.1.weight"].shape == (1, 512)

    def test_ensure_weights_returns_existing_path_without_redownload(self, tmp_path):
        existing_weights = tmp_path / "existing.pth"
        existing_weights.write_bytes(b"existing-bytes")

        with patch("src.download_weights.download_weights") as mock_dl:
            res = ensure_weights(weights_path=existing_weights, force=False)
            assert res == existing_weights
            mock_dl.assert_not_called()
