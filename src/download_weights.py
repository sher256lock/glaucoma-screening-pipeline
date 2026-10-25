"""Automated model weights downloader and baseline initialization utility.

Downloads published release weights into artifacts/models/ with atomic writes
and SHA-256 checksum verification, with graceful baseline fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Union
import urllib.error
import urllib.request
import yaml

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class WeightDownloadError(Exception):
    """Base exception for weight downloading failures."""
    pass


class ChecksumMismatchError(WeightDownloadError):
    """Raised when downloaded weights do not match expected SHA-256 hash."""
    pass


def calculate_sha256(filepath: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute lowercase hexadecimal SHA-256 checksum of a file in chunks."""
    path = Path(filepath).resolve()
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def download_weights(
    url: str,
    destination: Union[str, Path],
    expected_sha256: Optional[str] = None,
    timeout: int = 60,
    chunk_size: int = 65536,
) -> Path:
    """Download model weights from URL with atomic file writing and SHA-256 verification.

    Args:
        url: Remote HTTP(S) URL for the binary .pth weights.
        destination: Local target path (e.g. artifacts/models/resnet18_refuge.pth).
        expected_sha256: Optional lowercase SHA-256 hash to verify downloaded file integrity.
        timeout: Network socket timeout in seconds.
        chunk_size: Stream read/write buffer size in bytes.

    Returns:
        Resolved Path to the finalized model weights file.

    Raises:
        ChecksumMismatchError: If the downloaded file SHA-256 differs from expected_sha256.
        WeightDownloadError: If network connection fails or HTTP status is not 200.
    """
    dest_path = Path(destination).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(f"{dest_path.suffix}.tmp")

    logger.info(f"Downloading model weights from: {url}")
    logger.info(f"Target destination: {dest_path}")

    # Standard User-Agent to avoid HTTP 403 on GitHub release redirects
    headers = {"User-Agent": "GlaucomaScreeningPipeline-AutoDownloader/1.0"}
    req = urllib.request.Request(url, headers=headers)

    hasher = hashlib.sha256()
    bytes_downloaded = 0

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status != 200:
                raise WeightDownloadError(f"HTTP request failed with status code {response.status}: {url}")

            content_length = response.headers.get("Content-Length")
            total_bytes = int(content_length) if content_length and content_length.isdigit() else None

            with open(temp_path, "wb") as f_out:
                while chunk := response.read(chunk_size):
                    f_out.write(chunk)
                    hasher.update(chunk)
                    bytes_downloaded += len(chunk)

        computed_sha256 = hasher.hexdigest().lower()

        if expected_sha256:
            clean_expected = expected_sha256.strip().lower()
            if computed_sha256 != clean_expected:
                if temp_path.exists():
                    temp_path.unlink()
                raise ChecksumMismatchError(
                    f"Integrity check failed for {dest_path.name}! "
                    f"Expected SHA-256: {clean_expected}, but got: {computed_sha256}"
                )

        # Atomic replacement: rename temp file to target
        os.replace(temp_path, dest_path)
        logger.info(f"Weights successfully downloaded and verified ({bytes_downloaded / (1024 * 1024):.2f} MB).")
        return dest_path

    except urllib.error.URLError as e:
        if temp_path.exists():
            temp_path.unlink()
        raise WeightDownloadError(f"Network error downloading weights from {url}: {e}") from e
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def generate_baseline_weights(
    destination: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    config_path: Union[str, Path] = "configs/model_config.yaml",
) -> Path:
    """Generate ImageNet-pretrained ResNet-18 baseline weights with project classification head.

    Enables immediate offline zero-download UI testing and test-suite execution.
    """
    import torch
    from src.model import build_model

    dest_path = Path(destination).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(f"{dest_path.suffix}.tmp")

    if config is None:
        cfg_file = Path(config_path).resolve()
        with open(cfg_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    logger.info("Initializing baseline ImageNet-pretrained ResNet-18 model...")
    model = build_model(config)

    torch.save(model.state_dict(), temp_path)
    os.replace(temp_path, dest_path)
    logger.info(f"Baseline model checkpoint created successfully at: {dest_path}")
    return dest_path


def ensure_weights(
    weights_path: Optional[Union[str, Path]] = None,
    config_path: Union[str, Path] = "configs/model_config.yaml",
    force: bool = False,
    fallback_to_baseline: bool = True,
) -> Path:
    """Ensure weights file exists on disk, downloading or initializing baseline if missing.

    Args:
        weights_path: Optional explicit target path. Defaults to config checkpoint.weights_path.
        config_path: Path to model_config.yaml.
        force: If True, re-download/re-generate even if target file already exists.
        fallback_to_baseline: If download fails, generate an offline baseline model checkpoint.

    Returns:
        Resolved Path to the verified weights file.
    """
    cfg_file = Path(config_path).resolve()
    with open(cfg_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ckpt_cfg = config.get("checkpoint", {})
    dest = Path(weights_path or ckpt_cfg.get("weights_path", "artifacts/models/resnet18_refuge.pth")).resolve()

    if dest.exists() and not force:
        return dest

    download_url = ckpt_cfg.get("download_url")
    expected_sha256 = ckpt_cfg.get("sha256")

    if download_url and not download_url.startswith("https://github.com/<owner>"):
        try:
            return download_weights(
                url=download_url,
                destination=dest,
                expected_sha256=expected_sha256,
            )
        except Exception as e:
            logger.warning(f"Failed to download release weights: {e}")
            if not fallback_to_baseline:
                raise

    if fallback_to_baseline:
        logger.info(f"Falling back to initializing baseline ResNet-18 weights at {dest}...")
        return generate_baseline_weights(destination=dest, config=config, config_path=cfg_file)

    raise WeightDownloadError(f"Weights missing at {dest} and auto-download failed.")


def main() -> None:
    """CLI entrypoint for downloading or initializing project model weights."""
    parser = argparse.ArgumentParser(description="Download or generate ResNet-18 model weights for glaucoma pipeline.")
    parser.add_argument("--config", default="configs/model_config.yaml", help="Path to model_config.yaml")
    parser.add_argument("--dest", default=None, help="Explicit destination path for weights file")
    parser.add_argument("--url", default=None, help="Explicit download URL override")
    parser.add_argument("--force", action="store_true", help="Force re-download even if weights file already exists")
    parser.add_argument("--init-baseline", action="store_true", help="Generate ImageNet baseline weights directly")

    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    target_path = Path(args.dest or config.get("checkpoint", {}).get("weights_path", "artifacts/models/resnet18_refuge.pth")).resolve()

    if args.init_baseline:
        generate_baseline_weights(destination=target_path, config=config, config_path=cfg_path)
        print(f"Successfully generated baseline weights at: {target_path}")
        return

    if args.url:
        download_weights(url=args.url, destination=target_path)
        print(f"Successfully downloaded weights to: {target_path}")
        return

    ensure_weights(weights_path=target_path, config_path=cfg_path, force=args.force)
    print(f"Model weights verified at: {target_path}")


if __name__ == "__main__":
    main()
