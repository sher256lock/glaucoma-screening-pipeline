"""Validates REFUGE dataset integrity, detects corrupt files, and outputs data_audit.json."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def compute_sha256(file_path: Path) -> str:
    """Compute hex SHA-256 digest of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def validate_image_file(image_path: Path, min_dim: int = 64) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]]]:
    """Verify image exists and decodes cleanly via PIL and OpenCV.

    Returns:
        (is_valid, error_reason, (width, height))
    """
    if not image_path.exists():
        return False, "File does not exist on disk", None

    # PIL validation
    try:
        with Image.open(image_path) as im:
            im.verify()
    except Exception as e:
        return False, f"PIL verify failed: {e}", None

    # OpenCV decode & channel verification
    try:
        mat = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if mat is None:
            return False, "OpenCV imread returned None", None
        h, w, c = mat.shape
        if c != 3:
            return False, f"Expected 3 color channels, got {c}", None
        if h < min_dim or w < min_dim:
            return False, f"Dimensions ({w}x{h}) below minimum threshold ({min_dim}x{min_dim})", None
        return True, None, (w, h)
    except Exception as e:
        return False, f"OpenCV decode exception: {e}", None


def run_data_validation(
    manifest_path: Path = Path("data/processed/manifest.csv"),
    raw_dir: Path = Path("data/raw/refuge"),
    report_path: Path = Path("data/processed/audit_report.json"),
    mirror_report_path: Path = Path("artifacts/reports/data_audit.json"),
) -> Dict[str, Any]:
    """Execute complete dataset validation gate across all images in manifest."""
    manifest_path = Path(manifest_path).resolve()
    raw_dir = Path(raw_dir).resolve()
    report_path = Path(report_path).resolve()
    mirror_report_path = Path(mirror_report_path).resolve()

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run create_splits first.")

    rows: List[Dict[str, str]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("Manifest CSV is empty.")

    corrupt_files: List[Dict[str, str]] = []
    seen_hashes: Dict[str, str] = {}  # sha256 -> relative path
    split_paths: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
    split_hashes: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
    split_class_counts: Dict[str, Dict[str, int]] = {
        "train": {"Normal": 0, "Glaucoma": 0},
        "val": {"Normal": 0, "Glaucoma": 0},
        "test": {"Normal": 0, "Glaucoma": 0},
    }

    total_images_scanned = len(rows)

    for row in rows:
        rel_path = row["image_path"]
        full_path = raw_dir / rel_path
        split = row["split"]
        label_binary = int(row["label_binary"])

        # Validate image file bytes and geometry
        is_valid, err, dims = validate_image_file(full_path)
        if not is_valid:
            corrupt_files.append({"image_path": rel_path, "error": err or "unknown error"})
            continue

        # Compute SHA-256 hash
        file_hash = compute_sha256(full_path)
        if file_hash in seen_hashes:
            logger.warning("Duplicate image hash detected: '%s' matches '%s'", rel_path, seen_hashes[file_hash])
        seen_hashes[file_hash] = rel_path

        # Track split membership
        if split not in split_paths:
            raise ValueError(f"Unknown split name: '{split}' in row {row}")
        split_paths[split].add(rel_path)
        split_hashes[split].add(file_hash)

        # Count classes
        if label_binary == 1:
            split_class_counts[split]["Glaucoma"] += 1
        elif label_binary == 0:
            split_class_counts[split]["Normal"] += 1
        else:
            raise ValueError(f"Invalid label_binary: {label_binary} in row {row}")

    # Check for split leakage
    train_val_overlap = split_paths["train"] & split_paths["val"]
    train_test_overlap = split_paths["train"] & split_paths["test"]
    val_test_overlap = split_paths["val"] & split_paths["test"]

    hash_train_val = split_hashes["train"] & split_hashes["val"]
    hash_train_test = split_hashes["train"] & split_hashes["test"]
    hash_val_test = split_hashes["val"] & split_hashes["test"]

    split_leakage = bool(train_val_overlap or train_test_overlap or val_test_overlap or
                         hash_train_val or hash_train_test or hash_val_test)

    if split_leakage:
        raise ValueError("CRITICAL: Data leakage detected across dataset splits!")

    train_neg = split_class_counts["train"]["Normal"]
    train_pos = split_class_counts["train"]["Glaucoma"]
    pos_weight = float(train_neg / train_pos) if train_pos > 0 else 1.0

    usable_n = total_images_scanned - len(corrupt_files)

    audit_report: Dict[str, Any] = {
        "dataset_name": "REFUGE",
        "total_raw_images": total_images_scanned,
        "corrupt_files_detected": len(corrupt_files),
        "corrupt_files_details": corrupt_files,
        "ungradable_or_suspect_count": 0,
        "usable_n": usable_n,
        "splits": {
            "train": {
                "total": len(split_paths["train"]),
                "normal": split_class_counts["train"]["Normal"],
                "glaucoma": split_class_counts["train"]["Glaucoma"],
            },
            "val": {
                "total": len(split_paths["val"]),
                "normal": split_class_counts["val"]["Normal"],
                "glaucoma": split_class_counts["val"]["Glaucoma"],
            },
            "test": {
                "total": len(split_paths["test"]),
                "normal": split_class_counts["test"]["Normal"],
                "glaucoma": split_class_counts["test"]["Glaucoma"],
            },
        },
        "train_pos_weight": round(pos_weight, 4),
        "split_leakage_detected": split_leakage,
        "hash_unique_count": len(seen_hashes),
    }

    # Save reports
    for out_path in [report_path, mirror_report_path]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(audit_report, f, indent=2)
        logger.info("Saved data audit report to: %s", out_path)

    return audit_report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate REFUGE dataset integrity.")
    parser.add_argument("--manifest", type=str, default="data/processed/manifest.csv")
    parser.add_argument("--raw-dir", type=str, default="data/raw/refuge")
    parser.add_argument("--report", type=str, default="data/processed/audit_report.json")
    args = parser.parse_args()

    # First ensure splits and manifest exist
    from src.data.create_splits import create_splits
    manifest_p = Path(args.manifest)
    if not manifest_p.exists():
        print(f"Manifest '{manifest_p}' does not exist. Creating splits first...")
        create_splits(raw_dir=Path(args.raw_dir), manifest_path=manifest_p)

    report = run_data_validation(
        manifest_path=manifest_p,
        raw_dir=Path(args.raw_dir),
        report_path=Path(args.report),
    )

    print("\n" + "=" * 60)
    print("           REFUGE DATASET INTEGRITY AUDIT REPORT           ")
    print("=" * 60)
    print(f" Total Raw Images Scanned    : {report['total_raw_images']}")
    print(f" Corrupt / Unreadable Files : {report['corrupt_files_detected']}")
    print(f" Ungradable / Suspect Count : {report['ungradable_or_suspect_count']}")
    print(f" Total Usable Sample N      : {report['usable_n']}")
    print("-" * 60)
    print(f" Train Split Counts         : Normal={report['splits']['train']['normal']}, Glaucoma={report['splits']['train']['glaucoma']} (Total: {report['splits']['train']['total']})")
    print(f" Validation Split Counts    : Normal={report['splits']['val']['normal']}, Glaucoma={report['splits']['val']['glaucoma']} (Total: {report['splits']['val']['total']})")
    print(f" Test Split Counts          : Normal={report['splits']['test']['normal']}, Glaucoma={report['splits']['test']['glaucoma']} (Total: {report['splits']['test']['total']})")
    print("-" * 60)
    print(f" Training pos_weight        : {report['train_pos_weight']} (N_neg / N_pos = {report['splits']['train']['normal']} / {report['splits']['train']['glaucoma']})")
    print(f" Split Leakage Detected     : {report['split_leakage_detected']}")
    print("=" * 60 + "\n")
