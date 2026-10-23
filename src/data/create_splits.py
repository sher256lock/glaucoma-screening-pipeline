"""Emits data/processed/manifest.csv and data/processed/splits/ preserving official REFUGE splits."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.data.load_refuge import load_refuge_records

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CONTRACT_COLUMNS = ["image_path", "label_original", "label_binary", "split", "notes"]


def verify_split_disjointness(records: List[Dict[str, Any]]) -> None:
    """Verify that train, val, and test splits have strictly zero image path overlap."""
    train_paths = {r["image_path"] for r in records if r["split"] == "train"}
    val_paths = {r["image_path"] for r in records if r["split"] == "val"}
    test_paths = {r["image_path"] for r in records if r["split"] == "test"}

    train_val_overlap = train_paths & val_paths
    train_test_overlap = train_paths & test_paths
    val_test_overlap = val_paths & test_paths

    if train_val_overlap:
        raise ValueError(f"CRITICAL: Train and Val splits overlap on {len(train_val_overlap)} images: {train_val_overlap}")
    if train_test_overlap:
        raise ValueError(f"CRITICAL: Train and Test splits overlap on {len(train_test_overlap)} images: {train_test_overlap}")
    if val_test_overlap:
        raise ValueError(f"CRITICAL: Val and Test splits overlap on {len(val_test_overlap)} images: {val_test_overlap}")

    logger.info("Split disjointness verified: Train (%d), Val (%d), Test (%d) have zero overlap.",
                len(train_paths), len(val_paths), len(test_paths))


def write_csv_manifest(records: List[Dict[str, Any]], output_path: Path) -> None:
    """Write records to a CSV file matching the exact contract schema."""
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CONTRACT_COLUMNS)
        writer.writeheader()
        for record in records:
            filtered_record = {k: record[k] for k in CONTRACT_COLUMNS}
            writer.writerow(filtered_record)

    logger.info("Wrote %d records to %s", len(records), output_path)


def create_splits(
    raw_dir: Path = Path("data/raw/refuge"),
    manifest_path: Path = Path("data/processed/manifest.csv"),
    splits_dir: Path = Path("data/processed/splits"),
) -> None:
    """Generate master manifest and per-split CSV files from raw REFUGE directory."""
    raw_dir = Path(raw_dir)
    manifest_path = Path(manifest_path)
    splits_dir = Path(splits_dir)

    usable_records, excluded_records = load_refuge_records(raw_dir)
    if not usable_records:
        raise RuntimeError("Zero usable records loaded from raw REFUGE dataset.")

    verify_split_disjointness(usable_records)

    # Write master manifest
    write_csv_manifest(usable_records, manifest_path)

    # Write split-specific manifests
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        split_records = [r for r in usable_records if r["split"] == split_name]
        split_csv = splits_dir / f"{split_name}.csv"
        write_csv_manifest(split_records, split_csv)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create REFUGE official split manifests.")
    parser.add_argument("--raw-dir", type=str, default="data/raw/refuge", help="Path to raw REFUGE directory")
    parser.add_argument("--output", type=str, default="data/processed/manifest.csv", help="Master manifest CSV path")
    parser.add_argument("--splits-dir", type=str, default="data/processed/splits", help="Directory for split CSVs")
    args = parser.parse_args()

    create_splits(
        raw_dir=Path(args.raw_dir),
        manifest_path=Path(args.output),
        splits_dir=Path(args.splits_dir),
    )
