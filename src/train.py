"""Two-phase fine-tuning pipeline on REFUGE dataset for NVIDIA RTX 4060 GPU."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import yaml

from src.data.dataset import RefugeDataset
from src.model import (
    build_model,
    configure_phase_a,
    configure_phase_b,
    get_phase_b_param_groups,
)

logger = logging.getLogger("train")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic training."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_file_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a configuration file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_pos_weight(csv_path: Path) -> float:
    """Compute training pos_weight = n_neg / n_pos from train split CSV."""
    import csv
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        labels = [int(r["label_binary"]) for r in reader]

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        raise ValueError("Training split contains 0 positive samples.")
    pos_weight = float(n_neg / n_pos)
    logger.info("Computed training split pos_weight: %.4f (n_neg=%d, n_pos=%d)", pos_weight, n_neg, n_pos)
    return pos_weight


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.BCEWithLogitsLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    model.train()
    # Unconditionally lock all BatchNorm modules into eval mode across the entire network
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()

    running_loss = 0.0
    total_samples = 0

    use_cuda_amp = (device.type == "cuda" and scaler is not None)

    for images, targets, _ in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad(set_to_none=True)

        if use_cuda_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on validation split; return (val_loss, val_auroc)."""
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_scores: List[float] = []
    all_targets: List[int] = []

    for images, targets, _ in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)

        logits = model(images)
        loss = criterion(logits, targets)

        running_loss += loss.item() * batch_size
        total_samples += batch_size

        scores = torch.sigmoid(logits).cpu().squeeze().tolist()
        targs = targets.cpu().squeeze().tolist()

        if isinstance(scores, float):
            scores = [scores]
            targs = [targs]

        all_scores.extend(scores)
        all_targets.extend([int(t) for t in targs])

    mean_loss = running_loss / max(total_samples, 1)

    try:
        auroc = float(roc_auc_score(all_targets, all_scores))
    except Exception as e:
        logger.warning("AUROC computation exception: %s. Defaulting to 0.5", e)
        auroc = 0.5

    return mean_loss, auroc


def plot_training_curves(
    history: Dict[str, List[float]],
    phase_a_epochs: int,
    output_path: Path,
) -> None:
    """Plot and save training/validation loss and validation AUROC curves."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curves
    ax1.plot(epochs, history["train_loss"], label="Train Loss (Weighted BCE)", color="#1f77b4", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Validation Loss", color="#ff7f0e", lw=2)
    if phase_a_epochs < len(epochs):
        ax1.axvline(x=phase_a_epochs + 0.5, color="gray", linestyle="--", alpha=0.7, label="Phase A/B Boundary")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # AUROC Curves
    ax2.plot(epochs, history["val_auroc"], label="Validation AUROC", color="#2ca02c", lw=2)
    ax2.axhline(y=0.75, color="red", linestyle=":", alpha=0.7, label="Phase A Gate (0.75)")
    ax2.axhline(y=0.85, color="blue", linestyle=":", alpha=0.7, label="Phase B Floor (0.85)")
    if phase_a_epochs < len(epochs):
        ax2.axvline(x=phase_a_epochs + 0.5, color="gray", linestyle="--", alpha=0.7)

    # Mark best epoch
    best_idx = int(np.argmax(history["val_auroc"]))
    best_epoch = epochs[best_idx]
    best_auroc = history["val_auroc"][best_idx]
    ax2.scatter([best_epoch], [best_auroc], color="#d62728", s=100, zorder=5, label=f"Best ({best_auroc:.4f} @ Ep {best_epoch})")

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("AUROC", fontsize=12)
    ax2.set_title("Validation AUROC Progression", fontsize=14, fontweight="bold")
    ax2.set_ylim(0.4, 1.02)
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Saved training curves plot to: %s", output_path)


def run_training(config_path: Path) -> Dict[str, Any]:
    """Execute complete two-phase training protocol."""
    config_path = Path(config_path).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_cfg = config.get("training", {})
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    # Hardware configuration
    hw_cfg = train_cfg.get("hardware", {})
    preferred_device = hw_cfg.get("device", "cuda")
    device_str = "cuda" if (preferred_device == "cuda" and torch.cuda.is_available()) else "cpu"
    device = torch.device(device_str)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        scaler: Optional[torch.amp.GradScaler] = torch.amp.GradScaler("cuda")
        logger.info("RTX 4060 GPU active: TF32=True, AMP FP16=True, Device=%s", torch.cuda.get_device_name(0))
    else:
        scaler = None
        logger.warning("CUDA unavailable. Running training on CPU.")

    # Paths
    dataset_cfg = config.get("dataset", {})
    raw_dir = Path(dataset_cfg.get("raw_dir", "data/raw/refuge"))
    splits_dir = Path(dataset_cfg.get("splits_dir", "data/processed/splits"))
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"

    if not train_csv.exists() or not val_csv.exists():
        from src.data.create_splits import create_splits
        logger.info("Splits missing. Generating official 400/400/400 splits...")
        create_splits(raw_dir=raw_dir)

    # Dynamic pos_weight on train split only
    pos_weight_val = compute_pos_weight(train_csv)
    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # DataLoaders: Consistent mild augmentations across both Phase A and Phase B to prevent distribution shock
    dl_cfg = train_cfg.get("dataloader", {})
    batch_size = int(dl_cfg.get("batch_size", 64))
    num_workers = int(dl_cfg.get("num_workers", 2))
    pin_memory = (device.type == "cuda")

    train_dataset = RefugeDataset(train_csv, raw_dir=raw_dir, config=config, is_training=True)
    val_dataset = RefugeDataset(val_csv, raw_dir=raw_dir, config=config, is_training=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Build Model
    model = build_model(config)
    model.to(device)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_auroc": [],
    }

    # =========================================================================
    # Phase 3a: Head-Only Warmup
    # =========================================================================
    phase_a_cfg = train_cfg.get("phase_a", {})
    logger.info("--- Starting Phase 3a: %s ---", phase_a_cfg.get("description", "Head-only warmup"))
    configure_phase_a(model)

    opt_a = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(phase_a_cfg.get("lr", 1.0e-3)),
        weight_decay=float(phase_a_cfg.get("weight_decay", 1.0e-4)),
    )

    epochs_a = int(phase_a_cfg.get("epochs", 10))
    gate_min_auroc = float(phase_a_cfg.get("gate_min_val_auroc", 0.75))

    best_phase_a_auroc = -1.0
    best_phase_a_state = copy.deepcopy(model.state_dict())
    best_phase_a_epoch = 0

    for ep in range(1, epochs_a + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, opt_a, device, scaler)
        v_loss, v_auroc = evaluate_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(v_loss)
        history["val_auroc"].append(v_auroc)

        logger.info(
            "[Phase A | Ep %02d/%02d] train_loss: %.4f | val_loss: %.4f | val_auroc: %.4f",
            ep, epochs_a, tr_loss, v_loss, v_auroc,
        )

        if v_auroc > best_phase_a_auroc:
            best_phase_a_auroc = v_auroc
            best_phase_a_state = copy.deepcopy(model.state_dict())
            best_phase_a_epoch = ep

    logger.info("Phase 3a Finished. Best Val AUROC: %.4f at Epoch %d", best_phase_a_auroc, best_phase_a_epoch)

    # Phase 3a Gate Check
    if best_phase_a_auroc < gate_min_auroc:
        raise RuntimeError(
            f"Phase 3a Gate FAILED: Best Val AUROC ({best_phase_a_auroc:.4f}) < {gate_min_auroc}. "
            f"Halting before Phase 3b compute expenditure."
        )
    logger.info("Phase 3a Gate PASSED: %.4f > %.2f", best_phase_a_auroc, gate_min_auroc)

    # =========================================================================
    # Phase 3b: Selective Representation Adaptation (layer3 + layer4 + fc)
    # =========================================================================
    phase_b_cfg = train_cfg.get("phase_b", {})
    logger.info("--- Starting Phase 3b: %s ---", phase_b_cfg.get("description", "Representation adaptation"))

    # Load best Phase A weights
    model.load_state_dict(best_phase_a_state)
    configure_phase_b(model)

    diff_lr_cfg = phase_b_cfg.get("differential_lr", {})
    lr_layer3 = float(diff_lr_cfg.get("layer3", 1.0e-5))
    lr_layer4 = float(diff_lr_cfg.get("layer4", 3.0e-5))
    lr_fc = float(diff_lr_cfg.get("fc", 5.0e-5))
    weight_decay = float(phase_b_cfg.get("weight_decay", 1.0e-3))

    param_groups = get_phase_b_param_groups(
        model,
        lr_layer3=lr_layer3,
        lr_layer4=lr_layer4,
        lr_fc=lr_fc,
        weight_decay=weight_decay,
    )
    opt_b = AdamW(param_groups)

    sched_cfg = phase_b_cfg.get("scheduler", {})
    sched_type = sched_cfg.get("type", "reduce_on_plateau")
    epochs_b = int(phase_b_cfg.get("epochs", 20))
    if sched_type == "cosine":
        scheduler = CosineAnnealingLR(
            opt_b,
            T_max=epochs_b,
            eta_min=float(sched_cfg.get("min_lr", 1.0e-7)),
        )
    else:
        scheduler = ReduceLROnPlateau(
            opt_b,
            mode=sched_cfg.get("mode", "min"),
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 3)),
            min_lr=float(sched_cfg.get("min_lr", 1.0e-7)),
        )

    early_stop_patience = int(phase_b_cfg.get("early_stop", {}).get("patience", 8))

    overall_best_auroc = best_phase_a_auroc
    overall_best_state = copy.deepcopy(best_phase_a_state)
    overall_best_epoch = best_phase_a_epoch
    overall_best_phase = "A"

    no_improve_count = 0

    for ep_b in range(1, epochs_b + 1):
        global_ep = epochs_a + ep_b
        tr_loss = train_one_epoch(model, train_loader, criterion, opt_b, device, scaler)
        v_loss, v_auroc = evaluate_epoch(model, val_loader, criterion, device)
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(v_loss)
        else:
            scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(v_loss)
        history["val_auroc"].append(v_auroc)

        current_lr_fc = opt_b.param_groups[2]["lr"]
        current_lr_l4 = opt_b.param_groups[1]["lr"]
        logger.info(
            "[Phase B | Ep %02d/%02d (Total %02d)] train_loss: %.4f | val_loss: %.4f | val_auroc: %.4f | lr_fc: %.2e | lr_l4: %.2e",
            ep_b, epochs_b, global_ep, tr_loss, v_loss, v_auroc, current_lr_fc, current_lr_l4,
        )

        if v_auroc > overall_best_auroc:
            overall_best_auroc = v_auroc
            overall_best_state = copy.deepcopy(model.state_dict())
            overall_best_epoch = global_ep
            overall_best_phase = "B"
            no_improve_count = 0
            logger.info("--> New overall best checkpoint! Val AUROC: %.4f (Phase B)", overall_best_auroc)
        else:
            no_improve_count += 1
            if no_improve_count >= early_stop_patience:
                logger.info("Early stopping triggered after %d epochs without val_auroc improvement.", no_improve_count)
                break

    # =========================================================================
    # Serialization & Artifact Generation
    # =========================================================================
    ckpt_cfg = config.get("checkpoint", {})
    weights_path = Path(ckpt_cfg.get("weights_path", "artifacts/models/resnet18_refuge.pth")).resolve()
    meta_path = Path(ckpt_cfg.get("metadata_path", "artifacts/models/resnet18_refuge.meta.json")).resolve()
    curves_path = Path(ckpt_cfg.get("training_curves_path", "artifacts/reports/training_curves.png")).resolve()

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    # Save state_dict
    torch.save(overall_best_state, weights_path)
    logger.info("Saved best model state_dict to: %s", weights_path)

    # Save metadata contract
    config_hash = compute_file_sha256(config_path)
    clahe_setting = config.get("input", {}).get("clahe", {})
    clahe_bool = bool(clahe_setting.get("enabled", True) if isinstance(clahe_setting, dict) else clahe_setting)

    metadata = {
        "arch": "resnet18",
        "framework": "pytorch",
        "pytorch_version": torch.__version__,
        "input_size": int(config.get("input", {}).get("size", 224)),
        "channels": int(config.get("input", {}).get("channels", 3)),
        "normalization": config.get("input", {}).get("normalization", {}),
        "clahe": clahe_bool,
        "dataset": "REFUGE",
        "dataset_version": "2026-10-22",
        "usable_n": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": 400,
        },
        "split_protocol": dataset_cfg.get("split_protocol", "official_400_400_400"),
        "pos_weight": round(pos_weight_val, 4),
        "threshold": None,  # Strictly populated during Phase 4 evaluation on val
        "threshold_procedure": config.get("evaluation", {}).get("threshold_procedure", "youden_j_on_val"),
        "val_auroc": round(float(overall_best_auroc), 4),
        "val_sensitivity_at_threshold": None,
        "val_specificity_at_threshold": None,
        "git_sha": "clean",
        "config_hash": config_hash,
        "best_epoch": int(overall_best_epoch),
        "training_phase": overall_best_phase,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved checkpoint metadata contract to: %s", meta_path)

    # Save curves plot
    plot_training_curves(history, epochs_a, curves_path)

    return {
        "best_epoch": overall_best_epoch,
        "best_phase": overall_best_phase,
        "best_val_auroc": overall_best_auroc,
        "weights_path": str(weights_path),
        "meta_path": str(meta_path),
        "curves_path": str(curves_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-phase fine-tuning on REFUGE dataset.")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Path to model config")
    args = parser.parse_args()

    results = run_training(Path(args.config))
    print("\n" + "=" * 60)
    print("                 PHASE 3 TRAINING COMPLETE                  ")
    print("=" * 60)
    print(f" Best Checkpoint Epoch : {results['best_epoch']} (Phase {results['best_phase']})")
    print(f" Best Validation AUROC : {results['best_val_auroc']:.4f}")
    print(f" Saved Model Weights   : {results['weights_path']}")
    print(f" Saved Metadata JSON   : {results['meta_path']}")
    print(f" Saved Training Curves : {results['curves_path']}")
    print("=" * 60 + "\n")
