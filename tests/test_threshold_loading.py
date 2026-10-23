"""Unit tests verifying threshold freezing in metadata, Grad-CAM execution, and evaluation artifacts."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml

from src.explainability import classify_outcomes, generate_gradcam_overlay
from src.model import build_model

CONFIG_PATH = Path("configs/model_config.yaml")
META_PATH = Path("artifacts/models/resnet18_refuge.meta.json")
EVAL_JSON_PATH = Path("artifacts/reports/evaluation.json")
EVAL_MD_PATH = Path("artifacts/reports/evaluation.md")
GRADCAM_GRID_PATH = Path("artifacts/reports/gradcam_grid.png")
ROC_PR_PATH = Path("artifacts/reports/roc_pr_curves.png")


@pytest.fixture(scope="module")
def config():
    assert CONFIG_PATH.exists()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestThresholdMetadataAndContracts:
    """Verify metadata threshold freezing and evaluation report contracts."""

    def test_metadata_contains_frozen_threshold(self):
        """Verify resnet18_refuge.meta.json contains a non-null float threshold and metric floor compliance."""
        if not META_PATH.exists():
            pytest.skip("resnet18_refuge.meta.json not found")

        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert "threshold" in meta
        assert meta["threshold"] is not None
        assert isinstance(meta["threshold"], float)
        assert 0.0 < meta["threshold"] < 1.0, f"Threshold {meta['threshold']} must be in (0, 1)"

        assert "val_sensitivity_at_threshold" in meta
        assert meta["val_sensitivity_at_threshold"] is not None
        assert isinstance(meta["val_sensitivity_at_threshold"], float)
        assert meta["val_sensitivity_at_threshold"] >= 0.80, (
            f"Validation sensitivity floor failed: {meta['val_sensitivity_at_threshold']} < 0.80"
        )

        assert "val_specificity_at_threshold" in meta
        assert meta["val_specificity_at_threshold"] is not None
        assert isinstance(meta["val_specificity_at_threshold"], float)
        assert 0.0 < meta["val_specificity_at_threshold"] < 1.0

        assert meta.get("threshold_procedure") in [
            "youden_j_on_val",
            "youden_j_sensitivity_adjusted_on_val",
        ]

    def test_evaluation_artifacts_exist_and_match_contract(self):
        """Verify evaluation.json, evaluation.md, gradcam_grid.png, and roc_pr_curves.png match specifications."""
        if not EVAL_JSON_PATH.exists():
            pytest.skip("evaluation.json not yet generated (run src.evaluate first)")

        with open(EVAL_JSON_PATH, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        # Check JSON schema compliance
        assert eval_data["test_n"] == 400
        assert eval_data["test_prevalence_glaucoma"] == 0.10
        assert "threshold_used" in eval_data and 0.0 < eval_data["threshold_used"] < 1.0
        assert "metrics" in eval_data
        for m in ["auroc", "sensitivity", "specificity", "f1"]:
            assert m in eval_data["metrics"]
            assert "value" in eval_data["metrics"][m]
            assert "ci95" in eval_data["metrics"][m]
            assert len(eval_data["metrics"][m]["ci95"]) == 2
        assert "brier_diagnostic_only" in eval_data["metrics"]

        # Confusion matrix checks
        cm = eval_data["confusion_matrix"]
        assert cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"] == 400
        assert cm["tp"] + cm["fn"] == 40  # 40 positive cases
        assert cm["tn"] + cm["fp"] == 360  # 360 normal cases

        # Latency checks
        latency = eval_data["latency_ms_cpu"]
        assert latency["device"] == "cpu"
        assert latency["p50"] > 0
        assert latency["p95"] >= latency["p50"]
        assert latency["threads"] > 0

        # Markdown checks
        assert EVAL_MD_PATH.exists()
        with open(EVAL_MD_PATH, "r", encoding="utf-8") as f:
            md_text = f.read()

        # Check 7 mandatory sections
        assert "## 1. Test Cohort & Prevalence" in md_text
        assert "## 2. Performance Metrics" in md_text
        assert "## 3. Test Split Confusion Matrix" in md_text
        assert "## 4. Frozen Threshold & Validation Calibration Procedure" in md_text
        assert "## 5. Local CPU Inference Latency Benchmark" in md_text
        assert "## 6. Verbatim Limitations Disclosures" in md_text
        assert "## 7. Mandatory Regulatory & Statistical Disclaimer" in md_text

        # Check verbatim disclosures
        assert "REFUGE Patient-ID Limitation Disclosure (Verbatim)" in md_text
        assert "Transparency & Label Stratification Disclosure (Verbatim)" in md_text
        assert "RESEARCH PROTOTYPE ONLY" in md_text

        # Check images
        assert GRADCAM_GRID_PATH.exists() and GRADCAM_GRID_PATH.stat().st_size > 1000
        assert ROC_PR_PATH.exists() and ROC_PR_PATH.stat().st_size > 1000


class TestExplainabilityAndOutcomes:
    """Verify Grad-CAM implementation, hook cleanup, and outcome partitioning."""

    def test_classify_outcomes_partitioning(self):
        """Verify classify_outcomes accurately splits test outcomes into disjoint sets."""
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_score = np.array([0.9, 0.3, 0.8, 0.1, 0.6, 0.2])
        threshold = 0.5

        outcomes = classify_outcomes(y_true, y_score, threshold)
        assert set(outcomes["tp_indices"]) == {0, 4}
        assert set(outcomes["fp_indices"]) == {2}
        assert set(outcomes["tn_indices"]) == {3, 5}
        assert set(outcomes["fn_indices"]) == {1}

        total_classified = (
            len(outcomes["tp_indices"])
            + len(outcomes["fp_indices"])
            + len(outcomes["tn_indices"])
            + len(outcomes["fn_indices"])
        )
        assert total_classified == len(y_true)

    def test_gradcam_overlay_and_hook_cleanup(self, config):
        """Verify Grad-CAM produces normalized heatmap, correct overlay, and leaves zero lingering hooks."""
        model = build_model(config)
        model.eval()

        input_tensor = torch.randn(1, 3, 224, 224)
        orig_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        heatmap, overlay = generate_gradcam_overlay(
            model=model,
            input_tensor=input_tensor,
            original_rgb_uint8=orig_img,
        )

        assert heatmap.shape == (224, 224)
        assert heatmap.dtype == np.float32
        assert 0.0 <= heatmap.min()
        assert heatmap.max() <= 1.0

        assert overlay.shape == (224, 224, 3)
        assert overlay.dtype == np.uint8

        # Target layer hooks must be cleanly and unconditionally removed
        target_layer = model.layer4[-1]
        assert len(target_layer._forward_hooks) == 0, "Lingering forward hook found on layer4[-1]!"
        assert len(target_layer._backward_hooks) == 0, "Lingering backward hook found on layer4[-1]!"
