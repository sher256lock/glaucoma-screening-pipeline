"""Tests verifying schema integrity and constraints of configs/model_config.yaml."""

from pathlib import Path
import pytest
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "model_config.yaml"


@pytest.fixture(scope="module")
def config():
    """Load and parse the canonical YAML configuration file."""
    assert CONFIG_PATH.exists(), f"Configuration file not found at: {CONFIG_PATH}"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict), "Configuration root must be a YAML dictionary mapping."
    return cfg


def test_config_file_exists_and_loads(config):
    """Verify that model_config.yaml exists and loads cleanly."""
    assert config is not None
    assert len(config) > 0


def test_required_top_level_sections(config):
    """Ensure all required architectural sections are present in configuration."""
    expected_sections = {
        "project",
        "model",
        "input",
        "quality_checks",
        "dataset",
        "training",
        "checkpoint",
        "evaluation",
        "gradcam",
        "inference",
        "server",
    }
    missing = expected_sections - set(config.keys())
    assert not missing, f"Missing required top-level configuration sections: {missing}"


def test_model_architecture_constraints(config):
    """Verify non-negotiable model architecture constraints."""
    model_cfg = config.get("model", {})
    assert model_cfg.get("arch") == "resnet18", (
        f"Strict constraint: model.arch must be 'resnet18', got '{model_cfg.get('arch')}'"
    )
    assert model_cfg.get("num_classes") == 1, (
        f"Binary logit output requires num_classes=1, got {model_cfg.get('num_classes')}"
    )
    head_cfg = model_cfg.get("head", {})
    assert head_cfg.get("in_features") == 512, "ResNet-18 fc in_features must be 512"
    assert head_cfg.get("out_features") == 1, "ResNet-18 fc out_features must be 1"
    assert head_cfg.get("dropout") == 0.3, "Classification head dropout must be 0.3"


def test_input_normalization_and_dimensions(config):
    """Verify standard ImageNet input dimensions and normalization constants."""
    input_cfg = config.get("input", {})
    assert input_cfg.get("size") == 224, "Input image spatial size must be 224x224"
    assert input_cfg.get("channels") == 3, "Input channels must be 3 (RGB)"
    norm_cfg = input_cfg.get("normalization", {})
    assert norm_cfg.get("mean") == [0.485, 0.456, 0.406], "ImageNet mean mismatch"
    assert norm_cfg.get("std") == [0.229, 0.224, 0.225], "ImageNet std mismatch"


def test_terminology_and_governance(config):
    """Enforce terminology rules: 'Glaucoma Suspicion Score' only; no confidence/probability claims."""
    project_cfg = config.get("project", {})
    score_label = project_cfg.get("score_label", "")
    assert score_label == "Glaucoma Suspicion Score", (
        f"Score label must strictly be 'Glaucoma Suspicion Score', got '{score_label}'"
    )
    assert "disclaimer" in project_cfg and len(project_cfg["disclaimer"]) > 0

    server_cfg = config.get("server", {})
    assert server_cfg.get("framework") == "fastapi", (
        "Server framework must be 'fastapi'. Streamlit is strictly prohibited."
    )


def test_training_phase_definitions(config):
    """Verify Phase A and Phase B fine-tuning module specifications."""
    training_cfg = config.get("training", {})
    assert training_cfg.get("phase_a", {}).get("trainable_modules") == ["fc"], (
        "Phase A must warm up the 'fc' head only"
    )
    assert training_cfg.get("phase_b", {}).get("trainable_modules") in [["layer4", "fc"], ["layer3", "layer4", "fc"]], (
        "Phase B must adapt 'layer4' and 'fc' (optionally including 'layer3')"
    )


def test_dataset_protocol_and_paths(config):
    """Verify REFUGE dataset paths and split protocol specification."""
    dataset_cfg = config.get("dataset", {})
    assert dataset_cfg.get("name") == "REFUGE"
    assert dataset_cfg.get("raw_dir") == "data/raw/refuge"
    assert dataset_cfg.get("split_protocol") == "official_400_400_400"
    label_map = dataset_cfg.get("label_map", {})
    assert label_map.get("Glaucoma") == 1
    assert label_map.get("Normal") == 0


def test_checkpoint_and_evaluation_contracts(config):
    """Verify checkpoint paths and Youden's J evaluation protocol."""
    checkpoint_cfg = config.get("checkpoint", {})
    assert checkpoint_cfg.get("weights_path") == "artifacts/models/resnet18_refuge.pth"
    assert checkpoint_cfg.get("metadata_path") == "artifacts/models/resnet18_refuge.meta.json"

    eval_cfg = config.get("evaluation", {})
    assert eval_cfg.get("threshold_procedure") == "youden_j_on_val"
