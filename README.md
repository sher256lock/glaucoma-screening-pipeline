<div align="center">

# Glaucoma Binary Screening Pipeline — PyTorch ResNet-18 / REFUGE

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B%20%7C%20CUDA%2012.4-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-267CB9?style=flat-square)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-555555?style=flat-square)](https://github.com/sher256lock/glaucoma-screening-pipeline)
[![Tests](https://img.shields.io/badge/Tests-61%20Passed-2ea44f?style=flat-square&logo=pytest&logoColor=white)](tests/)

<p align="center">
  <strong>Research and educational prototype for binary glaucoma screening from retinal fundus photographs.</strong><br>
  Powered by a class-weighted PyTorch ResNet-18 backbone, Grad-CAM visual explainability, FastAPI REST engine, and a zero-build clinical dashboard.
</p>

<p align="center">
  <a href="#about-the-project">About</a> •
  <a href="#getting-started">Quickstart</a> •
  <a href="#usage">Usage Modes</a> •
  <a href="#architecture--specifications">Architecture</a> •
  <a href="#documentation-index">Documentation</a> •
  <a href="#mandatory-regulatory--statistical-disclaimer">Disclaimer</a>
</p>

</div>

> [!IMPORTANT]
> **MANDATORY CLINICAL & REGULATORY NOTICE**  
> This system is an **educational and research prototype**. It is **not a clinically validated diagnostic device** and is **not for clinical decision-making or diagnosis**. Model outputs are reported strictly as a continuous **Glaucoma Suspicion Score** and do not represent calibrated statistical probabilities. All screening outputs require review by a licensed ophthalmologist.

---

## 📋 Table of Contents

- [About the Project](#about-the-project)
  - [Overview & Clinical Motivation](#overview--clinical-motivation)
  - [Key Architectural Features](#key-architectural-features)
  - [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Automated Model Weights Protocol](#automated-model-weights-protocol)
- [Usage](#usage)
  - [Surface 1: Interactive Web Dashboard](#surface-1-interactive-web-dashboard)
  - [Surface 2: Headless CLI Screening](#surface-2-headless-cli-screening)
  - [Surface 3: Two-Phase Training & Evaluation Pipeline](#surface-3-two-phase-training--evaluation-pipeline)
- [Architecture & Specifications](#architecture--specifications)
  - [Executive Specification Matrix](#executive-specification-matrix)
  - [Visual Explainability (Grad-CAM)](#visual-explainability-grad-cam)
- [Documentation Index](#documentation-index)
- [Contributing](#contributing)
- [License](#license)
- [Mandatory Regulatory & Statistical Disclaimer](#mandatory-regulatory--statistical-disclaimer)

---

## 🔬 About the Project

Glaucoma is one of the leading causes of irreversible visual impairment worldwide, frequently dubbed the "silent thief of sight" due to its asymptomatic progression in early stages. Population-scale screening using digital fundus photography offers a vital intervention window, yet clinical diagnostic workflows face severe specialist shortages and high inter-observer variability.

This repository provides an **end-to-end, reproducible deep learning screening pipeline** developed on the gold-standard **REFUGE (Retinal Fundus Glaucoma Challenge)** dataset (400 train / 400 validation / 400 test split).

### Key Architectural Features

* **Calibrated Class Imbalance Handling**: Trains with dynamically weighted Binary Cross-Entropy (`pos_weight = 9.0`) to balance the severe 9:1 negative-to-positive class skew inherent to screening populations without artificial synthetic oversampling.
* **Two-Phase Transfer Learning Schedule**: Protects pretrained visual features via a strict Phase A (`fc` head warmup) followed by Phase B (differential learning rate adaptation on `layer3`, `layer4`, and `fc`).
* **Youden's J Threshold Freezing**: Eliminates arbitrary 0.50 decision boundaries by calibrating and freezing optimal sensitivity/specificity operating thresholds on the held-out validation cohort (`threshold = 0.2921`).
* **Visual Explainability via Grad-CAM**: Automatically computes gradient-weighted class activation mappings against `model.layer4[-1]`, overlaying a bilinear heatmap directly over the optic nerve head and neuroretinal rim.
* **Deterministic CPU Inference Engine**: Enforces single-image batching (`batch_size = 1`), intra-op CPU thread pinning, and strict in-memory PHI lifecycle management (zero disk persistence of patient uploads).
* **Zero-Build Local Web Surface**: Replaces heavyweight web runtimes with a responsive HTML5/Vanilla JS clinical dashboard served via FastAPI and Uvicorn. *(Streamlit is strictly banned per architectural constraints).*

### Built With

* [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/) — Tensor computation, GPU acceleration & deep learning
* [![Torchvision](https://img.shields.io/badge/Torchvision-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/vision/) — ResNet-18 ImageNet backbone & transformations
* [![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/) — High-performance asynchronous REST API backend
* [![Uvicorn](https://img.shields.io/badge/Uvicorn-499848?style=flat-square&logo=gunicorn&logoColor=white)](https://www.uvicorn.org/) — Lightning-fast ASGI production web server
* [![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/) — CLAHE contrast enhancement & Laplacian blur heuristics
* [![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/) — ROC/PR curves, Youden's J, and bootstrap statistical metrics
* [![ReportLab](https://img.shields.io/badge/ReportLab-C8372D?style=flat-square&logo=adobeacrobatreader&logoColor=white)](https://www.reportlab.com/) — Single-page vector clinical PDF report generation
* [![Pytest](https://img.shields.io/badge/pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white)](https://pytest.org/) — Automated 61-test verification harness

---

## 🚀 Getting Started

Follow these steps to set up and run the screening pipeline locally from a fresh clone.

### Prerequisites

* **Python**: 3.10, 3.11, 3.12, or 3.13.
* **Training Hardware (Optional)**: NVIDIA GPU with Tensor Cores (e.g., RTX 4060 8GB, CUDA 12.1/12.4).
* **Inference Hardware**: Standard x86_64 CPU (Intel Core / AMD Ryzen).

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sher256lock/glaucoma-screening-pipeline.git
   cd glaucoma-screening-pipeline
   ```

2. **Create and activate a virtual environment:**
   * **Windows PowerShell:**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
     *(If script execution is blocked on your system, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

   * **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

   > [!TIP]
   > **Windows PowerShell Path Syntax Gotcha**:  
   > When invoking relative executables in PowerShell, always prefix with `.\` (e.g., `.\.venv\Scripts\python.exe`). Omitting `.\` causes PowerShell's parser to interpret `.venv` as a module name to import, resulting in an `Import-Module .venv` failure.

3. **Install dependencies:**
   ```bash
   # Core dependencies (FastAPI, OpenCV, Scikit-Learn, ReportLab, Pytest):
   pip install -r requirements.txt
   ```

   *(Optional — For NVIDIA RTX GPU training acceleration):*
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

---

### Automated Model Weights Protocol

> [!NOTE]
> **Zero-Bloat Automated Weight Acquisition**:  
> Binary weights (`artifacts/models/resnet18_refuge.pth`, ~44.8 MB) are excluded from Git tracking via `.gitignore` to prevent repository bloat.  
> The inference engine features an **automated fetcher**: on startup, if weights are missing, it downloads them from the configured GitHub Release asset (`checkpoint.download_url` in `configs/model_config.yaml`), verifies the SHA-256 checksum, and performs an atomic save.

You can also manage weights manually via any of the following options:

* **Option A — Automated CLI Fetcher & Integrity Verifier (Recommended)**:
  ```bash
  # Download from release and verify SHA-256 hash:
  python -m src.download_weights

  # Force re-download:
  python -m src.download_weights --force
  ```

* **Option B — Instant Offline Baseline Initialization (Zero-Download UI Testing)**:
  ```bash
  # Instantiate an ImageNet-pretrained ResNet-18 model locally for instant demo testing:
  python -m src.download_weights --init-baseline
  ```

* **Option C — Deploy an Existing Trained Checkpoint**:
  Place your pre-trained weights file directly into:
  ```text
  artifacts/models/resnet18_refuge.pth
  ```

---

## 💻 Usage

### Surface 1: Interactive Web Dashboard

Launch the local FastAPI service with mounted zero-build static UI:

```bash
# With activated virtual environment:
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Direct invocation in Windows PowerShell (without activating .venv):
.\.venv\Scripts\python.exe -m uvicorn src.api:app --host 0.0.0.0 --port 8000

# Direct invocation in Linux / macOS (without activating .venv):
.venv/bin/python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

* **Clinical Dashboard**: Open **`http://localhost:8000`** in any browser.
* **Features**:
  * In-memory drag-and-drop retinal photograph screening (pure RAM, zero disk leakage).
  * Dynamic Glaucoma Suspicion Score meter with classification indicator (`Normal` vs. `Glaucoma`).
  * Interactive Grad-CAM heatmap visualization over the optic disc.
  * Instant 1-click **Download Clinical PDF Report** generation.
* **REST Endpoints**:
  * `POST /api/v1/predict`: Returns continuous suspicion score, binary classification, threshold, latency, and Base64 Grad-CAM overlay.
  * `POST /api/v1/report`: Returns an in-memory generated single-page clinical PDF report.
  * `GET /api/v1/evaluation`: Returns official test cohort metrics, confusion matrix, and bootstrap 95% CIs.
  * `GET /health`: Returns engine status, physical threads, device, and frozen threshold.

---

### Surface 2: Headless CLI Screening

For scripting, batch pipelines, and automated test environments:

```bash
# Screen a single image with terminal report and auto-generated PDF:
python -m src.cli --file path/to/fundus.jpg --pdf

# Screen a single image with explicit Grad-CAM and PDF destinations:
python -m src.cli --file path/to/fundus.jpg --save-gradcam artifacts/reports/sample_cam.png --pdf artifacts/reports/sample_report.pdf

# Batch screening across an entire directory:
python -m src.cli --dir path/to/fundus_dir/
```

---

### Surface 3: Two-Phase Training & Evaluation Pipeline

To reproduce model weights from scratch using the official REFUGE dataset (400 / 400 / 400 split):

```bash
# Step 1: Verify hardware acceleration and run 61-test unit harness:
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
pytest -q

# Step 2: Ingest REFUGE dataset and audit manifest splits (Phase 2 Gate):
python -m src.data.load_refuge --raw-dir data/raw/refuge --output data/processed/manifest.csv
python -m src.data.validate_data --manifest data/processed/manifest.csv --report artifacts/reports/data_audit.json
python -m src.data.create_splits --manifest data/processed/manifest.csv --output-dir data/processed/splits

# Step 3: Run Two-Phase Training on RTX 4060 GPU:
python -m src.train --config configs/model_config.yaml

# Step 4: Evaluate test cohort, freeze Youden's J threshold, and generate Grad-CAM grid:
python -m src.evaluate --config configs/model_config.yaml
```

---

## 🏛️ Architecture & Specifications

### Executive Specification Matrix

| Dimension | Specification | Governance Enforcement |
| :--- | :--- | :--- |
| **Backbone Architecture** | `torchvision.models.resnet18(weights=IMAGENET1K_V1)` | Fixed backbone. Over-parameterized alternatives (ResNet-50, ViT) strictly forbidden. |
| **Classification Head** | `nn.Sequential(nn.Dropout(p=0.3), nn.Linear(512, 1))` | Single-layer linear projection to prevent overfitting on *N* = 400 positive-sparse data. |
| **Dataset & Split** | **REFUGE Challenge** (400 train / 400 val / 400 test) | Official challenge split. Excludes ungradable and suspect labels. |
| **Loss Function** | `BCEWithLogitsLoss(pos_weight=9.0)` | Dynamically computed on train split (*N*~neg~ / *N*~pos~) to balance class skew. |
| **Training Schedule** | **Phase A** (`fc` warmup) → **Phase B** (`layer3` + `layer4` + `fc`) | Protects pretrained early filters from disruptive initial gradient surges. |
| **Operating Threshold** | **Youden's J = 0.2921** (calibrated on validation split) | Sensitivity floor ≥ 0.80; specificity ≈ 0.864. Never hardcoded to 0.50. |
| **Inference Hardware** | Local **x86_64 CPU** (`batch_size = 1`, thread-pinned) | Single-image latency ≤ 150 ms; deterministic intra-op parallelism. |
| **Web Service Surface** | **FastAPI + Uvicorn** + Static Vanilla HTML5/JS UI | Pure in-memory streaming; Streamlit is strictly prohibited. |
| **Explainability** | **Grad-CAM (`model.layer4[-1]`)** | Bilinear heatmap interpolation overlaid onto original fundus photograph. |

### Visual Explainability (Grad-CAM)

The pipeline integrates Gradient-weighted Class Activation Mapping (Grad-CAM) at the final convolutional feature layer (`model.layer4[-1]`).

Importance weights $\alpha_k$ are derived by global average pooling the gradients of the target class logit $y$ with respect to feature activation maps $A^k$:

$$\alpha_k = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y}{\partial A_{i,j}^k}$$

A rectified linear combination yields the 2D coarse localization map:

$$L_{\text{Grad-CAM}} = \text{ReLU}\left( \sum_{k} \alpha_k A^k \right)$$

This activation map is bilinearly upsampled to 224 × 224 and blended as a JET color overlay over the original fundus photograph, confirming that network attention focuses on the **optic disc excavation and neuroretinal rim thinning** rather than peripheral retinal artifacts.

---

## 📚 Documentation Index

All project governance, data protocols, and architectural specifications reside in [`docs/`](docs/):

1. **[`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md)** — Canonical project specification, hard constraints, two-phase training schedules, and phase gates.
2. **[`docs/architecture.md`](docs/architecture.md)** — Data flow diagrams, module boundaries, JSON/Pydantic schemas, and in-memory PHI lifecycle.
3. **[`docs/dataset_protocol.md`](docs/dataset_protocol.md)** — REFUGE dataset acquisition, ungradable exclusion rules, and patient-ID leakage disclosures.
4. **[`docs/evaluation_protocol.md`](docs/evaluation_protocol.md)** — Statistical metrics, Youden's J threshold freezing, 1,000-resample bootstrap 95% CIs, and Grad-CAM error grids.
5. **[`docs/CODEMAPS/`](docs/CODEMAPS/)** — Token-lean architecture, data pipeline, and backend service codemaps.
6. **[`docs/rejected_patterns.md`](docs/rejected_patterns.md)** — Explicit architectural anti-patterns (including bans on Streamlit, ResNet-50, and uncalibrated probability claims).

---

## 🤝 Contributing

Contributions to improve research tooling, inference efficiency, or documentation are welcome:

1. Fork the Project.
2. Create your Feature Branch:
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your Changes adhering to Conventional Commits:
   ```bash
   git commit -m 'feat: add AmazingFeature'
   ```
4. Push to the Branch:
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request.

Please review [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`docs/rejected_patterns.md`](docs/rejected_patterns.md) before submitting code changes.

---

## 📄 License

Distributed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) for complete terms.

---

## ⚠️ Mandatory Regulatory & Statistical Disclaimer

This system is an **educational and research prototype**. Model outputs are reported strictly as a **"Glaucoma Suspicion Score"** derived from a class-weighted binary cross-entropy objective (`BCEWithLogitsLoss`). 

Because class weighting shifts the prior log-odds by `+ln(w_pos)`, these continuous output values **are not calibrated statistical probabilities** and must never be interpreted, marketed, or presented as "confidence" or "probability." All outputs require formal diagnostic evaluation and clinical confirmation by a qualified ophthalmologist.