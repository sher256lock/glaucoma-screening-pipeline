# Glaucoma Binary Screening Pipeline — PyTorch ResNet-18 / REFUGE

Research and educational prototype for binary glaucoma screening from retinal fundus photographs.  
**Not a clinically validated diagnostic device. Not for clinical decision-making.**

---

## 1. Executive Summary

| Dimension | Specification |
| :--- | :--- |
| **Backbone Architecture** | `torchvision.models.resnet18(weights=IMAGENET1K_V1)` |
| **Classification Head** | `nn.Sequential(nn.Dropout(p=0.3), nn.Linear(512, 1))` |
| **Dataset & Split Protocol** | **REFUGE** (Retinal Fundus Glaucoma Challenge), official $400 / 400 / 400$ split |
| **Training Hardware Target** | Local **NVIDIA GeForce RTX 4060 (8 GB GDDR6)** with Tensor Cores (`TF32` / `AMP FP16`) |
| **Inference Hardware Target** | Local **x86_64 CPU** (e.g., Intel Core i7-13650HX), `model.eval()`, `torch.no_grad()`, $\text{batch\_size} = 1$ |
| **Inference Surface** | **FastAPI + Uvicorn** REST API & zero-build static HTML5/JS Local Dashboard + **Headless CLI** (*No Streamlit*) |
| **Explainability** | **Grad-CAM (`model.layer4[-1]`)** + Bilinear Heatmap Overlay |

---

## 2. Documentation Index (`docs/`)

All project governance and architectural specifications reside in `docs/`:

1. [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md) — Canonical specification, hard constraints, two-phase training schedule, and phase gates.
2. [`docs/architecture.md`](docs/architecture.md) — System data flow, FastAPI & CLI inference surfaces, module boundaries, JSON/Pydantic contracts, and in-memory PHI lifecycle.
3. [`docs/dataset_protocol.md`](docs/dataset_protocol.md) — REFUGE acquisition, Phase 2 manifest verification gate, `ungradable` exclusion rules, and patient-ID leakage disclosure.
4. [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md) — Statistical evaluation metrics, Youden's $J$ threshold freezing, $1,000$-resample bootstrap $95\%$ CIs, and TP/FP/TN/FN Grad-CAM error analysis.
5. [`docs/CODEMAPS/`](docs/CODEMAPS/) — Token-lean architecture, backend services, data pipeline, and dependency codemaps.
6. [`docs/rejected_patterns.md`](docs/rejected_patterns.md) — Explicit architectural anti-patterns and forbidden constructs (including the ban on Streamlit, ResNet-50, and uncalibrated probability claims).

---

## 3. Environment & Hardware Setup

### 3.1 Python Environment Creation & Activation
Requires Python $3.10$, $3.11$, $3.12$, or $3.13$.

```bash
# 1. Create the virtual environment
python -m venv .venv

# 2. Activate the virtual environment:
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# (Note: If script execution is restricted in PowerShell, run first: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)

# Linux / macOS:
source .venv/bin/activate
```

> [!TIP]
> **Windows PowerShell Path Syntax Gotcha**:  
> In PowerShell, executing a script or binary using a relative path requires the leading `.\` prefix (e.g., `.\.venv\Scripts\python.exe`). Omitting `.\` causes PowerShell's parser to interpret `.venv` as a module name to import, resulting in:  
> `The module '.venv' could not be loaded. For more information, run 'Import-Module .venv'.`

### 3.2 PyTorch with CUDA (Hardware Acceleration)
Install CUDA-enabled PyTorch wheels to leverage NVIDIA Tensor Cores (e.g., RTX 4060 Ada Lovelace `SM 8.9`):

```bash
# Windows / Linux with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Or for CUDA 12.1:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
*(On CPU-only systems, standard `pip install torch torchvision` is sufficient).*

### 3.3 Core Dependencies (FastAPI, OpenCV, Scikit-Learn, ReportLab, Pytest)
```bash
pip install -r requirements.txt
```

---

## 4. Model Checkpoint & Automated Weights Protocol

> [!NOTE]
> **Zero-Bloat Automated Weight Acquisition**:  
> Binary weights (`artifacts/models/resnet18_refuge.pth`, ~44.8 MB) are excluded from Git via `.gitignore` to keep the repository lightweight.  
> `GlaucomaInferenceEngine` includes an **automatic fetcher**: on startup, if weights are missing, it downloads them from the configured GitHub Release asset (`checkpoint.download_url` in `configs/model_config.yaml`), verifies the SHA-256 checksum, and performs an atomic save.

You can also acquire or manage model weights manually via any of the following methods:

### Option A — Automated CLI Fetcher & Integrity Verifier (Recommended)
Download the verified release checkpoint directly via the CLI utility:
```bash
# Automated download and SHA-256 checksum verification:
python -m src.download_weights

# Force re-download:
python -m src.download_weights --force
```

### Option B — Instant Offline Baseline Initialization (Zero-Download UI Testing)
To test the web UI, Grad-CAM overlays, and PDF reporting without downloading or training on full weights:
```bash
python -m src.download_weights --init-baseline
```

### Option C — Deploy an Existing Trained Checkpoint
If you possess a local `resnet18_refuge.pth` file, place it directly into:
```text
artifacts/models/resnet18_refuge.pth
```

### Option D — Full Retraining on REFUGE (GPU Accelerated)
To reproduce the full model from scratch on the official REFUGE dataset (400 train / 400 val / 400 test split):
```bash
# 1. Place raw REFUGE images in data/raw/refuge/ and generate audited splits:
python -m src.data.load_refuge --raw-dir data/raw/refuge --output data/processed/manifest.csv
python -m src.data.validate_data --manifest data/processed/manifest.csv --report artifacts/reports/data_audit.json
python -m src.data.create_splits --manifest data/processed/manifest.csv --output-dir data/processed/splits

# 2. Execute Two-Phase Training on RTX 4060 (Phase A: Warmup, Phase B: Selective Fine-Tuning):
python -m src.train --config configs/model_config.yaml

# 3. Evaluate held-out test cohort, freeze Youden's J threshold, and generate Grad-CAM grid:
python -m src.evaluate --config configs/model_config.yaml
```

---

## 5. End-to-End Execution Commands

### Step 1: Verify Hardware & Test Harness (Phase 0 & 1)
```bash
# Check PyTorch device and CUDA availability:
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Run full test suite (55 unit tests):
pytest -q
```

### Step 2: Local CPU Inference & Clinical Deployment (Phase 5 & 6 — No Streamlit)

#### Surface 1 — Local FastAPI REST Service + Zero-Build HTML5 Clinical Dashboard:
```bash
# With activated virtual environment:
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Direct invocation in Windows PowerShell (without activating .venv):
.\.venv\Scripts\python.exe -m uvicorn src.api:app --host 0.0.0.0 --port 8000

# Direct invocation in Linux / macOS (without activating .venv):
.venv/bin/python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```
- **Web Clinical Dashboard**: Open `http://localhost:8000` in any browser to access the interactive screening dashboard with in-memory upload, score meter, Grad-CAM viewer, and instant PDF download.
- **REST Endpoints**:
  - `POST /api/v1/predict`: In-memory screening returning Glaucoma Suspicion Score, threshold, and Base64 Grad-CAM overlay.
  - `POST /api/v1/report`: In-memory generation of single-page clinical PDF report.
  - `GET /api/v1/evaluation`: Official held-out test evaluation report and bootstrap 95% CIs.
  - `GET /health`: Engine status, CPU threads, device, and frozen threshold.

#### Surface 2 — Headless CLI Inference (Single Image & Batch with Grad-CAM & PDF Report):
```bash
# Single image with clinical PDF report:
python -m src.cli --file path/to/fundus.jpg --pdf

# Single image with explicit PDF and Grad-CAM output paths:
python -m src.cli --file path/to/fundus.jpg --save-gradcam artifacts/reports/sample_cam.png --pdf artifacts/reports/sample_report.pdf

# Batch directory screening:
python -m src.cli --dir path/to/fundus_dir/
```

---

## 6. Mandatory Regulatory & Statistical Disclaimer

This system is an **educational and research prototype**. Model outputs are reported strictly as a **"Glaucoma Suspicion Score"** derived from a class-weighted binary cross-entropy objective (`BCEWithLogitsLoss`). Because class weighting shifts the prior log-odds, these scores **are not calibrated statistical probabilities** and must never be presented as "confidence" or "probability." All outputs require review by a qualified ophthalmologist.