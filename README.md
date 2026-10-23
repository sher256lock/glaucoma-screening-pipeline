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

### 3.1 Python Environment
Requires Python $3.10$ or $3.11$.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate
```

### 3.2 PyTorch with CUDA 12.1 (NVIDIA RTX 4060 8GB)
Install the CUDA-enabled PyTorch wheels to unlock the RTX 4060 Ada Lovelace Tensor Cores (`SM 8.9`):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3.3 Core Dependencies (FastAPI, OpenCV, Scikit-Learn, ReportLab, Pytest)
```bash
pip install -r requirements.txt
```

---

## 4. End-to-End Execution Commands

### Step 1: Verify Hardware & Test Harness (Phase 0 & 1)
```bash
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
pytest -q
```

### Step 2: REFUGE Manifest Audit & Split Generation (Phase 2 Gate)
*Must be completed and recorded in `docs/dataset_protocol.md` before training.*
```bash
python -m src.data.load_refuge --raw-dir data/raw/refuge --output data/processed/manifest.csv
python -m src.data.validate_data --manifest data/processed/manifest.csv --report artifacts/reports/data_audit.json
python -m src.data.create_splits --manifest data/processed/manifest.csv --output-dir data/processed/splits
```

### Step 3: Two-Phase Model Training on RTX 4060 (Phase 3a & 3b)
```bash
python -m src.train --config configs/model_config.yaml
```

### Step 4: Held-Out Evaluation, Threshold Freezing & Grad-CAM Grid (Phase 4)
```bash
python -m src.evaluate --config configs/model_config.yaml
```

### Step 5: Local CPU Inference & Clinical Deployment (Phase 5 & 6 — No Streamlit)
**Option A — Headless CLI Inference (Single Image & Batch with Grad-CAM & PDF Report):**
```bash
# Single image with clinical PDF report:
python -m src.cli --file path/to/fundus.jpg --pdf

# Single image with explicit PDF and Grad-CAM output paths:
python -m src.cli --file path/to/fundus.jpg --save-gradcam artifacts/reports/sample_cam.png --pdf artifacts/reports/sample_report.pdf

# Batch directory screening:
python -m src.cli --dir path/to/fundus_dir/
```

**Option B — Local FastAPI REST Service + Zero-Build HTML5 Clinical Dashboard:**
```bash
# With activated virtual environment:
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Direct invocation in Windows PowerShell (if virtualenv is not activated):
.venv\Scripts\python.exe -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```
- Web Clinical UI: Open `http://localhost:8000` in any browser to access the interactive screening dashboard with in-memory upload, score meter, Grad-CAM viewer, and instant PDF download.
- REST Endpoints:
  - `POST /api/v1/predict`: In-memory screening returning Glaucoma Suspicion Score, threshold, and Base64 Grad-CAM overlay.
  - `POST /api/v1/report`: In-memory generation of single-page clinical PDF report.
  - `GET /api/v1/evaluation`: Official held-out test evaluation report and bootstrap 95% CIs.
  - `GET /health`: Engine status, CPU threads, device, and frozen threshold.

---

## 5. Mandatory Regulatory & Statistical Disclaimer

This system is an **educational and research prototype**. Model outputs are reported strictly as a **"Glaucoma Suspicion Score"** derived from a class-weighted binary cross-entropy objective (`BCEWithLogitsLoss`). Because class weighting shifts the prior log-odds, these scores **are not calibrated statistical probabilities** and must never be presented as "confidence" or "probability." All outputs require review by a qualified ophthalmologist.