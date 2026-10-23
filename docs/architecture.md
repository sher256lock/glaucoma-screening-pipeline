# System Architecture & Interface Contracts

## 1. High-Level Data Flow

### 1.1 Offline Training & Evaluation Pipeline (NVIDIA RTX 4060 8GB GPU)

```text
[REFUGE Raw Archive]
        │
        ▼
┌──────────────────────────────┐
│ src/data/load_refuge.py      │ ──► Parses directory tree & official ground-truth annotations
│ src/data/validate_data.py    │ ──► Excludes 'ungradable'/'suspect'; scans for corrupt images
│ src/data/create_splits.py    │ ──► Emits train.csv, val.csv, test.csv (Official 400/400/400)
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ src/train.py (RTX 4060 GPU)  │
│  • TF32 + AMP (FP16) enabled │
│  • Compute pos_weight (train)│
│  • Phase A: fc only (8 ep)   │ ──► Gate: Val AUROC > 0.75
│  • Phase B: layer4+fc (20 ep)│ ──► Saves best val AUROC state_dict
└──────────────┬───────────────┘
               │
               ├──► artifacts/models/resnet18_refuge.pth
               ├──► artifacts/reports/training_curves.png
               ▼
┌──────────────────────────────┐
│ src/evaluate.py              │
│  • Computes Youden's J on Val│ ──► Writes threshold into resnet18_refuge.meta.json
│  • Evaluates frozen Test set │ ──► 1,000-iteration bootstrap 95% CIs
│  • Benchmarks CPU P50/P95 ms │ ──► 100 warm runs (batch=1, physical cores)
│  • Hooks model.layer4[-1]    │ ──► Generates 16-panel TP/FP/TN/FN Grad-CAM grid
└──────────────┬───────────────┘
               │
               ├──► artifacts/models/resnet18_refuge.meta.json
               ├──► artifacts/reports/evaluation.json & evaluation.md
               └──► artifacts/reports/gradcam_grid.png
```

### 1.2 Online Local CPU Inference Pipeline (FastAPI + CLI — No Streamlit)

```text
   [Client Upload (HTTP Multipart) or CLI File Path]
                          │
                          ▼
┌───────────────────────────────────────────────────┐
│ 1. In-Memory Ingestion & Validation               │
│    • Verify MIME/extension (.jpg, .jpeg, .png)    │
│    • Decode via PIL/OpenCV from io.BytesIO        │
│    • Verify 3-channel RGB & min dimensions (>=64) │
└─────────────────────────┬─────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────┐
│ 2. Non-Blocking Heuristic Quality Checks          │
│    • src/quality_checks.py (OpenCV)               │
│    • Variance of Laplacian (blur warning)         │
│    • Mean/Std grayscale intensity (illumination)  │
└─────────────────────────┬─────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────┐
│ 3. Config-Driven Preprocessing                    │
│    • src/preprocessing.py (reads model_config)    │
│    • Optional Green-channel/CLAHE (if trained)    │
│    • Bilinear resize to 224x224, ImageNet norm    │
└─────────────────────────┬─────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────┐
│ 4. CPU Inference & Grad-CAM Attribution           │
│    • src/inference.py + src/explainability.py     │
│    • ResNet-18 eval(), torch.no_grad()            │
│    • Sigmoid -> Glaucoma Suspicion Score          │
│    • Compare against frozen threshold from meta   │
│    • Optional Grad-CAM hook on model.layer4[-1]   │
└─────────────────────────┬─────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────┐
│ 5. Response Delivery (Zero Disk Retention)        │
│    • FastAPI JSON Response + Base64 Grad-CAM PNG  │
│    • In-memory ReportLab PDF stream (/api/v1/pdf) │
│    • Or CLI structured stdout / file export       │
└───────────────────────────────────────────────────┘
```

---

## 2. Hardware Optimization Architecture

### 2.1 NVIDIA RTX 4060 (8 GB VRAM) Training Configuration
The RTX 4060 Mobile/Desktop (`AD107`, Compute Capability `8.9`) features 4th-Generation Tensor Cores and 24 MB of L2 cache, with a 128-bit memory bus (256 GB/s). To maximize throughput without hitting 8 GB VRAM ceilings:
- **Mixed Precision (`AMP`):** `train.py` wraps the forward pass in `torch.amp.autocast(device_type="cuda", dtype=torch.float16)` with `torch.cuda.amp.GradScaler()`. At $224 \times 224$ with $\text{batch\_size} = 64$, ResNet-18 consumes $< 1.8\text{ GB}$ VRAM, leaving ample headroom and saturating Tensor Cores.
- **TF32 Matmul:** `torch.backends.cuda.matmul.allow_tf32 = True` and `torch.backends.cudnn.allow_tf32 = True` are enabled for any non-FP16 fallback operations.
- **DataLoader Pipeline:** `num_workers = 4`, `pin_memory = True`, `persistent_workers = True` to eliminate CPU-to-GPU PCIe starvation during training.

### 2.2 Local CPU Inference Optimization (Intel Core i7-13650HX)
For single-image screening ($\text{batch\_size} = 1$), thread contention across hybrid P-cores and E-cores can degrade latency if left unconfigured:
- **Thread Pinning:** At startup, `src/inference.py` calls `torch.set_num_threads(psutil.cpu_count(logical=False) or 8)` to bind PyTorch intra-op parallelism to physical cores.
- **Singleton Lifespan Loading:** In `src/api.py`, the `GlaucomaPredictor` instance loads `resnet18_refuge.pth` and `resnet18_refuge.meta.json` **once** inside FastAPI's `@asynccontextmanager def lifespan(app: FastAPI):` block. No weights are reloaded per request.

---

## 3. Module Boundaries & Responsibilities

| Module | Responsibility | Forbidden Actions |
| :--- | :--- | :--- |
| `src/data/*` | Dataset parsing, manifest generation, corrupt image detection, split creation. | Importing `torchvision.models` or modifying image pixels. |
| `src/preprocessing.py` | Deterministic tensor transformation from `PIL.Image` / `np.ndarray` to $(1, 3, 224, 224)$ tensor based strictly on `model_config.yaml`. | Hardcoding mean/std or image dimensions outside config. |
| `src/quality_checks.py` | Computing Laplacian variance ($\text{blur\_score}$) and mean luminance ($\text{brightness\_score}$) and returning human-readable warning strings. | Blocking inference or altering classification scores. |
| `src/model.py` | Instantiating `resnet18`, replacing `model.fc` with `Dropout(0.3) -> Linear(512, 1)`, and toggling `requires_grad` for Phase A and Phase B. | Running training loops or file I/O. |
| `src/train.py` | Executing Phase A and Phase B training, logging metrics, saving best `.pth` weights, initial `.meta.json`, and `training_curves.png`. | Evaluating on the held-out `test` split. |
| `src/evaluate.py` | Computing Youden's $J$ on `val`, freezing `threshold` into `.meta.json`, computing test metrics + bootstrap CIs, benchmarking CPU latency, and saving `gradcam_grid.png`. | Modifying model weights or tuning hyperparameters on `test`. |
| `src/explainability.py` | Registering forward and backward hooks on `model.layer4[-1]`, computing channel-weighted ReLU activation maps, bilinearly upsampling $7 \times 7 \to 224 \times 224$, and generating colormap overlays. | Using external black-box Grad-CAM wrappers that obscure layer alignment. |
| `src/inference.py` | Loading frozen `.pth` + `.meta.json`, validating schema compatibility, running `eval()` / `no_grad()` forward pass, and optionally invoking `explainability.py`. | Triggering training, recomputing thresholds, or silently falling back to random/ImageNet weights. |
| `src/report.py` | Rendering a single-page clinical screening summary PDF into an in-memory `io.BytesIO` buffer using ReportLab. | Writing patient images or PDFs to persistent server storage in API mode. |
| `src/api.py` & `src/static/index.html` | Serving REST endpoints (`/api/v1/health`, `/api/v1/predict`, `/api/v1/metrics`, `/api/v1/report`) and the local browser UI. | Using Streamlit or persisting uploaded bytes to disk. |
| `src/cli.py` | Providing a `argparse`-driven CLI for headless single-image or directory inference and automated test harness verification. | Modifying model checkpoints. |

---

## 4. Strict Data & API Contracts

### 4.1 Checkpoint Metadata Contract (`artifacts/models/resnet18_refuge.meta.json`)
```json
{
  "arch": "resnet18",
  "framework": "pytorch",
  "pytorch_version": "2.2.2+cu121",
  "input_size": 224,
  "channels": 3,
  "normalization": {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  },
  "clahe": false,
  "dataset": "REFUGE",
  "dataset_version": "2026-10-22",
  "usable_n": {
    "train": 400,
    "val": 400,
    "test": 400
  },
  "split_protocol": "official_400_400_400",
  "pos_weight": 9.0,
  "threshold": 0.4182,
  "threshold_procedure": "youden_j_on_val",
  "val_auroc": 0.924,
  "val_sensitivity_at_threshold": 0.875,
  "val_specificity_at_threshold": 0.861,
  "git_sha": "a1b2c3d",
  "config_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "best_epoch": 14,
  "training_phase": "B",
  "created_at": "2026-10-23T00:00:00Z"
}
```

### 4.2 FastAPI Prediction Response Schema (`POST /api/v1/predict`)
```json
{
  "reference_id": "req_8f92a1c4",
  "predicted_class": "Glaucoma",
  "suspicion_score": 0.8142,
  "threshold_used": 0.4182,
  "latency_ms": 54.3,
  "quality_warnings": [
    "Low illumination detected (mean intensity: 38.2 < 45.0). Verify fundus exposure."
  ],
  "gradcam_overlay_base64": "iVBORw0KGgoAAAANSUhEUgAAAOAAAADg...",
  "model_version": "resnet18-a1b2c3d",
  "disclaimer": "RESEARCH PROTOTYPE ONLY. Glaucoma Suspicion Score is not a calibrated probability and is not for clinical diagnosis."
}
```

### 4.3 Evaluation Report Schema (`artifacts/reports/evaluation.json`)
```json
{
  "test_n": 400,
  "test_prevalence_glaucoma": 0.10,
  "threshold_used": 0.4182,
  "threshold_procedure": "youden_j_on_val",
  "metrics": {
    "auroc": { "value": 0.918, "ci95": [0.872, 0.958] },
    "sensitivity": { "value": 0.850, "ci95": [0.732, 0.949] },
    "specificity": { "value": 0.867, "ci95": [0.831, 0.901] },
    "f1": { "value": 0.618, "ci95": [0.505, 0.721] },
    "brier_diagnostic_only": 0.112
  },
  "confusion_matrix": {
    "tp": 34,
    "fp": 48,
    "tn": 312,
    "fn": 6
  },
  "latency_ms_cpu": {
    "p50": 52.4,
    "p95": 68.1,
    "n_runs": 100,
    "threads": 14,
    "device": "cpu"
  },
  "limitations": [
    "Single-dataset training (REFUGE). No external validation cohort.",
    "REFUGE public release lacks patient-level identifiers; official 400/400/400 split used; inter-eye contamination cannot be ruled out.",
    "Test N=400 (40 positive cases) yields wide 95% bootstrap confidence intervals.",
    "Glaucoma Suspicion Score is derived from pos_weight-adjusted BCE and is not a calibrated statistical probability."
  ]
}
```