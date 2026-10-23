# Project Specification — Glaucoma Binary Screening System

## 1. Architectural Evolution & Engineering Synthesis

Based on a rigorous review of the VisionGuard design iterations and reference repositories (`Daramanohar/Glaucoma-detection`, `SaiHitesh16/Glaucoma-Detection-using-CNN`, `sabinagio/do-you-see-what-AI-see`), this specification locks a disciplined, high-reliability vertical slice:

1. **Elimination of Cascading Multi-Model Fragility:** Early iterations proposed chaining an Image Quality Assessment (IQA) CNN, a YOLO/UNet Optic Disc localizer, a dual-branch Cup-to-Disc Ratio (CDR) segmenter, and $10\times$ Monte Carlo (MC) Dropout passes. In production medical AI, each unverified upstream model compounds failure rates and bloats memory. All secondary neural networks are deferred in favor of deterministic OpenCV heuristic quality checks and a single, well-regularized classifier.
2. **Resolution of Label Noise & Calibration Distortion:** Merging "Suspect" cases into the positive class and applying Focal Loss ($\gamma = 2$) injects severe boundary noise while pushing logits toward extreme values ($0$ or $1$). This project enforces strict binary labels (`0 = Normal`, `1 = Glaucoma`, excluding `ungradable`/`suspect` samples) and uses **Weighted Binary Cross-Entropy** computed solely on the training split.
3. **Replacement of Streamlit with FastAPI + CLI:** Streamlit's top-to-bottom script re-execution model creates severe hazards in medical ML prototypes (accidental state mutation, hidden `@st.cache` retention of patient images, and tight coupling between UI rendering and inference logic). This system replaces Streamlit entirely with:
   - A stateless **FastAPI** backend (`src/api.py`) managing model lifecycle via an async `lifespan` context manager and serving a lightweight, zero-build **HTML5/CSS/JS local client** (`src/static/index.html`).
   - A deterministic **Command-Line Interface** (`src/cli.py`) for batch or single-image headless CPU inference and automated verification.
4. **High-Value Grafts Retained:**
   - **Two-Phase Fine-Tuning Schedule:** Head-only warmup (Phase A) followed by selective top-block unfreezing (`layer4` + `fc` in Phase B) to prevent gradient shock on small medical datasets.
   - **Outcome-Stratified Grad-CAM Error Analysis:** Automated generation of a 16-panel True Positive (TP), False Positive (FP), True Negative (TN), and False Negative (FN) Grad-CAM grid during test evaluation.

---

## 2. Hard Constraints (Non-Negotiable)

- **Framework & Backbone:** PyTorch only (`torch`, `torchvision`). Strictly `torchvision.models.resnet18(weights=IMAGENET1K_V1)`. Do not substitute ResNet-50, MobileNet, EfficientNet, or Vision Transformers.
- **Classification Head:** Replace `model.fc` with `nn.Sequential(nn.Dropout(p=0.3), nn.Linear(512, 1))` outputting a single raw logit $z \in (-\infty, \infty)$.
- **Dataset & Splits:** **REFUGE** only. No multi-dataset merging. Use the official $400 / 400 / 400$ (Train / Val / Test) split verbatim.
- **Label Policy:** `0 = Normal` (`Non-Glaucoma`), `1 = Glaucoma`. Any sample labeled `ungradable` or `suspect` must be excluded during Phase 2 manifest verification, and the resulting usable sample count $N$ must be updated across all reports.
- **Input & Normalization:** $224 \times 224$ RGB images normalized with ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$). Single source of truth: `configs/model_config.yaml`.
- **Training Hardware Optimization:** Configured for **NVIDIA RTX 4060 (8 GB VRAM)** using TensorFloat-32 (`allow_tf32 = True`) and Automatic Mixed Precision (`torch.amp.autocast('cuda', dtype=torch.float16)`).
- **Inference Runtime:** Local x86_64 CPU (target: Intel Core i7-13650HX), `model.eval()`, `torch.no_grad()`, $\text{batch\_size} = 1$, thread-pinned to physical cores.
- **UI Framework Constraint:** **No Streamlit.** Do not import `streamlit` anywhere in the repository. Use FastAPI + Uvicorn and the CLI runner.
- **Decision Threshold Governance:** Computed once on the validation set via **Youden's $J$ statistic**, serialized into `artifacts/models/resnet18_refuge.meta.json`, and loaded read-only during evaluation and inference.
- **Terminology & Calibration Honesty:** Model scores must be labeled **"Glaucoma Suspicion Score"**. The words *"confidence"* and *"probability"* are strictly banned in API keys, UI labels, CLI output, and PDF reports.
- **Governance Boundary:** `.agents/rules/` must remain untouched. All project specifications reside in `docs/`.

---

## 3. Mathematical Formulation

### 3.1 Loss Function & Prior Shift
Given training split counts $N_{\text{neg}}$ (`Normal`) and $N_{\text{pos}}$ (`Glaucoma`), the positive class weight is computed exclusively on the training split:

$$w_{\text{pos}} = \frac{N_{\text{neg}}}{N_{\text{pos}}}$$

For a raw logit $z_i$ and binary target $y_i \in \{0, 1\}$, the loss is:

$$\mathcal{L}(z_i, y_i) = -\left[ w_{\text{pos}} \, y_i \ln \sigma(z_i) + (1 - y_i) \ln(1 - \sigma(z_i)) \right]$$

where $\sigma(z_i) = \frac{1}{1 + e^{-z_i}}$. Because $w_{\text{pos}} > 1$ shifts the Bayes-optimal logit by $+\ln(w_{\text{pos}})$, $\sigma(z_i)$ is **not** an empirical posterior probability. It is reported strictly as a continuous **Glaucoma Suspicion Score** $s_i = \sigma(z_i) \in [0, 1]$.

### 3.2 Optimal Threshold Selection (Youden's $J$)
On the held-out **validation split** only, let $\text{TPR}(t)$ be sensitivity and $\text{TNR}(t) = 1 - \text{FPR}(t)$ be specificity at threshold $t \in [0, 1]$. The optimal threshold $t^*$ maximizes Youden's $J$ index:

$$t^* = \arg\max_{t \in [0, 1]} \left( \text{TPR}_{\text{val}}(t) + \text{TNR}_{\text{val}}(t) - 1 \right)$$

---

## 4. Two-Phase Training Protocol (RTX 4060 8GB)

### Phase A — Classification Head Warmup
- **Parameter Freeze:** Freeze all parameters (`param.requires_grad = False`) except `model.fc`.
- **Optimizer:** `AdamW` ($\text{lr} = 1.0 \times 10^{-3}$, $\text{weight\_decay} = 1.0 \times 10^{-4}$).
- **Duration:** $8$ epochs, $\text{batch\_size} = 64$, mixed precision (`FP16`) enabled.
- **Gate to Phase B:** Validation $\text{AUROC} > 0.75$. If Phase A fails to exceed $0.75$, halt and audit the data loader and label mapping before spending Phase B compute.

### Phase B — Selective Representation Adaptation
- **Parameter Unfreeze:** Unfreeze `model.layer4` and `model.fc`. Keep `conv1`, `bn1`, `layer1`, `layer2`, and `layer3` frozen to preserve low/mid-level vascular and edge filters.
- **Optimizer:** `AdamW` ($\text{lr} = 1.0 \times 10^{-5}$, $\text{weight\_decay} = 1.0 \times 10^{-4}$).
- **Scheduler:** `ReduceLROnPlateau` monitoring `val_loss` ($\text{mode} = \text{'min'}$, $\text{factor} = 0.5$, $\text{patience} = 3$).
- **Duration & Early Stopping:** Up to $20$ epochs; early stopping monitoring `val_auroc` ($\text{mode} = \text{'max'}$, $\text{patience} = 8$).
- **Checkpoint Rule:** Save `state_dict` at the epoch achieving maximum `val_auroc`.

---

## 5. Metric Floor & Adjustment Clause (Phase 4 Gate)

Before running final test-set evaluation, the best Phase B model must satisfy on the **validation set**:
1. $\text{AUROC}_{\text{val}} > 0.85$
2. $\text{Sensitivity}_{\text{val}}(t^*) > 0.80$

**Sensitivity Adjustment Clause:**  
If $\text{AUROC}_{\text{val}} \ge 0.85$ but $\text{Sensitivity}_{\text{val}}(t^*) < 0.80$ at the Youden threshold $t^*$, do **not** enter an infinite retraining loop. Select the highest threshold $t_{\text{adj}} < t^*$ on the validation ROC curve such that $\text{Sensitivity}_{\text{val}}(t_{\text{adj}}) \ge 0.80$, record `"threshold_procedure": "sensitivity_priority_80_on_val"` and the resulting specificity trade-off in `resnet18_refuge.meta.json`, and proceed to test evaluation. If $\text{AUROC}_{\text{val}} < 0.85$, retrain.

---

## 6. Repository Structure & Deliverables

```text
glaucoma-screening-pytorch/
├── configs/
│   └── model_config.yaml                   # Single source of truth for train/eval/serve
├── data/
│   ├── raw/                                # Gitignored raw REFUGE archive
│   └── processed/                          # Manifests and split CSVs
├── artifacts/
│   ├── models/                             # Gitignored .pth weights; tracked .meta.json
│   └── reports/
│       ├── data_audit.json                 # Phase 2 manifest verification output
│       ├── training_curves.png             # Loss & AUROC curves across Phase A + B
│       ├── evaluation.json                 # Machine-readable test metrics & 95% CIs
│       ├── evaluation.md                   # Human-readable clinical/engineering report
│       └── gradcam_grid.png                # 16-panel TP / FP / TN / FN Grad-CAM grid
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── load_refuge.py                  # Parses REFUGE release into manifest.csv
│   │   ├── validate_data.py                # Corrupt scan, ungradable filter, count audit
│   │   ├── create_splits.py                # Verifies & emits official 400/400/400 splits
│   │   └── dataset.py                      # PyTorch Dataset class reading config transforms
│   ├── preprocessing.py                    # Deterministic resize, normalization, optional CLAHE
│   ├── quality_checks.py                   # OpenCV Laplacian blur & luminance warnings
│   ├── model.py                            # ResNet-18 factory + phase freeze/unfreeze helpers
│   ├── train.py                            # Two-phase RTX 4060 AMP training loop
│   ├── evaluate.py                         # Youden's J, bootstrap CIs, latency, Grad-CAM grid
│   ├── explainability.py                   # Standalone PyTorch hook Grad-CAM on layer4[-1]
│   ├── inference.py                        # Pure CPU inference engine reading .meta.json
│   ├── report.py                           # ReportLab PDF generator (in-memory BytesIO)
│   ├── cli.py                              # Headless CLI inference runner
│   ├── api.py                              # FastAPI application (REST + static UI mount)
│   └── static/
│       └── index.html                      # Zero-build HTML5/JS clinical screening UI
├── tests/
│   ├── test_config_schema.py
│   ├── test_preprocessing.py
│   ├── test_label_mapping.py
│   ├── test_checkpoint_metadata.py
│   ├── test_inference_errors.py
│   ├── test_threshold_loading.py
│   └── test_api_endpoints.py
├── docs/
│   ├── PROJECT_SPEC.md
│   ├── architecture.md
│   ├── dataset_protocol.md
│   ├── evaluation_protocol.md
│   ├── rejected_patterns.md
│   └── CODEMAPS/
│       ├── architecture.md
│       ├── backend.md
│       ├── data.md
│       └── dependencies.md
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 7. Implementation Phase Plan & Acceptance Checklist

| Phase | Milestone | Verification Gate |
| :--- | :--- | :--- |
| **Phase 0** | Environment, CUDA 12.1 verification on RTX 4060, `requirements.txt` | `pytest -q` executes cleanly; CUDA & CPU thread checks pass |
| **Phase 1** | Config schema (`model_config.yaml`), directory skeleton, unit test harness | `test_config_schema.py` passes |
| **Phase 2** | REFUGE manifest parser, `ungradable` check, official $400/400/400$ splits | `data_audit.json` emitted; `docs/dataset_protocol.md` table filled; zero path overlap |
| **Phase 3a** | Phase A (`fc`-only warmup, 8 epochs on RTX 4060) | Checkpoint loads in isolated process; $\text{AUROC}_{\text{val}} > 0.75$ |
| **Phase 3b** | Phase B (`layer4` + `fc` fine-tuning, up to 20 epochs) | Metric floor met ($\text{AUROC}_{\text{val}} > 0.85$, $\text{Sens}_{\text{val}} \ge 0.80$) |
| **Phase 4** | Held-out test evaluation, Youden threshold freeze, $1,000$ bootstrap CIs, Grad-CAM grid | `evaluation.{json,md}` and `gradcam_grid.png` generated from single CLI command |
| **Phase 5** | CPU `inference.py`, `cli.py`, and FastAPI service (`api.py` + `static/index.html`) | Corrupt file & missing checkpoint tests pass; `test_api_endpoints.py` green; zero Streamlit imports |
| **Phase 6** | In-memory ReportLab PDF export, `/verification-loop`, `/orch-review` | All 12 acceptance criteria below verified |

### Final Acceptance Checklist
- [ ] Zero references to `streamlit` in code or dependencies; FastAPI (`uvicorn src.api:app`) and CLI (`python -m src.cli`) launch cleanly.
- [ ] `POST /api/v1/predict` and `src/cli.py` accept valid JPG/PNG files and reject corrupt or misnamed files (e.g., `.txt` renamed to `.jpg`) with structured HTTP 422 / CLI exit codes rather than raw tracebacks.
- [ ] OpenCV heuristic quality warnings (blur, illumination) are returned in a distinct `quality_warnings` array, separate from the diagnostic suspicion score.
- [ ] Output payload and UI display `predicted_class`, `suspicion_score`, `threshold_used`, and `latency_ms`.
- [ ] Neither FastAPI startup nor image upload triggers model training; missing `.pth` or `.meta.json` raises a structured `CheckpointNotFoundError`.
- [ ] Image preprocessing in `train.py`, `evaluate.py`, `cli.py`, and `api.py` imports the exact same function from `src/preprocessing.py` driven by `configs/model_config.yaml`.
- [ ] `artifacts/reports/evaluation.{json,md}` is regenerable via `python -m src.evaluate --config configs/model_config.yaml`.
- [ ] Test set $N$, class prevalence, raw confusion matrix counts, and $95\%$ bootstrap confidence intervals are disclosed in the report and UI evaluation tab.
- [ ] `artifacts/reports/gradcam_grid.png` renders TP, FP, TN, and FN panels hooked from `model.layer4[-1]`.
- [ ] CPU inference latency ($P_{50}$ and $P_{95}$) is empirically measured over $100$ warm runs on the target CPU and recorded in `evaluation.json`.
- [ ] Uploaded images are processed strictly in-memory (`io.BytesIO`) with zero disk persistence of patient images.
- [ ] Research-prototype disclaimer is visible on the FastAPI web dashboard, CLI output, and generated PDF reports.