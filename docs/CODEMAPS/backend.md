<!-- Generated: 2026-10-24 | Files scanned: 8 | Token estimate: ~750 -->
# Backend Services & Inference Engine Codemap

## API Endpoints & Routes (`src/api.py`)

| Method | Path | Handler | Input Contract | Output / Stream |
| :--- | :--- | :--- | :--- | :--- |
| `GET, HEAD` | `/` | `index_html` | None | Static HTML5 Clinical Dashboard |
| `GET, HEAD` | `/health` | `health_check` | None | JSON: `{status, device, threads, threshold}` |
| `GET, HEAD` | `/api/v1/health` | `health_check_v1` | None | JSON: Alias of `/health` |
| `POST` | `/api/v1/predict` | `predict_image` | Multipart File (`file`), Query (`gradcam: bool`) | JSON: `{reference_id, predicted_class, suspicion_score, threshold, latency_ms, gradcam_base64, quality_warnings}` |
| `POST` | `/api/v1/report` | `generate_report` | Multipart File (`file`) | Streaming PDF: `application/pdf` (`io.BytesIO`) |
| `GET` | `/api/v1/evaluation` | `get_evaluation` | None | JSON: Held-out test metrics, confusion matrix, 95% CIs |

## Core Backend Modules

### 1. In-Memory Inference Engine (`src/inference.py`)
- **Class:** `GlaucomaInferenceEngine`
- **Responsibilities:**
  - Instantiates `torchvision.models.resnet18` with custom 1-logit classification head.
  - Loads weights (`artifacts/models/resnet18_refuge.pth`) and reads frozen metadata (`resnet18_refuge.meta.json`).
  - Executes deterministic CPU inference under `torch.no_grad()` with single-batch tensor inputs.
  - Computes `suspicion_score = float(torch.sigmoid(logits).item())`.
  - Classifies as `"Glaucoma Suspect"` if `suspicion_score >= threshold` else `"Normal"`.

### 2. Explainability Engine (`src/explainability.py`)
- **Class:** `GradCAM`
- **Target Layer:** `model.backbone.layer4[-1]`
- **Hook Hygiene:** Registers forward and backward PyTorch hooks; enforces guaranteed hook deregistration via `try...finally` to eliminate CPU/GPU memory leaks.
- **Colormap:** Generates jet overlay onto RGB image resized to $224 \times 224$.

### 3. Image Quality Heuristics (`src/quality_checks.py`)
- **Functions:**
  - `check_blur(img, threshold=100.0)`: OpenCV Laplacian variance evaluation.
  - `check_illumination(img, min_thresh=40.0, max_thresh=220.0)`: Grayscale mean luminance check.
  - `run_quality_checks(img)`: Aggregates non-blocking warnings (`list[str]`).

### 4. ReportLab PDF Engine (`src/report.py`)
- **Function:** `generate_clinical_pdf(image_bytes, result_dict)`
- **Output:** In-memory `io.BytesIO` containing clean, single-page clinical report with patient UUID, score meter, Grad-CAM panel, quality disclosures, and research disclaimer.

### 5. Headless CLI Runner (`src/cli.py`)
- **Single File:** `python -m src.cli --file path/to/fundus.jpg [--pdf] [--save-gradcam]`
- **Batch Directory:** `python -m src.cli --dir path/to/images/ [--json]`
