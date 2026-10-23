<!-- Generated: 2026-10-24 | Files scanned: 3 | Token estimate: ~450 -->
# Dependencies & Environment Codemap

## Runtime Environments & Hardware

| Target | Hardware Specification | Runtime Profile | Purpose |
| :--- | :--- | :--- | :--- |
| **Training Target** | NVIDIA GeForce RTX 4060 (8 GB GDDR6) | PyTorch 2.6+ (`cu121`/`cu124`), AMP FP16, TF32 | Two-phase differential AdamW fine-tuning |
| **Inference Target** | Local x86_64 CPU (e.g. Intel Core i7-13650HX) | PyTorch CPU, `torch.no_grad()`, 14 threads | Sub-50ms CPU clinical screening & CLI |

## Core Dependency Manifest (`requirements.txt`)

| Package | Version Pin | Core Role in Pipeline |
| :--- | :--- | :--- |
| `torch` | $\ge 2.1.0$ | Deep learning runtime, tensor operations, ResNet-18 backbone |
| `torchvision` | $\ge 0.16.0$ | `IMAGENET1K_V1` weights, standard image transforms |
| `fastapi` | $\ge 0.110.0$ | Asynchronous REST API service for clinical screening & reports |
| `uvicorn` | $\ge 0.28.0$ | High-performance ASGI production web server |
| `pydantic` | $\ge 2.6.0$ | Data contract validation, configuration schemas, API models |
| `opencv-python` | $\ge 4.9.0$ | LAB CLAHE preprocessing, Laplacian blur & illumination checks |
| `reportlab` | $\ge 4.1.0$ | In-memory clinical PDF report generation (`io.BytesIO`) |
| `scikit-learn` | $\ge 1.4.0$ | AUROC, sensitivity, specificity, Youden's $J$ thresholding |
| `pandas` | $\ge 2.2.0$ | REFUGE manifest parsing, label mapping, CSV split audits |
| `numpy` | $\ge 1.26.0$ | Matrix calculations, colormap conversions, array operations |
| `pyyaml` | $\ge 6.0.1$ | Single-source configuration parsing (`configs/model_config.yaml`) |
| `pytest` | $\ge 8.0.0$ | Unit and integration test suite ($57$ tests) |
| `httpx` | $\ge 0.27.0$ | FastAPI `TestClient` HTTP endpoint verification |
