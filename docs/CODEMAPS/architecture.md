<!-- Generated: 2026-10-24 | Files scanned: 22 | Token estimate: ~650 -->
# System Architecture Codemap

## High-Level System Topology

```
[ Retinal Fundus Image (JPG/PNG) ]
             │
             ▼
  ┌──────────────────────────────────────────────┐
  │         Ingestion & Validation Boundary      │
  │   - Magic Byte & PIL/OpenCV Decode Validation│
  │   - In-Memory BytesIO Stream (Zero Disk PHI) │
  │   - Quality Heuristics: Blur & Illumination  │
  └──────────────────────┬───────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────┐
  │         Inference Engine (CPU x86_64)        │
  │   - Preprocessing: LAB CLAHE + Resize + Norm │
  │   - ResNet-18 Backbone (TorchScript/Eval)    │
  │   - Single-Logit Head -> Sigmoid Activation  │
  │   - Glaucoma Suspicion Score Evaluation      │
  │   - Frozen Youden's J Threshold (t* = 0.2921)│
  └──────────────────────┬───────────────────────┘
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
┌─────────────────────────┐ ┌─────────────────────────┐
│ Explainability Engine   │ │ Reporting & Deployment  │
│ - Grad-CAM layer4[-1]   │ │ - FastAPI REST Engine   │
│ - 7x7 -> 224x224 Overlay│ │ - Zero-Build HTML5 UI   │
│ - Base64 Heatmap Stream │ │ - ReportLab PDF Engine  │
│                         │ │ - Headless Batch CLI    │
└─────────────────────────┘ └─────────────────────────┘
```

## System Service Boundaries

| Service Boundary | Primary Modules | Data Contracts / I/O | In-Memory Guarantees |
| :--- | :--- | :--- | :--- |
| **Data & Preprocessing** | `src/data/`, `src/preprocessing.py` | PIL Image / NumPy $\to$ Float32 Tensor $(1, 3, 224, 224)$ | Deterministic single-source transforms |
| **Model & Inference** | `src/model.py`, `src/inference.py` | Tensor $\to$ Glaucoma Suspicion Score $[0, 1]$, Classification, Latency | Thread-safe CPU inference; loads `.meta.json` |
| **Explainability** | `src/explainability.py` | Forward/Backward Hook on `layer4[-1]` $\to$ Heatmap $(224, 224, 3)$ | Strict hook removal in `finally` blocks |
| **Reporting & Quality** | `src/quality_checks.py`, `src/report.py` | Raw Image Bytes $\to$ In-Memory PDF (`io.BytesIO`) | Zero disk persistence; No PHI stored |
| **Serving Interfaces** | `src/api.py`, `src/cli.py`, `src/static/` | HTTP Multipart / CLI Args $\to$ JSON, Stream, HTML5 | Non-blocking streaming responses |
