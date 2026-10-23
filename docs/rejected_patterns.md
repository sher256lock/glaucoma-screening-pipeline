# Rejected Patterns & Negative Constraints

If any implementation proposal, pull request, or agent action introduces a pattern listed below, reject it immediately and refer back to `docs/PROJECT_SPEC.md`.

---

## 1. UI & Serving Framework Anti-Patterns

- **Streamlit (`import streamlit as st`):** **STRICTLY FORBIDDEN.** Streamlit is excluded from this project due to top-to-bottom script re-execution hazards, state management fragility, and `@st.cache` memory retention risks. All web serving must use **FastAPI + Uvicorn** (`src/api.py` + `src/static/index.html`) alongside the headless CLI (`src/cli.py`).
- **Disk Persistence of Uploaded Images:** Never save user-uploaded fundus images or generated patient PDFs to local disk (`tmp/`, `uploads/`, etc.) during API inference. Decode and process all uploads strictly in RAM via `io.BytesIO`.
- **Retraining on Startup or Upload:** Neither `src/api.py` nor `src/cli.py` may ever import or invoke `src/train.py`.

---

## 2. Model & Training Anti-Patterns

- **Alternative Backbones (ResNet-50, ResNet-34, ViT, Swin, ConvNeXt, MobileNet, EfficientNet):** Rejected. The architecture is locked to `torchvision.models.resnet18(weights=IMAGENET1K_V1)`.
- **Over-Parameterized MLP Classification Heads:** Do not copy the `Dense(512) -> BN -> Dropout(0.5) -> Dense(128) -> BN -> Dropout(0.3) -> Dense(1)` head from `Daramanohar/Glaucoma-detection`. On $N = 400$ training images, multi-layer dense heads rapidly overfit. Use `nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 1))`.
- **Single-Phase Full-Network Unfreezing from Epoch 0:** Unfreezing all convolutional layers at epoch 0 with a random linear head corrupts pretrained filters via large initial gradients. Always execute Phase A (`fc`-only warmup) before Phase B (`layer4` + `fc` fine-tuning).
- **Focal Loss:** Rejected for this prototype. Focal Loss ($\gamma = 2$) severely distorts logit magnitudes and destroys rank-calibration behavior. Use `BCEWithLogitsLoss(pos_weight=n_neg/n_pos)` computed on the training split.
- **Knowledge Distillation, Dual-Branch Segmentation, or IQA Neural Networks:** Rejected as premature complexity creep that introduces cascading failure modes.

---

## 3. Data & Evaluation Anti-Patterns

- **Multi-Dataset Merging (PAPILA + REFUGE + DRISHTI-GS1 + RIM-ONE + G1020):** Rejected. Merging datasets with heterogeneous labeling criteria (especially binary vs. 3-class "Suspect" definitions) introduces uncontrolled domain-label confounding. Use **REFUGE** exclusively.
- **Mapping "Suspect" or "Ungradable" into Glaucoma:** Rejected. Exclude `ungradable` and `suspect` samples during Phase 2 validation and disclose the resulting usable $N$.
- **Claiming Patient-Level Split Isolation on Public REFUGE:** REFUGE's public release does not provide `patient_id`. Do not write fallback logic pretending patient IDs exist; use the official $400 / 400 / 400$ split and explicitly disclose the inter-eye contamination limitation.
- **Train/Serve Resolution Skew ($224 \times 224$ vs. $256 \times 256$):** The reference repository trained at $224 \times 224$ and inferred at $256 \times 256$. All scripts must read `input.size: 224` from `configs/model_config.yaml`.
- **Hardcoded $0.50$ or $0.85$ Decision Thresholds:** Never hardcode classification cutoffs in `inference.py` or `api.py`. Always read the frozen Youden's $J$ validation threshold from `artifacts/models/resnet18_refuge.meta.json`.

---

## 4. Statistical Framing & Scope Creep Anti-Patterns

- **"Confidence" or "Probability" Terminology:** Because `pos_weight` shifts the log-odds prior by $+\ln(w_{\text{pos}})$, raw sigmoid outputs are **not** calibrated probabilities. The words `"confidence"` and `"probability"` are banned in JSON keys, UI text, CLI logs, and PDF reports. Use **"Glaucoma Suspicion Score"**.
- **LLM / RAG / Groq Integration:** Do not create `groq_interface.py`, `rag_data/`, or LLM chat wrappers (as seen in reference repositories). This system is a deterministic computer vision screening pipeline.
- **Modifying `.agents/rules/`:** Project documentation belongs strictly in `docs/`. Do not write project rules into `.agents/rules/`.