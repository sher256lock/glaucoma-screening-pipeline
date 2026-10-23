# ECC & Antigravity Orchestration Playbook

This playbook maps the project lifecycle directly to the autonomous agents, specialized on-demand skills, and interactive slash-command workflows cataloged in `AGENTS_SKILLS_WORKFLOWS.md` (`.agents/`).

> **Governance Rule:** Do **not** create or modify files inside `.agents/rules/`. All project-specific constraints are governed by `docs/PROJECT_SPEC.md` and `docs/rejected_patterns.md`.

---

## 1. CLI Initialization & Skill Resolution

Run the following commands in the project root to verify the Everything Claude Code (ECC) v2.2.3 installation for the Antigravity adapter:

```powershell
cd <glaucoma-project-root>
npm view ecc-universal version
npx ecc-universal@2.2.3 install --profile minimal --target antigravity
npx ecc-universal@2.2.3 consult "PyTorch ResNet-18 image classification FastAPI Python testing healthcare compliance CPU inference" --target antigravity
```

Verify in **Antigravity -> Settings -> Customizations** that the `.agents/` catalog is active, then start a **new conversation** before running Phase 0.

---

## 2. Phase-by-Phase Routing Matrix (Agents, Skills & Workflows)

Every phase uses the exact identifiers installed in `.agents/agents/`, `.agents/skills/`, and `.agents/workflows/`:

| Phase | Milestone | Autonomous Agents (`.agents/agents/`) | Specialized Skills (`.agents/skills/`) | Interactive Workflows (`.agents/workflows/`) |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 0 & 1** | Environment, Config & Scaffold | `planner`, `code-architect`, `architect` | `intent-driven-development`, `architecture-decision-records`, `python-patterns`, `coding-standards` | `/plan`, `/orch-build-mvp` |
| **Phase 2** | REFUGE Manifest & Split Pipeline | `python-reviewer`, `silent-failure-hunter`, `tdd-guide` | `python-patterns`, `error-handling`, `python-testing`, `tdd-workflow` | `/feature-dev`, `/python-review` |
| **Phase 3a & 3b** | RTX 4060 Two-Phase ResNet-18 Training | `pytorch-build-resolver`, `mle-reviewer`, `performance-optimizer` | `pytorch-patterns`, `mle-workflow`, `python-patterns` | `/build-fix`, `/quality-gate` |
| **Phase 4** | Evaluation, Bootstrap CIs & Grad-CAM Grid | `mle-reviewer`, `python-reviewer`, `tdd-guide` | `eval-harness`, `python-testing`, `pytorch-patterns`, `mle-workflow` | `/test-coverage`, `/python-review` |
| **Phase 5** | CPU Inference, CLI & FastAPI + Local HTML UI | `fastapi-reviewer`, `python-reviewer`, `silent-failure-hunter`, `a11y-architect` | `fastapi-patterns`, `api-design`, `contract-first`, `error-handling`, `accessibility` | `/feature-dev`, `/fastapi-review`, `/python-review` |
| **Phase 6** | ReportLab PDF, PHI Safety & Final Gate | `healthcare-reviewer`, `security-reviewer`, `code-reviewer` | `healthcare-phi-compliance`, `healthcare-eval-harness`, `security-review`, `verification-loop` | `/security-scan`, `/orch-review`, `/checkpoint` |

---

## 3. Copy-Paste Antigravity Kickoff Prompts

Paste these prompts sequentially into Antigravity. Wait for each phase gate to pass and review actual pytest/CLI outputs before advancing to the next prompt.

### Prompt 1 — Master Onboarding & Phase 0/1 (Scaffold & Config)
```text
Act as a Principal Machine Learning Engineer. First, read `docs/PROJECT_SPEC.md`, `docs/architecture.md`, `docs/dataset_protocol.md`, `docs/evaluation_protocol.md`, `docs/rejected_patterns.md`, and `configs/model_config.yaml`. Do NOT modify any files inside `.agents/rules/`.

Invoke `/plan` using the `planner` and `code-architect` agents with `intent-driven-development` and `python-patterns` skills to execute Phase 0 and Phase 1:
1. Verify Python environment, `requirements.txt` (PyTorch CUDA 12.1, FastAPI, Uvicorn, OpenCV, Scikit-Learn, ReportLab, Pytest — zero Streamlit), and check `torch.cuda.is_available()` for the NVIDIA RTX 4060 GPU.
2. Create the directory skeleton (`src/data/`, `src/static/`, `tests/`, `artifacts/models/`, `artifacts/reports/`).
3. Implement `tests/test_config_schema.py` to validate every section of `configs/model_config.yaml` (ResNet-18 architecture, 224x224 input, ImageNet normalization, two-phase training schedule, Youden's J threshold rule, CPU inference settings, and FastAPI server config).
4. Run `pytest -q` and report actual terminal output before proceeding to Phase 2.
```

### Prompt 2 — Phase 2 (REFUGE Manifest Verification Gate)
```text
Execute Phase 2 per `docs/dataset_protocol.md` using `python-patterns`, `error-handling`, and `python-testing` skills, followed by a review from `python-reviewer` and `silent-failure-hunter`:
1. Implement `src/data/load_refuge.py` to parse the REFUGE directory structure and ground-truth annotations into `data/processed/manifest.csv`.
2. Implement `src/data/validate_data.py` to scan all images for corruption, compute SHA-256 hashes for duplicate detection, exclude any `ungradable` or `suspect` rows per Section 3 of `docs/dataset_protocol.md`, and write `artifacts/reports/data_audit.json`.
3. Implement `src/data/create_splits.py` enforcing the official 400/400/400 split verbatim and verifying zero path or hash overlap across splits.
4. Implement `tests/test_label_mapping.py` (testing binary mapping, ungradable exclusion, and raising `DataIntegrityError` on unknown labels).
5. Run the data validation scripts and `pytest -q`. Populate the Verified Counts table in `docs/dataset_protocol.md` with the actual numbers from `data_audit.json`. Stop and report the verified counts before touching `src/train.py`.
```

### Prompt 3 — Phase 3a & 3b (RTX 4060 Two-Phase Training)
```text
Execute Phase 3a and Phase 3b per `docs/PROJECT_SPEC.md` using `pytorch-patterns` and `mle-workflow` skills, with review by `mle-reviewer` and `pytorch-build-resolver`:
1. Implement `src/preprocessing.py` driven strictly by `configs/model_config.yaml`.
2. Implement `src/data/dataset.py` (`RefugeDataset`) using `src/preprocessing.py` and training-only augmentations (random horizontal flip, random rotation +-15 deg, mild color jitter).
3. Implement `src/model.py` (`build_resnet18(config)` with `Dropout(0.3) -> Linear(512, 1)`, plus `set_phase_a_trainable(model)` and `set_phase_b_trainable(model)`).
4. Implement `src/train.py` configured for the NVIDIA RTX 4060 8GB GPU:
   - Enable `allow_tf32` and `torch.amp.autocast('cuda', dtype=torch.float16)` with `GradScaler`.
   - Compute `pos_weight = n_neg / n_pos` strictly from the training split.
   - Run Phase A (8 epochs, `fc` only, `lr=1e-3`). Enforce the Phase B gate: require `val_auroc > 0.75`.
   - Run Phase B (up to 20 epochs, `layer4` + `fc` unfrozen, `lr=1e-5`, `ReduceLROnPlateau` on `val_loss`, early stopping patience 8 on `val_auroc`).
   - Save best `resnet18_refuge.pth`, initial `resnet18_refuge.meta.json`, and `artifacts/reports/training_curves.png`.
5. Implement `tests/test_preprocessing.py` and `tests/test_checkpoint_metadata.py`, verify checkpoint loading in an isolated subprocess, and report actual validation AUROC.
```

### Prompt 4 — Phase 4 (Evaluation, Threshold Freezing & Grad-CAM Grid)
```text
Execute Phase 4 per `docs/evaluation_protocol.md` using `eval-harness`, `pytorch-patterns`, and `python-testing` skills, reviewed by `mle-reviewer`:
1. Implement `src/explainability.py` with manual forward and full-backward PyTorch hooks on `model.layer4[-1]`, bilinearly upsampling the 7x7 ReLU attribution map to 224x224.
2. Implement `src/evaluate.py`:
   - Compute Youden's J threshold on the validation split and enforce the Metric Floor (`val_auroc > 0.85` and `val_sensitivity > 0.80`, applying the sensitivity-adjustment clause if needed).
   - Freeze the selected threshold into `artifacts/models/resnet18_refuge.meta.json`.
   - Evaluate the held-out test split once at the frozen threshold. Compute AUROC, Sensitivity, Specificity, F1, raw & normalized Confusion Matrix, and diagnostic Brier score.
   - Compute 1,000-iteration bootstrap 95% confidence intervals (`seed=42`).
   - Benchmark local CPU inference latency (P50 and P95 over 100 warm runs, `batch_size=1`, pinned physical threads).
   - Generate the 16-panel TP / FP / TN / FN Grad-CAM outcome grid (`artifacts/reports/gradcam_grid.png`).
   - Emit `artifacts/reports/evaluation.json` and `artifacts/reports/evaluation.md` including the verbatim REFUGE patient-ID limitation disclosure.
3. Implement `tests/test_threshold_loading.py` and run `pytest -q`.
```

### Prompt 5 — Phase 5 & 6 (CPU Inference, CLI, FastAPI + Local HTML UI, PDF & Final Gate)
```text
Execute Phase 5 and Phase 6 per `docs/architecture.md` using `fastapi-patterns`, `api-design`, `error-handling`, `healthcare-phi-compliance`, and `verification-loop` skills. Remember: DO NOT use Streamlit.
1. Implement `src/quality_checks.py` (OpenCV Laplacian blur and luminance checks returning non-blocking warning strings).
2. Implement `src/inference.py` (`GlaucomaPredictor` running on CPU with `eval()`, `torch.no_grad()`, reading `threshold` from `resnet18_refuge.meta.json`, raising structured exceptions on missing/corrupt checkpoints or invalid images, and outputting `"suspicion_score"`—never `"confidence"` or `"probability"`).
3. Implement `src/report.py` to generate an in-memory (`io.BytesIO`) ReportLab PDF report containing the reference ID, timestamp, predicted class, Glaucoma Suspicion Score, threshold used, quality warnings, Grad-CAM overlay, and research-prototype disclaimer.
4. Implement `src/cli.py` for headless CLI inference and `src/api.py` (FastAPI with `lifespan` singleton model loading, `POST /api/v1/predict`, `POST /api/v1/report`, `GET /api/v1/metrics`, `GET /api/v1/health`, and mounting `src/static/index.html`).
5. Implement `src/static/index.html` (clean, accessible HTML5/CSS/JS clinical screening interface displaying image preview, separated quality warnings, Glaucoma Suspicion Score gauge, Grad-CAM overlay toggle, PDF download, and evaluation metrics panel).
6. Implement `tests/test_inference_errors.py` and `tests/test_api_endpoints.py` (including corrupt file upload and `.txt`-renamed-to-`.jpg` rejection tests).
7. Run `/fastapi-review`, `/python-review`, `/security-scan` (with `healthcare-reviewer`), and `/orch-review` to verify all 12 acceptance criteria in `docs/PROJECT_SPEC.md`.
```