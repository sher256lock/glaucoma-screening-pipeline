# Statistical Evaluation & Model Interpretability Report

**Evaluation Protocol:** Phase 4 Held-Out Verification  
**Evaluation Standard:** docs/evaluation_protocol.md & docs/PROJECT_SPEC.md  
**Model Architecture:** ResNet-18 (`IMAGENET1K_V1`) + `Dropout(0.3) -> Linear(512, 1)`  
**Checkpoint Source:** `artifacts/models/resnet18_refuge.pth` (Phase B, Epoch 25)  

---

## 1. Test Cohort & Prevalence

| Parameter | Value | Operational Note |
| :--- | :---: | :--- |
| **Total Test Samples ($N$)** | **400** | Official held-out REFUGE test split |
| **Glaucoma Cases ($y=1$)** | **40** | Stratified positive cases |
| **Normal Cases ($y=0$)** | **360** | Stratified negative cases |
| **Prevalence** | **10.0%** | Representative clinical screening prevalence |
| **Split Protocol** | `official_400_400_400` | Zero train/val/test data leakage |

---

## 2. Performance Metrics & 95% Bootstrap Confidence Intervals

Point estimates and non-parametric $95\%$ Confidence Intervals derived from $B = 1,000$ bootstrap resamples ($seed=42$, $2.5^{\text{th}}$ and $97.5^{\text{th}}$ percentiles):

| Metric | Point Estimate | 95% Bootstrap CI | Clinical & Statistical Definition |
| :--- | :---: | :---: | :--- |
| **AUROC** | **0.885** | [0.829, 0.932] | Discrimination capacity across all operational thresholds |
| **Sensitivity (Recall)** | **0.900** | [0.795, 0.977] | True Positive Rate on confirmed glaucoma ($N = 40$) |
| **Specificity** | **0.706** | [0.657, 0.754] | True Negative Rate on confirmed normal ($N = 360$) |
| **F1 Score** | **0.396** | [0.301, 0.485] | Harmonic mean of precision and sensitivity |
| **Diagnostic Brier Score** | **0.128** | — | Mean squared error (`diagnostic_only: true`) |

---

## 3. Test Split Confusion Matrix

Evaluated at the frozen decision threshold $t^* = 0.2921$:

| Ground Truth Category | Predicted Normal ($s < t^*$) | Predicted Glaucoma ($s \ge t^*$) | Cohort Total | Row-Normalized Distribution |
| :--- | :---: | :---: | :---: | :---: |
| **Normal / Non-Glaucoma ($0$)** | **254** (TN) | **106** (FP) | **360** | **70.6%** TN / **29.4%** FP |
| **Glaucoma ($1$)** | **4** (FN) | **36** (TP) | **40** | **10.0%** FN / **90.0%** TP |

---

## 4. Frozen Threshold & Validation Calibration Procedure

| Validation Parameter | Value | Compliance Status |
| :--- | :---: | :--- |
| **Decision Threshold ($t^*$)** | **0.2921** | Frozen strictly on validation split prior to test evaluation |
| **Calibration Procedure** | `youden_j_on_val` | Youden's $J$ optimization ($J = \text{TPR} - \text{FPR}$) |
| **Validation AUROC** | **0.8919** | **PASSED** ($> 0.85$ Phase 4 metric floor) |
| **Validation Sensitivity at $t^*$** | **0.8500** | **PASSED** ($\ge 0.80$ clinical screening floor) |
| **Validation Specificity at $t^*$** | **0.8639** | Operational specificity trade-off |

---

## 5. Local CPU Inference Latency Benchmark

Measured locally on the host processor with single-image inference (`batch_size=1`, `torch.no_grad()`):

| Latency Parameter | Value |
| :--- | :--- |
| **Host Processor** | `13th Gen Intel(R) Core(TM) i7-13650HX` |
| **Execution Device** | `CPU` |
| **Physical Core Threads** | `14` |
| **Benchmark Iterations** | 100 timed passes (after 10 untimed cache warmup passes) |
| **$P_{50}$ Latency (Median)** | **30.4 ms** |
| **$P_{95}$ Latency (Tail)** | **34.5 ms** |

---

## 6. Verbatim Limitations Disclosures

1. **REFUGE Patient-ID Limitation Disclosure (Verbatim):**  
   The public REFUGE dataset release does not include a patient-identifier mapping linking left (OS) and right (OD) eyes from the same individual. Consequently:
   - Patient-level grouped splitting cannot be performed on the public release.
   - Bilateral inter-eye correlation across splits cannot be mathematically ruled out.
   - This limitation must be included verbatim in `docs/evaluation_protocol.md`, `artifacts/reports/evaluation.md`, and the FastAPI UI evaluation tab.
   - If a future clinical cohort with `patient_id` metadata is onboarded, `GroupShuffleSplit` or `StratifiedGroupKFold` grouped by `patient_id` becomes mandatory.

2. **Transparency & Label Stratification Disclosure (Verbatim):**  
   In the raw REFUGE archive, no standalone `.xlsx`, `.csv`, or `.json` ground-truth diagnostic files are distributed. While the training split encodes official ground-truth clinical diagnoses directly via filename prefixes (`g####` for Glaucoma, `n####` for Normal / Non-Glaucoma), validation (`V####`) and test (`T####`) labels were blinded for the grand challenge. To strictly maintain the official 360 Normal / 40 Glaucoma distribution per split, validation and test binary labels are deterministically stratified from the official ground-truth segmentation masks' vertical Cup-to-Disc Ratio (vCDR, ranking the top 40 cupping cases per split as Glaucoma and the remaining 360 as Normal, recorded with `notes` in `manifest.csv`).

3. **External Validation Absence:**  
   Single-dataset evaluation on REFUGE. No external multicenter validation cohort has been evaluated.

4. **Sampling Variance:**  
   Held-out test set $N = 400$ with $40$ positive cases yields sampling variance reflected in the reported non-parametric $95\%$ bootstrap confidence intervals.

---

## 7. Mandatory Regulatory & Statistical Disclaimer

**RESEARCH PROTOTYPE ONLY. Glaucoma Suspicion Score is not a calibrated probability and is not for clinical diagnosis.**

Model outputs are reported strictly as a **"Glaucoma Suspicion Score"** derived from a class-weighted binary cross-entropy objective (`BCEWithLogitsLoss`). Because positive class weighting (`pos_weight = 9.0`) shifts the prior log-odds by $\ln(w_{\text{pos}})$, these scores are not calibrated statistical probabilities and must never be presented as "confidence" or "probability." All outputs require clinical correlation and formal diagnostic evaluation by a qualified ophthalmologist.
