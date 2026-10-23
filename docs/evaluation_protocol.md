# Statistical Evaluation & Explainability Protocol

## 1. Evaluation Governance Rules

1. **Zero Test-Set Tuning:** The held-out REFUGE test split ($N = 400$) is evaluated strictly after Phase A and Phase B training are complete and the decision threshold $t^*$ has been frozen on the validation split.
2. **Single-Command Reproducibility:** Running `python -m src.evaluate --config configs/model_config.yaml` must deterministically regenerate `artifacts/reports/evaluation.json`, `artifacts/reports/evaluation.md`, and `artifacts/reports/gradcam_grid.png`.

---

## 2. Metrics & Mathematical Definitions

Let $\hat{s}_i = \sigma(z_i) \in [0, 1]$ be the model's Glaucoma Suspicion Score for test sample $i \in \{1, \dots, N_{\text{test}}\}$, $y_i \in \{0, 1\}$ be the ground-truth label, and $t^*$ be the frozen validation threshold such that $\hat{y}_i = \mathbb{I}(\hat{s}_i \ge t^*)$.

| Metric | Mathematical Formula | Reporting Format |
| :--- | :--- | :--- |
| **AUROC** | $\int_{0}^{1} \text{TPR}(\text{FPR}^{-1}(u)) \, du$ | Point estimate + $95\%$ Bootstrap CI |
| **Sensitivity (Recall)** | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | Point estimate + $95\%$ Bootstrap CI |
| **Specificity** | $\frac{\text{TN}}{\text{TN} + \text{FP}}$ | Point estimate + $95\%$ Bootstrap CI |
| **F1 Score** | $\frac{2 \, \text{TP}}{2 \, \text{TP} + \text{FP} + \text{FN}}$ | Point estimate + $95\%$ Bootstrap CI |
| **Brier Score (Diagnostic)** | $\frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} (\hat{s}_i - y_i)^2$ | Point estimate only (flagged as `diagnostic_only: true` due to `pos_weight` prior shift) |
| **Confusion Matrix** | $\begin{bmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{bmatrix}$ | Raw integer counts **and** row-normalized percentages |

---

## 3. Threshold Freezing & Metric Floor Protocol

1. Load the best Phase B checkpoint (`resnet18_refuge.pth`) and run inference over the **validation split** ($N_{\text{val}} = 400$).
2. Compute the ROC curve `(fpr, tpr, thresholds) = sklearn.metrics.roc_curve(y_val, s_val)`.
3. Calculate Youden's $J$ statistic across all candidate thresholds:
   $$J(t) = \text{tpr}(t) + (1 - \text{fpr}(t)) - 1$$
4. Select $t^* = \text{thresholds}[\arg\max J(t)]$.
5. **Check Phase 4 Metric Floor on Validation Split:**
   - **Condition 1:** $\text{AUROC}_{\text{val}} > 0.85$. (If false, halt evaluation and return to Phase 3 training).
   - **Condition 2:** $\text{Sensitivity}_{\text{val}}(t^*) > 0.80$.
   - **Adjustment Clause:** If Condition 1 passes ($\text{AUROC}_{\text{val}} \ge 0.85$) but $\text{Sensitivity}_{\text{val}}(t^*) < 0.80$, select the largest threshold $t_{\text{adj}}$ on the validation ROC curve satisfying $\text{Sensitivity}_{\text{val}}(t_{\text{adj}}) \ge 0.80$. Record `"threshold": t_adj`, `"threshold_procedure": "sensitivity_priority_80_on_val"`, and the validation specificity tradeoff in `resnet18_refuge.meta.json`.
6. Serialize the chosen threshold into `artifacts/models/resnet18_refuge.meta.json`.
7. Evaluate the **test split** once using the serialized threshold.

---

## 4. Non-Parametric Bootstrap Confidence Intervals ($95\%$ CI)

Because $N_{\text{test}} = 400$ with approximately $10\%$ glaucoma prevalence ($40$ positive cases), point estimates carry sampling variance that must be disclosed:
- **Resampling Procedure:** Draw $B = 1,000$ bootstrap samples of size $N_{\text{test}}$ with replacement using a fixed random number generator `np.random.default_rng(seed=42)`.
- **Degenerate Resample Guard:** Skip any resample that contains only one class (though with $N=400$ and $p=0.10$, the probability of an all-negative resample is $(0.9)^{400} \approx 5.3 \times 10^{-19}$).
- **Interval Calculation:** Compute the $2.5\text{th}$ and $97.5\text{th}$ percentiles across the $1,000$ bootstrap replicates for AUROC, Sensitivity, Specificity, and F1.

---

## 5. Outcome-Stratified Grad-CAM Error Analysis

To audit whether the ResNet-18 model attends to clinically relevant anatomical structures (the optic nerve head and neuroretinal rim) rather than illumination artifacts or image borders, `src/evaluate.py` must generate `artifacts/reports/gradcam_grid.png`:

1. **Hook Target:** Attach a forward hook and a full backward hook (`register_full_backward_hook`) to `model.layer4[-1]`.
2. **Spatial Formulation:** Let $A^k \in \mathbb{R}^{7 \times 7}$ be the $k$-th feature map ($k \in \{1, \dots, 512\}$) of `layer4[-1]`, and let $\frac{\partial z}{\partial A^k_{ij}}$ be the gradient of the raw binary logit $z$ with respect to $A^k_{ij}$. Compute the channel importance weights:
   $$\alpha_k = \frac{1}{49} \sum_{i=1}^{7} \sum_{j=1}^{7} \frac{\partial z}{\partial A^k_{ij}}$$
   and the raw localization map:
   $$L_{\text{Grad-CAM}} = \text{ReLU}\left( \sum_{k=1}^{512} \alpha_k A^k \right) \in \mathbb{R}^{7 \times 7}$$
3. **Spatial Alignment:** Upsample $L_{\text{Grad-CAM}}$ from $7 \times 7$ to $224 \times 224$ using bilinear interpolation (`align_corners=False`), normalize to $[0, 1]$, and blend ($60\%$ unnormalized RGB fundus image + $40\%$ `cv2.COLORMAP_JET`).
4. **4x4 Outcome Grid (`gradcam_grid.png`):** Partition test-set predictions into **True Positives (TP)**, **False Positives (FP)**, **True Negatives (TN)**, and **False Negatives (FN)**. Render up to $4$ highest-priority examples per category ($16$ panels total), annotating each panel with its relative ID, true label, predicted class, and Suspicion Score $\hat{s}_i$.

---

## 6. Local CPU Latency Benchmarking Protocol

Latency claims must never be estimated or extrapolated from cloud vCPUs. `src/evaluate.py` measures local CPU inference latency using the following protocol:
- **Runtime Settings:** `device = torch.device("cpu")`, `model.eval()`, `torch.no_grad()`, `batch_size = 1`, `torch.set_num_threads(physical_cores)`.
- **Warmup:** Execute $10$ untimed warmup passes on a held-out $224 \times 224$ test image to warm CPU instruction and L2/L3 caches.
- **Measurement:** Execute $100$ consecutive timed passes (`time.perf_counter()`) covering preprocessing + forward pass + sigmoid scoring.
- **Reporting:** Record $P_{50}$ (median) and $P_{95}$ latency in milliseconds alongside the exact CPU model string (`platform.processor()`) and physical thread count in `evaluation.json` and `evaluation.md`.