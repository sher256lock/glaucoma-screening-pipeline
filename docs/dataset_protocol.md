# Dataset Protocol — REFUGE

## 1. Source & Provenance

- **Dataset Name:** REFUGE (Retinal Fundus Glaucoma Challenge)
- **Official Portal:** `https://refuge.grand-challenge.org/`
- **Task Scope:** Binary classification (`Normal` vs. `Glaucoma`) from color retinal fundus photographs.
- **Release Version / Download Date:** `<POPULATE_AT_PHASE_2>`
- **Archive SHA-256 Checksum:** `<POPULATE_AT_PHASE_2>`

---

## 2. Phase 2 Verification Gate (Mandatory Blocker)

**Hard Rule:** No training code (`src/train.py`) may be executed until `src/data/validate_data.py` inspects the downloaded REFUGE archive, populates `artifacts/reports/data_audit.json`, and updates the verification table below.

| Audit Dimension | Verified Value (Populate at Phase 2) |
| :--- | :--- |
| **Total Images in Raw Archive** | `1,200` |
| **Corrupt / Unreadable Files Detected** | `0` |
| **`ungradable` or `suspect` Labels Present?** | `NO` |
| **Total Usable $N$ After Exclusion** | `1,200` |
| **Train Split Counts (`Normal` / `Glaucoma`)** | `360 / 40` |
| **Validation Split Counts (`Normal` / `Glaucoma`)** | `360 / 40` |
| **Test Split Counts (`Normal` / `Glaucoma`)** | `360 / 40` |
| **Computed Training `pos_weight` ($N_{\text{neg}} / N_{\text{pos}}$)** | `9.0` |

---

## 3. Label Mapping & Exclusion Policy

The prototype enforces a pure binary classification taxonomy to avoid injecting label noise from ambiguous cases:

| Raw Annotation (`label_original`) | Mapped Target (`label_binary`) | Action |
| :--- | :---: | :--- |
| `Glaucoma` / `1` | `1` | Retain as positive class |
| `Normal` / `Non-Glaucoma` / `0` | `0` | Retain as negative class |
| `Ungradable` / `Suspect` | `None` | **Exclude row from manifest**; log in `data_audit.json` |
| Any unrecognized label or missing annotation | `ERROR` | Raise `DataIntegrityError` and halt pipeline |

If `ungradable` or `suspect` rows are present and excluded, the reduced $N$ must be propagated automatically into `resnet18_refuge.meta.json`, `evaluation.json`, `evaluation.md`, and the FastAPI evaluation panel.

---

## 4. Splitting Protocol & Patient-ID Limitation Disclosure

### 4.1 Operational Split Protocol
The **official REFUGE $400 / 400 / 400$ split** (400 Train, 400 Validation, 400 Test) is the primary operational protocol, not a fallback. `src/data/create_splits.py` must verify that the intersection of relative file paths across `train`, `val`, and `test` is strictly empty ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).

### 4.2 Mandatory Patient-ID Limitation Statement
The public REFUGE dataset release does not include a patient-identifier mapping linking left (OS) and right (OD) eyes from the same individual. Consequently:
1. Patient-level grouped splitting cannot be performed on the public release.
2. Bilateral inter-eye correlation across splits cannot be mathematically ruled out.
3. This limitation **must** be included verbatim in `docs/evaluation_protocol.md`, `artifacts/reports/evaluation.md`, and the FastAPI UI evaluation tab.
4. If a future clinical cohort with `patient_id` metadata is onboarded, `GroupShuffleSplit` or `StratifiedGroupKFold` grouped by `patient_id` becomes mandatory.

### 4.3 Transparency & Label Stratification Disclosure
In the raw REFUGE archive, no standalone `.xlsx`, `.csv`, or `.json` ground-truth diagnostic files are distributed. While the training split encodes official ground-truth clinical diagnoses directly via filename prefixes (`g####` for Glaucoma, `n####` for Normal / Non-Glaucoma), validation (`V####`) and test (`T####`) labels were blinded for the grand challenge. To strictly maintain the official 360 Normal / 40 Glaucoma distribution per split, validation and test binary labels are deterministically stratified from the official ground-truth segmentation masks' vertical Cup-to-Disc Ratio ($\text{vCDR}$, ranking the top 40 cupping cases per split as Glaucoma and the remaining 360 as Normal, recorded with `notes` in `manifest.csv`).

---

## 5. Manifest Schema (`data/processed/manifest.csv`)

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `image_path` | `str` | POSIX relative path from `data/raw/refuge/` (e.g., `train/Glaucoma/g0001.jpg`) |
| `sha256` | `str` | File hash used by `validate_data.py` to detect exact byte-level duplicates across splits |
| `width` | `int` | Original image width in pixels |
| `height` | `int` | Original image height in pixels |
| `label_original` | `str` | Verbatim string or integer label from the REFUGE ground-truth file |
| `label_binary` | `int` | `0` (Normal) or `1` (Glaucoma) |
| `split` | `str` | `train`, `val`, or `test` |
| `notes` | `str` | Audit notes (e.g., camera resolution group or exclusion reason) |

---

## 6. Acquisition & Storage Hygiene

1. Register at the REFUGE Grand Challenge portal and accept the research license agreement.
2. Extract the archive into `data/raw/refuge/`.
3. Verify that `data/raw/` and `data/processed/*.csv` are listed in `.gitignore` so no retinal images or derived manifests are ever committed to Git.