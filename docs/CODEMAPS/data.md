<!-- Generated: 2026-10-24 | Files scanned: 5 | Token estimate: ~550 -->
# Data Pipeline & Preprocessing Codemap

## REFUGE Dataset Organization

The Retinal Fundus Glaucoma Challenge (REFUGE) dataset is partitioned into official, leak-proof splits:

| Split | Images ($N$) | Glaucoma ($y=1$) | Normal ($y=0$) | Primary Camera Modality | Role |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Train** | $400$ | $40$ ($10.0\%$) | $360$ ($90.0\%$) | Zeiss Visucam 500 ($2124 \times 2056$) | Model training (Class-weighted BCE) |
| **Val** | $400$ | $40$ ($10.0\%$) | $360$ ($90.0\%$) | Canon CR-2 ($1634 \times 1634$) | Threshold selection (Youden's $J$) & Early stopping |
| **Test** | $400$ | $40$ ($10.0\%$) | $360$ ($90.0\%$) | Canon CR-2 ($1634 \times 1634$) | Final held-out evaluation (evaluated once) |

## Data Modules & Manifest Generation (`src/data/`)

- `src/data/load_refuge.py`:
  - Discovers images across `data/raw/refuge/{train,val,test}`.
  - Resolves ground truth labels (`0` = Normal, `1` = Glaucoma) from REFUGE Excel/CSV tables.
  - Emits canonical manifest: `data/processed/manifest.csv`.
- `src/data/validate_data.py`:
  - Verifies image readability, dimensions, and filters ungradable scans.
  - Outputs audit summary: `artifacts/reports/data_audit.json`.
- `src/data/create_splits.py`:
  - Enforces zero patient/image leakage across partitions.
  - Emits verified split files: `data/processed/splits/{train,val,test}.csv`.
- `src/data/dataset.py`:
  - `RefugeDataset(torch.utils.data.Dataset)`: PyTorch dataset implementing unified preprocessing.

## Preprocessing Pipeline (`src/preprocessing.py`)

All scripts (`train.py`, `evaluate.py`, `inference.py`, `api.py`, `cli.py`) share the exact same deterministic preprocessing logic:

```
Raw BGR/RGB Image
       │
       ▼
[ Optional LAB CLAHE ]
  - Converts RGB -> LAB color space
  - Applies cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) to L-channel
  - Converts LAB -> RGB (enhances optic cup/disc neuroretinal rim contrast)
       │
       ▼
[ Spatial Resize ]
  - Bilinear resize to 224 x 224 pixels
       │
       ▼
[ Tensor Conversion & Normalization ]
  - Scale pixel intensities: [0, 255] -> [0.0, 1.0]
  - ImageNet Standard Normalization:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
```
