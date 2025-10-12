# Data Directory

This directory should contain the raw and processed datasets.

## Structure

```
data/
├── raw/
│   ├── sleep-edf/       # Raw Sleep-EDF files (.edf)
│   └── wesad/           # Raw WESAD files (.pkl)
└── processed/
    ├── sleep-edf/       # Processed numpy arrays
    └── wesad/           # Processed numpy arrays
```

## Download Instructions

### Sleep-EDF Dataset

**Source**: [PhysioNet Sleep-EDF Database](https://physionet.org/content/sleep-edfx/1.0.0/)

**Download via wget**:
```bash
cd data/raw
wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/
```

**Or manually**: Download from the website and extract to `data/raw/sleep-edf/`

### WESAD Dataset

**Source**: [UCI ML Repository - WESAD](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29)

**Download**: Manual (requires registration)

1. Register and download from UCI
2. Extract `.pkl` files to `data/raw/wesad/`

## Google Drive Setup (for Colab)

If using Google Colab, create this structure in your Google Drive:

```
MyDrive/
└── mhealth-data/
    ├── raw/
    │   ├── sleep-edf/
    │   └── wesad/
    └── processed/
        ├── sleep-edf/
        └── wesad/
```

Then mount Drive in Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Processed Data Format

After running preprocessing notebooks, you'll have:

**Sleep-EDF**:
- `X_train.npy`, `X_val.npy`, `X_test.npy` (features)
- `y_train.npy`, `y_val.npy`, `y_test.npy` (labels)
- `scaler.pkl` (StandardScaler fitted on training data)

**WESAD**:
- Same structure as Sleep-EDF

## Notes

- Raw data files are **NOT** tracked by git (see `.gitignore`)
- Processed files are saved in Google Drive for persistence
- Download raw data only once, preprocessing is also done once

