# Data Directory

This directory should contain the raw and processed datasets.

## Structure

```
data/
├── raw/
│   ├── sleep-edf/       # Raw Sleep-EDF files (.rec, .hyp)
│   └── wesad/           # Raw WESAD files (.pkl)
└── processed/
    ├── sleep-edf/       # Processed numpy arrays
    └── wesad/           # Processed numpy arrays
```

## Download Instructions

### Sleep-EDF Dataset

**Source**: [PhysioNet Sleep-EDF Database](https://physionet.org/content/sleep-edfx/1.0.0/)

**File Types**:
- `.rec` files: Recording data (EEG, EOG, EMG signals)
- `.hyp` files: Hypnogram annotations (sleep stages)
- `RECORDS`: List of all recording files

**Download via wget**:
```bash
cd data/raw
wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/
```

**Or manually**: Download from the website and extract to `data/raw/sleep-edf/`

**Expected files in `data/raw/sleep-edf/`**:
```
sc4001e0.rec    # Recording file
sc4001e0.hyp    # Hypnogram file
sc4002e0.rec    # Recording file
sc4002e0.hyp    # Hypnogram file
...
RECORDS         # List of all files
```

### WESAD Dataset

**Source**: [UCI ML Repository - WESAD](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29)

**Download**: Manual (requires registration)

1. Register and download from UCI
2. Extract only the `.pkl` files to `data/raw/wesad/`

**File Structure**:
- Each subject has a folder (S2, S3, S4, ..., S17) - **S1 and S12 are missing**
- Each folder contains: `SX.pkl` (synchronized data and labels)
- **For simple usage**: Only need the `.pkl` files

**Expected files in `data/raw/wesad/`**:
```
S2.pkl    # Subject 2 data
S3.pkl    # Subject 3 data
S4.pkl    # Subject 4 data
...
S17.pkl   # Subject 17 data
```

**Note**: The `.pkl` files contain:
- `'subject'`: Subject ID
- `'signal'`: Raw data from chest (RespiBAN) and wrist (Empatica E4)
- `'label'`: Study protocol conditions (0=transient, 1=baseline, 2=stress, 3=amusement, 4=meditation)

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

