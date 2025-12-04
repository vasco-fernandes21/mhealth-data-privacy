# PrivacyHealth MVP

Federated Learning + Differential Privacy Simulator for mHealth Data

## Structure

- `server/` - FastAPI backend with PyTorch + Opacus
- `client/` - React + Vite PWA frontend

## Quick Start

### Backend

```bash
cd app/server
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.main:app --reload
```

Backend runs on `http://localhost:8000`

### Frontend

```bash
cd app/client
npm install
npm run dev
```

Frontend runs on `http://localhost:3000`

## Data

The server expects data in `data/processed/wesad/` and `data/processed/sleep-edf/` directories with the following structure:

- `X_train.npy`, `X_val.npy`, `X_test.npy` - Features
- `y_train.npy`, `y_val.npy`, `y_test.npy` - Labels
- `subjects_train.npy`, `subjects_val.npy`, `subjects_test.npy` - Subject IDs

The data path can be configured via `DATA_DIR` environment variable or defaults to `../../data/processed/` relative to the server directory.

