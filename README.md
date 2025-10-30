# Privacy-Preserving mHealth Data Analysis

This repository studies the privacy–utility trade-off in mobile health (mHealth) using Differential Privacy (DP) and Federated Learning (FL) on two real-world datasets: Sleep-EDF and WESAD. The work is developed in the context of the SIDM course (Information Systems for Mobile Devices).

## Overview

We implement and compare the following training settings:

1. Baseline: standard supervised training (no privacy)
2. Differential Privacy (DP): per-sample gradient clipping and noise
3. Federated Learning (FL): simulated multi-client training
4. DP + FL: differentially private federated training

## Project Structure

```
mhealth-data-privacy/
├── src/
│   ├── configs/                 # YAML configs for datasets and methods
│   ├── preprocessing/           # Dataset preprocessing pipelines
│   ├── models/                  # Model architectures (PyTorch)
│   ├── training/                # Trainers, schedulers, utils
│   ├── privacy/                 # DP and FL utilities
│   └── evaluation/              # Metrics and reporting
├── experiments/
│   ├── scenarios/               # Experiment scenario definitions
│   └── run_experiments.py       # Unified experiment runner
├── data/                        # Raw and processed data (not versioned)
├── results/                     # Training outputs and logs (not versioned)
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

```bash
git clone https://github.com/vasco-fernandes21/mhealth-data-privacy.git
cd mhealth-data-privacy

python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

pip install -U pip
pip install -e .
pip install -r requirements.txt
```

## Data Preprocessing

Preprocessed data is stored under `data/processed/`.

- WESAD (default: 32 Hz, window 1024 samples ≈ 32 s, 50% overlap):

```bash
python src/preprocessing/wesad.py \
  --data_dir data/raw/wesad \
  --output_dir data/processed/wesad \
  --target_freq 32 \
  --window_size 1024
```

- Sleep-EDF (already prepared as windows of length 10 with 24 features):

```bash
python src/preprocessing/sleep_edf.py \
  --data_dir data/raw/sleep-edf \
  --output_dir data/processed/sleep-edf
```

## Configuration

YAML files in `src/configs` define dataset- and method-specific parameters. The current baseline configurations are as follows.

- WESAD (`src/configs/datasets/wesad.yaml`):
  - Model: LSTM, bidirectional, input projection 128, `lstm_units=56`, `dropout=0.48`, `dense_layers=[112, 56]`.
  - Training: `batch_size=12`, `learning_rate=0.0008`, `optimizer=AdamW`, `label_smoothing=0.13`, warmup cosine scheduler (`warmup_epochs=3`), early stopping patience 18, gradient clipping.
  - Sequence length in config is for model reference; the actual sequence length is determined by preprocessing (currently 1024).

- Sleep-EDF (`src/configs/datasets/sleep-edf.yaml`):
  - 5-class classification with 24 features and sequence length 10 (already windowed in preprocessing).
  - Baseline uses the unified LSTM model with appropriate input dimensions set at runtime.

Method configs live in `src/configs/methods/` and can override training hyperparameters.

## Running Experiments

Experiments are driven by scenarios in `experiments/scenarios/*.yaml`. The runner loads the scenario, merges dataset and method configs, applies optional overrides, and executes training with consistent logging.

Examples:

```bash
# Baseline on WESAD
python experiments/run_experiments.py --scenario baseline --datasets wesad --auto

# Baseline on Sleep-EDF
python experiments/run_experiments.py --scenario baseline --datasets sleep-edf --auto

# Run all scenarios (baseline, dp, fl, dp_fl)
python experiments/run_experiments.py --scenario all --auto
```

Key flags:

- `--tags tier1` filter by tags defined in the scenario file
- `--methods baseline,dp` filter by method(s)
- `--epsilon 5.0` filter DP experiments by target epsilon
- `--clients 5` filter FL experiments by client count
- `--n_experiments N` limit the number of runs

## Results and Logging

- Per-run results are saved under `results/{method}/{dataset}/seed_{seed}/results.json`.
- A summary of all runs in a session is saved to `experiments/results_log.json`.
- The baseline trainer reports accuracy, precision, recall, F1 (weighted), per-class metrics, and the confusion matrix.

## Implementation Notes

- Framework: PyTorch
- Best-model evaluation: the runner enables checkpointing per run to restore best validation weights before test evaluation.

## Acknowledgements

Developed as part of the SIDM course (Information Systems for Mobile Devices). This repository focuses on rigorous, reproducible comparisons of privacy-preserving learning techniques in mHealth.


## License

This project is for academic research purposes.

