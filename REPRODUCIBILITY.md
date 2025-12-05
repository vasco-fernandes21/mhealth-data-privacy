# Reproducibility Guide

This document provides step-by-step instructions to reproduce all experiments and results from the paper.

## Environment Setup

### Python Version
- Python 3.12+ (tested with 3.12.12)

### Dependencies

```bash
pip install -r requirements.txt
```

### Key Package Versions
- PyTorch: 2.2.0
- Opacus: 1.4.0
- NumPy: 1.26.4
- scikit-learn: 1.4.1.post1

## Data Preparation

### WESAD Dataset

1. Download WESAD dataset from original source
2. Place raw data in `data/raw/wesad/`
3. Preprocess:
```bash
python src/preprocessing/wesad.py --data_dir data/raw/wesad --output_dir data/processed/wesad
```

### Sleep-EDF Dataset

1. Download Sleep-EDF dataset from PhysioNet
2. Place raw data in `data/raw/sleep-edf/`
3. Preprocess:
```bash
python src/preprocessing/sleep_edf.py --data_dir data/raw/sleep-edf --output_dir data/processed/sleep-edf
```

## Running Experiments

### Baseline Comparison

```bash
python experiments/run_experiments.py --scenario baseline --auto
```

### Differential Privacy Experiments

```bash
python experiments/run_experiments.py --scenario dp --auto
```

### Federated Learning Experiments

```bash
python experiments/run_experiments.py --scenario fl --auto
```

### Combined FL+DP Experiments

```bash
python experiments/run_experiments.py --scenario dp_fl --auto
```

### Full Suite

```bash
python experiments/run_experiments.py --scenario all --auto
```

## Seed Configuration

All experiments use deterministic seeds:
- Default seed: 42
- Multi-run experiments: [42, 123, 456, 789, 1011]

Seeds are set before model initialization to ensure reproducibility.

## Expected Runtime

Approximate runtimes on standard hardware (CPU, 16GB RAM):

- Baseline (single run): ~5 minutes
- DP training (single run): ~8 minutes
- FL training (5 clients): ~15 minutes
- FL+DP training (5 clients): ~20 minutes

Full experimental suite (all scenarios, 5 runs each): ~4-6 hours

## Results Location

All results are saved to `results/experiments/` with the following structure:

```
results/
├── experiments/
│   ├── baseline/
│   ├── dp/
│   ├── fl/
│   └── dp_fl/
└── aggregated_metrics.json
```

## Verification

To verify reproducibility:

1. Run experiments with same seeds
2. Compare results with `results/aggregated_metrics.json`
3. Metrics should match within floating-point precision

## Hardware Notes

Experiments were run on:
- CPU: Apple Silicon M-series or equivalent
- Memory: 16GB+ recommended
- Storage: 5GB+ for datasets and results

GPU is not required (experiments use CPU for consistency).

