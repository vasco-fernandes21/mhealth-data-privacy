# Privacy-Preserving Physiological Signal Classification

This repository contains the implementation and experimental code for the paper **"Privacy-Preserving Physiological Signal Classification in Mobile Health: A Comparative Study of Federated Learning and Differential Privacy"**.

It provides a comprehensive evaluation of privacy-preserving machine learning techniques applied to wearable sensor data, specifically focusing on the trade-offs between utility and privacy in mobile health (mHealth) scenarios.

## Scientific Goals

The primary objective of this project is to empirically evaluate how different privacy mechanisms affect model performance on real-world physiological data. We compare:

1.  **Federated Learning (FL)**: Decentralized training where raw data never leaves the client device.
2.  **Differential Privacy (DP)**: Formal privacy guarantees via gradient clipping and noise addition (DP-SGD).
3.  **FL + DP**: A combined approach offering both decentralization and formal privacy guarantees.

### Key Discovery: Class Weights vs. Differential Privacy

A major contribution of this work is the empirical demonstration that **standard class weighting techniques are rendered ineffective by gradient clipping in DP-SGD**.

-   In standard training, up-weighting minority classes amplifies their gradients to correct bias.
-   In DP training, per-sample gradient clipping (essential for privacy bounds) caps these amplified gradients, effectively neutralizing the weighting strategy.
-   **Result**: Random seed initialization dominates minority class performance by a ratio exceeding 100:1 compared to class weights.

## Datasets

We evaluate these methods on two distinct physiological signal datasets:

| Dataset | Task | Subjects | Signal Type | Challenge |
| :--- | :--- | :--- | :--- | :--- |
| **WESAD** | Stress Detection (Binary) | 15 | EDA, ECG, EMG | High Class Imbalance |
| **Sleep-EDF** | Sleep Staging (5-class) | ~80 | EEG, EOG | Multi-class Classification |

*Note: The system uses a unified MLP architecture on hand-crafted features (statistical and spectral) to ensure consistent comparisons across all experiments.*

## Project Structure

The codebase is organized to separate scientific logic from infrastructure:

```
mhealth-data-privacy/
├── src/
│   ├── configs/          # Experiment configurations (YAML)
│   ├── models/           # Unified MLP architecture (PyTorch)
│   ├── privacy/          # DP (Opacus) and FL implementation
│   ├── training/         # Specialized trainers (Baseline, DP, FL, DP+FL)
│   └── preprocessing/    # Feature extraction pipelines
├── experiments/
│   ├── scenarios/        # Definition of experimental scenarios
│   └── run_experiments.py # Main entry point for reproducibility
├── paper/                # LaTeX source and analysis scripts
└── results/              # Output logs and metrics
```

## Quick Start

To reproduce the experiments:

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Preprocess Data**:
    ```bash
    # Preprocess WESAD (example)
    python src/preprocessing/wesad.py --data_dir data/raw/wesad --output_dir data/processed/wesad
    ```

3.  **Run Experiments**:
    Experiments are driven by scenario files.
    ```bash
    # Run baseline comparison
    python experiments/run_experiments.py --scenario baseline --auto

    # Run full suite (Baseline, FL, DP, FL+DP)
    python experiments/run_experiments.py --scenario all --auto
    ```

## License

This project is intended for academic research purposes.
