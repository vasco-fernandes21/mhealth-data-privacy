# MVP: Privacy-Preserving ML Training Platform

## Overview

The MVP (Minimum Viable Product) is an interactive web-based platform that provides real-time visualization and experimentation capabilities for privacy-preserving machine learning techniques applied to physiological signal classification. The platform serves as both a **proof-of-concept validation tool** and an **interactive demonstration** of the research findings presented in the paper.

## Core Architecture

### Same "Brain" as the Paper

The MVP uses **exactly the same training logic and implementations** as the batch experiments that generated the paper's results:

- **Identical Models**: Unified MLP architecture (same as paper experiments)
- **Same Trainers**: BaselineTrainer, DPTrainer, FederatedTrainer (identical implementations)
- **Same Privacy Mechanisms**: Opacus-based DP-SGD with identical gradient clipping and noise injection
- **Same Feature Extraction**: Hand-crafted features (statistical and spectral) from WESAD and Sleep-EDF datasets
- **Same Datasets**: Preprocessed WESAD and Sleep-EDF data using identical pipelines
- **Same Hyperparameters**: Default configurations match paper experiments (40 epochs, batch size 128, etc.)

This ensures that results observed in the MVP directly correspond to the quantitative findings reported in the paper, providing **empirical validation** of the theoretical analysis.

## Technical Stack

### Backend (FastAPI + PyTorch)
- **Framework**: FastAPI (Python 3.12+)
- **ML Framework**: PyTorch with Opacus for Differential Privacy
- **Database**: SQLite (SQLModel) for experiment history
- **Communication**: HTTP REST API with polling mechanism
- **Architecture**: Asynchronous job execution with background tasks

### Frontend (React + TypeScript)
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **UI Library**: Tailwind CSS
- **Charts**: Recharts for real-time visualization
- **State Management**: React Hooks (useState, useEffect, useCallback)
- **PWA**: Progressive Web App capabilities

## Functionalities

### 1. Interactive Training Configuration

**ConfigPanel Component** (`app/client/src/components/Setup/ConfigPanel.tsx`)

Users can configure training parameters through an intuitive interface:

- **Dataset Selection**: Toggle between WESAD (stress detection) and Sleep-EDF (sleep staging)
- **Privacy Noise (σ)**: Slider from 0.0 to 2.0 controlling DP noise multiplier
  - σ = 0: No differential privacy
  - σ > 0: DP enabled with formal privacy guarantees
- **Federated Clients**: Number of federated learning clients (0 = centralized, 1-10 = federated)
- **Advanced Parameters**:
  - Batch size (default: 128)
  - Max gradient norm (default: 5.0 for DP)
  - Class weights (enable/disable)
  - Number of runs (1-5 for multi-seed experiments)

**Mode Detection**: The system automatically detects the training mode based on configuration:
- `BASELINE_CENTRALIZED`: clients=0, σ=0
- `DP_CENTRALIZED`: clients=0, σ>0
- `FEDERATED_LEARNING`: clients>0, σ=0
- `FEDERATED_LEARNING_DP`: clients>0, σ>0

### 2. Real-Time Training Execution

**Training API** (`app/server/src/api/routes.py`)

- **POST `/api/v1/train`**: Start a new training job
  - Accepts TrainingConfig (dataset, clients, sigma, batch_size, epochs, etc.)
  - Returns job_id immediately for polling
  - Executes training asynchronously in background task

- **GET `/api/v1/status/{job_id}`**: Poll job status
  - Returns current progress, metrics, logs, and error messages
  - Frontend polls every 2 seconds during training
  - Provides real-time updates on:
    - Training progress (0-100%)
    - Current epoch/round
    - Accuracy, loss, epsilon (privacy budget)
    - Minority class recall
    - Training logs

- **POST `/api/v1/stop/{job_id}`**: Stop running job
  - Gracefully terminates training
  - Saves partial results

### 3. Real-Time Visualization Dashboard

**ChartsGrid Component** (`app/client/src/components/Dashboard/ChartsGrid.tsx`)

The dashboard displays multiple synchronized charts:

- **Accuracy Over Time**: Line chart showing accuracy progression across epochs/rounds
- **Privacy Budget (ε)**: Privacy budget accumulation during DP training
- **Minority Class Recall**: Fairness metric tracking minority class performance
- **Multi-Run Aggregation**: When multiple runs are executed, shows:
  - Mean accuracy with min/max bands
  - Standard deviation across seeds
  - Individual run trajectories

**StatGrid Component**: Summary statistics cards showing:
- Final accuracy
- Final epsilon (privacy budget)
- Training time
- Current round/epoch

### 4. Federated Learning Visualization

**ClientGrid Component** (`app/client/src/components/Simulation/ClientGrid.tsx`)

Visual representation of federated learning process:

- **Client Status**: Grid showing all federated clients
- **Active Client Highlighting**: Real-time indication of which client is currently training
- **Client Metrics**: Per-client accuracy and contribution visualization
- **Network Visualization**: Conceptual representation of client-server communication

### 5. Training Logs and Terminal

**TerminalLogs Component** (`app/client/src/components/Simulation/TerminalLogs.tsx`)

Real-time streaming of training logs:

- **Epoch/Round Progress**: Detailed logging of each training step
- **Client Activity**: Logs showing which client is computing in FL mode
- **Aggregation Events**: Logs for federated aggregation rounds
- **Error Messages**: Clear error reporting if training fails
- **Privacy Metrics**: Epsilon updates during DP training

### 6. Experiment History and Persistence

**Database Integration** (`app/server/src/core/database.py`)

- **SQLite Database**: Stores all training runs with full configuration and results
- **GET `/api/v1/history`**: List all previous experiments
- **GET `/api/v1/export/{run_id}`**: Export experiment results as JSON
- **DELETE `/api/v1/history/{run_id}`**: Remove experiment from history

Each experiment is saved with:
- Complete configuration (dataset, clients, sigma, hyperparameters)
- Training metrics (accuracy, F1, recall per class, confusion matrix)
- Privacy metrics (epsilon, delta, noise multiplier)
- Timing information (training time, epochs)
- Multi-run summaries (mean, std, individual runs)

### 7. Privacy Budget Estimation

**POST `/api/v1/estimate-privacy`**

Pre-training privacy budget estimation:

- Calculates expected epsilon before training starts
- Based on sample rate, noise multiplier, and number of steps
- Helps users understand privacy-utility trade-offs before committing to training

### 8. Job Management

**GET `/api/v1/jobs`**: List recent training jobs with status

- Shows queued, running, completed, and failed jobs
- Includes job metadata (dataset, mode, timestamp)
- Supports job resumption and monitoring

## Testing Suite

### Test Coverage (68 tests total)

**API Tests** (`test_api.py` - 11 tests):
- Health check endpoint
- Training job creation for all modes
- Job status polling
- Input validation
- Error handling
- Job stopping
- Job listing

**ML Component Tests** (`test_ml_components.py` - 27 tests):
- Trainer mode detection
- Privacy budget estimation (epsilon calculation)
- Class weight computation
- Configuration building
- Model creation and forward pass
- Dataset configuration loading

**Data Loader Tests** (`test_data_loader.py` - 4 tests):
- Singleton pattern verification
- Sleep-EDF dataset loading (24D features, 5 classes)
- WESAD dataset loading (140D features, 2 classes)
- Cache functionality

**Trainer Tests** (`test_trainers.py` - 24 tests):
- BaselineTrainer: Complete training cycle, early stopping, callbacks
- DPTrainer: PrivacyEngine setup, epsilon accumulation, DP evaluation
- FLClient: Client initialization, DP-enabled client, training rounds
- FederatedTrainer: FL training cycle, FL+DP training, weight aggregation

**Integration Tests** (`test_integration_real_training.py` - 1 test):
- End-to-end DP training cycle with real data
- Validates final accuracy and epsilon values

### Test Execution

```bash
# Run all tests
pytest app/server/tests/ -v

# Run specific module
pytest app/server/tests/test_trainers.py -v

# Run fast tests only (exclude slow integration tests)
pytest app/server/tests/ -v -m "not slow"
```

## What the MVP Enables

### 1. Interactive Privacy-Utility Trade-off Exploration

Users can **dynamically adjust** privacy parameters (σ) and **immediately observe** the impact on model accuracy:

- **Low σ (0.3-0.6)**: Strong privacy (ε ≈ 0.2-2.5) with accuracy degradation
- **Medium σ (0.6-1.0)**: Moderate privacy (ε ≈ 7-25) with balanced performance
- **High σ (1.0-2.0)**: Weak privacy (ε > 25) with minimal accuracy loss

This **validates the paper's key finding** that Sleep-EDF maintains strong privacy with minimal degradation, while WESAD requires weaker privacy to remain useful.

### 2. Federated Learning Validation

The MVP demonstrates that **data fragmentation** (multiple clients) has minimal impact when features are robust:

- Sleep-EDF: Maintains >88% accuracy even with 10 clients
- WESAD: Shows degradation only at extreme fragmentation (10 clients: 13.9% loss)

This **corroborates the paper's conclusion** that feature quality dominates privacy robustness.

### 3. Real-Time Observation of Gradient Clipping Trap

The MVP allows users to **observe in real-time** how class weights become ineffective under DP:

- Configure class weights and watch minority recall
- Compare with baseline (no DP) to see weight neutralization
- Validate the paper's finding that random seed dominates by 100:1 ratio

### 4. Multi-Seed Variance Analysis

When running multiple seeds (1-5 runs), the MVP shows:

- **Seed Variance**: Standard deviation across different initializations
- **Weight Variance**: Minimal impact of class weights (validates paper's finding)
- **Variance Ratio**: Demonstrates that seed variance >> weight variance

### 5. Proof-of-Concept for Practical Deployment

The MVP demonstrates **computational feasibility**:

- Training times match paper results (6-10× overhead for DP)
- FL maintains near-baseline training times
- FL+DP provides efficiency gains (4.7× vs 6-9× for centralized DP)

This validates that privacy-preserving ML is **practical for resource-constrained devices**.

## Communication Mechanism

### HTTP Polling (Current Implementation)

The MVP uses **HTTP polling** for real-time updates:

- Frontend polls `/api/v1/status/{job_id}` every 2 seconds
- Backend provides incremental updates (progress, metrics, logs)
- Simple, reliable, and works across all network configurations
- No WebSocket infrastructure required

**Note**: WebSocket code exists in `app/client/src/api/socket.ts` but is **not actively used**. The current implementation uses HTTP polling via `useTraining` hook.

## Data Flow

1. **User Configuration**: Frontend sends training config to `/api/v1/train`
2. **Job Creation**: Backend creates job_id and queues background task
3. **Training Execution**: Background task uses same trainers as paper experiments
4. **Progress Updates**: Trainer callbacks update job store with metrics/logs
5. **Status Polling**: Frontend polls `/api/v1/status/{job_id}` every 2s
6. **Real-Time Visualization**: Charts update as metrics arrive
7. **Completion**: Results saved to database, frontend shows final metrics

## Key Differentiators

### 1. Scientific Rigor

Unlike generic ML platforms, the MVP is **specifically designed** to validate the paper's findings:

- Same experimental setup as batch experiments
- Identical hyperparameters and configurations
- Direct correspondence between MVP results and paper tables

### 2. Privacy-First Design

The MVP emphasizes **privacy metrics** alongside accuracy:

- Real-time epsilon tracking
- Privacy-utility trade-off visualization
- Formal privacy guarantee display

### 3. Fairness Awareness

Built-in support for **class imbalance analysis**:

- Minority class recall tracking
- Class weight effectiveness validation
- Gradient clipping trap demonstration

### 4. Educational Value

The MVP serves as an **interactive learning tool**:

- Visual demonstration of FL aggregation
- Real-time observation of DP noise impact
- Clear visualization of privacy-utility trade-offs

## Limitations and Future Enhancements

### Current Limitations

1. **Single Device Execution**: All training runs on server (no true distributed FL)
2. **HTTP Polling**: Not as efficient as WebSockets for high-frequency updates
3. **Limited Dataset Support**: Only WESAD and Sleep-EDF (by design)
4. **No Model Persistence**: Trained models are not saved/exported
5. **Basic Visualization**: Charts are functional but could be more interactive

### Potential Enhancements

1. **True Distributed FL**: Support for actual client devices
2. **WebSocket Integration**: Real-time bidirectional communication
3. **Model Export**: Download trained models for inference
4. **Advanced Analytics**: More detailed per-class metrics, confusion matrices
5. **Comparison Mode**: Side-by-side comparison of multiple configurations
6. **Export Reports**: Generate PDF reports of experiments

## Conclusion

The MVP is not just a demonstration tool—it is a **validated proof-of-concept** that uses the exact same training logic as the paper's experiments. This ensures that:

1. **Results are Reproducible**: MVP results match paper findings
2. **Findings are Validated**: Interactive exploration confirms theoretical analysis
3. **Deployment is Feasible**: Demonstrates practical viability for mHealth applications
4. **Education is Enhanced**: Visual, interactive learning of privacy-preserving ML

The MVP bridges the gap between **theoretical research** (paper) and **practical application** (deployment), providing both validation and demonstration of the research contributions.

