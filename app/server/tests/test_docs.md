# Test Documentation

## Overview

The test suite consists of 68 tests organized across 5 test modules, covering API endpoints, ML components, data loading, trainers, and integration scenarios.

## Test Structure

### Test Files

- `test_api.py` - REST API endpoint tests (11 tests)
- `test_ml_components.py` - ML component unit tests (27 tests)
- `test_data_loader.py` - Data loading tests (4 tests)
- `test_trainers.py` - Trainer implementation tests (24 tests)
- `test_integration_real_training.py` - End-to-end integration tests (1 test)

## Test Modules

### API Tests (`test_api.py`)

Tests REST API endpoints for training job management.

**Endpoints Tested:**
- `GET /health` - Health check endpoint
- `POST /api/v1/train` - Start training job
- `GET /api/v1/status/{job_id}` - Get job status
- `POST /api/v1/stop/{job_id}` - Stop running job
- `GET /api/v1/jobs` - List recent jobs

**Test Cases:**
- Health check returns 200 status
- Training job creation for all modes (baseline, DP, FL, FL+DP)
- Job status polling and retrieval
- Input validation (invalid parameters)
- Error handling (nonexistent jobs)
- Job stopping functionality
- Job listing with limits

**Fixtures:**
- `client` - FastAPI test client with API key authentication
- `mock_trainer` - Mocked trainer factory for fast execution
- `sample_config` - Standard training configuration

### ML Components Tests (`test_ml_components.py`)

Unit tests for core ML functionality.

**Test Classes:**

**TestGetTrainerMode** - Trainer mode detection
- Verifies correct mode selection based on clients and sigma parameters
- Tests all four modes: BASELINE_CENTRALIZED, DP_CENTRALIZED, FEDERATED_LEARNING, FEDERATED_LEARNING_DP

**TestEstimateEpsilon** - Privacy budget estimation
- Edge cases: zero/negative sigma, invalid sample rates
- Valid epsilon calculation
- Epsilon accumulation with steps
- Steps vs epochs flag handling

**TestComputeClassWeights** - Class weight computation
- Balanced class handling
- Imbalanced class weighting
- Single class scenarios
- Empty class edge cases

**TestBuildConfig** - Configuration building
- DP enable/disable based on sigma
- FL configuration when clients > 0
- Parameter overrides (max_grad_norm, class_weights)

**TestGetModel** - Model creation
- Model instantiation
- Forward pass functionality
- Opacus compatibility validation

**TestGetConfig** - Dataset configuration
- WESAD and Sleep-EDF config loading
- Unknown dataset error handling
- Parameter updates (sigma, clients)

### Data Loader Tests (`test_data_loader.py`)

Tests for dataset loading functionality.

**Test Cases:**
- Singleton pattern verification
- Sleep-EDF dataset loading (24D features, 5 classes)
- WESAD dataset loading (140D features, 2 classes)
- Cache functionality
- Unknown dataset error handling

**Mocks:**
- Dataset file loading functions
- NumPy file operations
- Path existence checks

### Trainer Tests (`test_trainers.py`)

Comprehensive tests for training implementations.

**TestBaselineTrainer** - Baseline centralized training
- Optimizer and loss setup
- Single epoch training
- Validation functionality
- Complete training cycle
- History tracking
- Callback invocation
- Full evaluation metrics
- Early stopping mechanism

**TestDPTrainer** - Differential privacy training
- PrivacyEngine setup
- DP-enabled training
- Epsilon accumulation
- DP-disabled fallback
- Evaluation with DP

**TestFLClient** - Federated learning client
- Client initialization
- DP-enabled client setup
- Training round execution
- Epsilon tracking in metrics

**TestFederatedTrainer** - Federated learning trainer
- Trainer initialization
- DP-enabled FL setup
- Complete FL training cycle
- FL+DP training cycle
- Full evaluation metrics
- Weight aggregation (FedAvg)
- Validation frequency handling

**Fixtures:**
- `sample_config` - Training configuration with all required sections
- `sample_data` - Synthetic train/validation data loaders

### Integration Tests (`test_integration_real_training.py`)

End-to-end tests with real training execution.

**Test Cases:**
- `test_real_dp_training_cycle` - Complete DP training cycle
  - Creates DPTrainer with sigma=0.6
  - Executes full fit() and evaluate_full()
  - Validates final accuracy and epsilon values
  - Marked as slow test (`@pytest.mark.slow`)

## Test Execution

### Running All Tests

```bash
pytest tests/ -v
```

### Running Specific Test Module

```bash
pytest tests/test_trainers.py -v
```

### Running Specific Test Class

```bash
pytest tests/test_trainers.py::TestBaselineTrainer -v
```

### Running Specific Test

```bash
pytest tests/test_trainers.py::TestBaselineTrainer::test_fit_completes -v
```

### Running Fast Tests Only

```bash
pytest tests/ -v -m "not slow"
```

## Test Coverage

### Current Coverage

| Component | Tests | Coverage |
|-----------|-------|----------|
| API Endpoints | 11 | 100% |
| Factory/Config | 27 | 90% |
| Data Loading | 4 | 30% |
| Trainers | 24 | 80% |
| Integration | 1 | Basic |

### Coverage Gaps

**High Priority:**
- Trainer error handling scenarios
- Privacy engine edge cases
- FL aggregation edge cases
- Concurrent job operations

**Medium Priority:**
- Security (API key validation)
- Job store thread safety
- Evaluation metric edge cases

## Test Fixtures

### Shared Fixtures (`conftest.py`)

- `client` - FastAPI test client with authentication headers
- `clean_job_store` - Automatic job store cleanup between tests
- `mock_trainer` - Mocked trainer factory for API tests
- `sample_config` - Standard training configuration

### Module-Specific Fixtures

- `sample_data` - Synthetic data loaders (test_trainers.py)
- `sample_config` - Extended config with FL settings (test_trainers.py)

## Test Patterns

### Mocking Strategy

- External dependencies (file I/O, dataset loading) are mocked
- ML training uses real implementations with synthetic data
- API tests mock trainer factory for speed

### Isolation

- Each test is independent
- Job store is cleared before each test
- No shared state between tests

### Assertions

- Type checking for return values
- Range validation for metrics (0-1 for accuracy)
- Dictionary key existence checks
- Exception handling verification

## Maintenance

### Adding New Tests

1. Follow existing test class structure
2. Use appropriate fixtures
3. Mock external dependencies
4. Keep tests isolated and independent
5. Use descriptive test names

### Test Naming Convention

- Test classes: `Test<ComponentName>`
- Test methods: `test_<functionality>_<expected_behavior>`