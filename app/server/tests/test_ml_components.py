import pytest
import numpy as np
import torch
from src.ml.factory import get_trainer_mode, build_config
from src.ml.privacy import estimate_epsilon
from src.core.dataset_configs import compute_class_weights, get_config
from src.ml.models import get_model


class TestGetTrainerMode:
    def test_baseline_centralized(self):
        assert get_trainer_mode(0, 0.0) == "BASELINE_CENTRALIZED"
    
    def test_dp_centralized(self):
        assert get_trainer_mode(0, 1.0) == "DP_CENTRALIZED"
    
    def test_federated_learning(self):
        assert get_trainer_mode(5, 0.0) == "FEDERATED_LEARNING"
    
    def test_federated_learning_dp(self):
        assert get_trainer_mode(5, 1.0) == "FEDERATED_LEARNING_DP"


class TestEstimateEpsilon:
    def test_zero_sigma_returns_zero(self):
        assert estimate_epsilon(0.0, 0.1, 10) == 0.0
    
    def test_negative_sigma_returns_zero(self):
        assert estimate_epsilon(-1.0, 0.1, 10) == 0.0
    
    def test_invalid_sample_rate_returns_inf(self):
        assert estimate_epsilon(1.0, 0.0, 10) == float('inf')
        assert estimate_epsilon(1.0, 1.5, 10) == float('inf')
    
    def test_valid_epsilon_estimation(self):
        epsilon = estimate_epsilon(1.0, 0.1, 10, delta=1e-5)
        assert isinstance(epsilon, float)
        assert epsilon >= 0.0
    
    def test_epsilon_increases_with_steps(self):
        eps1 = estimate_epsilon(1.0, 0.1, 10)
        eps2 = estimate_epsilon(1.0, 0.1, 20)
        assert eps2 >= eps1
    
    def test_is_steps_flag(self):
        eps_epochs = estimate_epsilon(1.0, 0.1, 100, is_steps=False)
        eps_steps = estimate_epsilon(1.0, 0.1, 100, is_steps=True)
        assert isinstance(eps_epochs, float)
        assert isinstance(eps_steps, float)


class TestComputeClassWeights:
    def test_balanced_classes(self):
        y = np.array([0, 0, 1, 1])
        weights = compute_class_weights(y)
        assert len(weights) == 2
        assert weights[0] == weights[1]
    
    def test_imbalanced_classes(self):
        y = np.array([0, 0, 0, 1])
        weights = compute_class_weights(y)
        assert weights[1] > weights[0]
    
    def test_single_class(self):
        y = np.array([0, 0, 0])
        weights = compute_class_weights(y)
        assert weights[0] == 1.0
    
    def test_empty_class_handling(self):
        y = np.array([0, 0, 0, 0])
        weights = compute_class_weights(y)
        assert 0 in weights
        assert weights[0] == 1.0


class TestBuildConfig:
    def test_dp_enabled_when_sigma_positive(self):
        config = build_config("wesad", 0, 1.0)
        assert config['differential_privacy']['enabled'] is True
        assert config['differential_privacy']['noise_multiplier'] == 1.0
    
    def test_dp_disabled_when_sigma_zero(self):
        config = build_config("wesad", 0, 0.0)
        assert config['differential_privacy']['enabled'] is False
    
    def test_fl_config_when_clients_positive(self):
        config = build_config("wesad", 5, 0.0)
        assert config['federated_learning']['n_clients'] == 5
        assert config['federated_learning']['enabled'] is True
    
    def test_max_grad_norm_override(self):
        config = build_config("wesad", 0, 1.0, max_grad_norm=10.0)
        assert config['differential_privacy']['max_grad_norm'] == 10.0
    
    def test_class_weights_override(self):
        y = np.array([0, 0, 1, 1])
        config = build_config("wesad", 0, 0.0, train_y=y, use_class_weights=True)
        assert 'class_weights' in config['dataset']


class TestGetModel:
    def test_model_creation(self):
        model = get_model(140, 2)
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_model_forward(self):
        model = get_model(140, 2)
        x = torch.randn(1, 140)
        output = model(x)
        assert output.shape == (1, 2)
    
    def test_model_opacus_compatible(self):
        model = get_model(140, 2)
        from opacus.validators import ModuleValidator
        assert ModuleValidator.is_valid(model)


class TestGetConfig:
    def test_wesad_config(self):
        config = get_config("wesad", 5, 1.0)
        assert config['dataset']['name'] == "wesad"
        assert config['dataset']['input_dim'] == 140
        assert config['dataset']['n_classes'] == 2
    
    def test_sleep_edf_config(self):
        config = get_config("sleep-edf", 5, 1.0)
        assert config['dataset']['name'] == "sleep-edf"
        assert config['dataset']['input_dim'] == 24
        assert config['dataset']['n_classes'] == 5
    
    def test_unknown_dataset_raises_error(self):
        with pytest.raises(ValueError):
            get_config("unknown", 5, 1.0)
    
    def test_config_updates_sigma(self):
        config = get_config("wesad", 5, 2.5)
        assert config['differential_privacy']['noise_multiplier'] == 2.5
    
    def test_config_updates_clients(self):
        config = get_config("wesad", 10, 1.0)
        assert config['federated_learning']['n_clients'] == 10

