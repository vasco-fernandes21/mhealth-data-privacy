#!/usr/bin/env python3
"""
Comprehensive test suite for all tiers (1, 2, 3).

Tests:
- Imports
- Configurations
- Models
- Trainers
- Data loading
- End-to-end training
"""

import sys
import os
from pathlib import Path
import tempfile
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import yaml


class TestRunner:
    """Test runner with results tracking."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.start_time = time.time()
    
    def test(self, name: str, func):
        """Run a test."""
        try:
            func()
            self.passed += 1
            print(f"  ✅ {name}")
            return True
        except Exception as e:
            self.failed += 1
            self.errors.append((name, str(e)))
            print(f"  ❌ {name}")
            print(f"     Error: {e}")
            return False
    
    def print_summary(self):
        """Print test summary."""
        elapsed = time.time() - self.start_time
        total = self.passed + self.failed
        
        print(f"\n{'='*70}")
        print(f"TEST SUMMARY")
        print(f"{'='*70}")
        print(f"✅ Passed: {self.passed}/{total}")
        print(f"❌ Failed: {self.failed}/{total}")
        print(f"⏱️  Time: {elapsed:.2f}s")
        
        if self.errors:
            print(f"\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error[:80]}")
        
        print(f"{'='*70}\n")
        
        return self.failed == 0


# ============================================================================
# TIER 1 TESTS
# ============================================================================

def test_tier1_imports():
    """Test Tier 1 imports."""
    print("\n" + "="*70)
    print("TIER 1: IMPORTS & BASIC COMPONENTS")
    print("="*70)
    
    runner = TestRunner()
    
    # Imports
    runner.test("Import seed_utils", lambda: __import__('src.utils.seed_utils', fromlist=['set_all_seeds']))
    runner.test("Import logging_utils", lambda: __import__('src.utils.logging_utils', fromlist=['setup_logging']))
    runner.test("Import base_model", lambda: __import__('src.models.base_model', fromlist=['BaseModel']))
    runner.test("Import sleep_edf_model", lambda: __import__('src.models.sleep_edf_model', fromlist=['SleepEDFModel']))
    runner.test("Import wesad_model", lambda: __import__('src.models.wesad_model', fromlist=['WESADModel']))
    runner.test("Import base_trainer", lambda: __import__('src.training.base_trainer', fromlist=['BaseTrainer']))
    runner.test("Import training utils", lambda: __import__('src.training.utils', fromlist=['ProgressBar']))
    
    return runner


def test_tier1_functionality():
    """Test Tier 1 functionality."""
    print("\n" + "="*70)
    print("TIER 1: FUNCTIONALITY TESTS")
    print("="*70)
    
    runner = TestRunner()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test seeding
    def test_seed():
        from src.utils.seed_utils import set_all_seeds
        set_all_seeds(42, verbose=False)
        x1 = torch.randn(5)
        set_all_seeds(42, verbose=False)
        x2 = torch.randn(5)
        assert torch.allclose(x1, x2), "Seeds not reproducible"
    
    runner.test("Seeding reproducibility", test_seed)
    
    # Test logging
    def test_logging():
        from src.utils.logging_utils import setup_logging, get_logger
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logging(tmpdir, level='INFO', verbose=False)
            assert logger is not None
    
    runner.test("Logging setup", test_logging)
    
    # Test Sleep-EDF model
    def test_sleep_edf():
        from src.models.sleep_edf_model import SleepEDFModel
        config = {
            'dataset': {'name': 'sleep-edf', 'n_features': 24, 'n_classes': 5, 'sequence_length': 10},
            'model': {'lstm_units': 128, 'lstm_layers': 2, 'dropout': 0.3, 'dense_layers': [64, 32]}
        }
        model = SleepEDFModel(config, device=device)
        x = torch.randn(32, 10, 24).to(device)
        y = model(x)
        assert y.shape == (32, 5), f"Expected (32, 5), got {y.shape}"
    
    runner.test("Sleep-EDF model forward pass", test_sleep_edf)
    
    # Test WESAD model
    def test_wesad():
        from src.models.wesad_model import WESADModel
        config = {
            'dataset': {'name': 'wesad', 'n_channels': 16, 'n_classes': 2, 'sequence_length': 1920},
            'model': {'lstm_units': 64, 'lstm_layers': 2, 'dropout': 0.3, 'dense_layers': [64, 32]}
        }
        model = WESADModel(config, device=device)
        x = torch.randn(64, 16, 1920).to(device)
        y = model(x)
        assert y.shape == (64, 2), f"Expected (64, 2), got {y.shape}"
    
    runner.test("WESAD model forward pass", test_wesad)
    
    # Test gradient monitor
    def test_gradient_monitor():
        from src.training.utils import GradientMonitor
        model = nn.Linear(10, 5)
        monitor = GradientMonitor(model)
        x = torch.randn(32, 10)
        y = torch.randint(0, 5, (32,))
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        
        grad_stats = monitor.log_gradients()
        assert grad_stats is not None
        assert 'norm' in grad_stats
    
    runner.test("Gradient monitor", test_gradient_monitor)
    
    return runner


# ============================================================================
# TIER 2 TESTS
# ============================================================================

def test_tier2_configs():
    """Test Tier 2 configurations."""
    print("\n" + "="*70)
    print("TIER 2: CONFIGURATIONS")
    print("="*70)
    
    runner = TestRunner()
    
    config_dir = Path(__file__).parent.parent / 'src/configs'
    
    # Test training defaults
    def test_training_defaults():
        config_file = config_dir / 'training_defaults.yaml'
        assert config_file.exists(), f"Config not found: {config_file}"
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
        assert 'training' in cfg
        assert 'epochs' in cfg['training']
    
    runner.test("Training defaults config", test_training_defaults)
    
    # Test privacy defaults
    def test_privacy_defaults():
        config_file = config_dir / 'privacy_defaults.yaml'
        assert config_file.exists()
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
        assert 'differential_privacy' in cfg
        assert 'federated_learning' in cfg
    
    runner.test("Privacy defaults config", test_privacy_defaults)
    
    # Test Sleep-EDF config
    def test_sleep_edf_config():
        config_file = config_dir / 'sleep_edf.yaml'
        assert config_file.exists()
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
        assert cfg['dataset']['name'] == 'sleep-edf'
        assert cfg['dataset']['n_features'] == 24
    
    runner.test("Sleep-EDF config", test_sleep_edf_config)
    
    # Test WESAD config
    def test_wesad_config():
        config_file = config_dir / 'wesad.yaml'
        assert config_file.exists()
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
        assert cfg['dataset']['name'] == 'wesad'
        assert cfg['dataset']['n_channels'] == 16
    
    runner.test("WESAD config", test_wesad_config)
    
    return runner


def test_tier2_trainers():
    """Test Tier 2 trainers."""
    print("\n" + "="*70)
    print("TIER 2: TRAINERS")
    print("="*70)
    
    runner = TestRunner()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test baseline trainer import
    def test_baseline_import():
        from src.training.trainers.baseline_trainer import BaselineTrainer
        assert BaselineTrainer is not None
    
    runner.test("Baseline trainer import", test_baseline_import)
    
    # Test DP trainer import
    def test_dp_import():
        from src.training.trainers.dp_trainer import DPTrainer
        assert DPTrainer is not None
    
    runner.test("DP trainer import", test_dp_import)
    
    # Test DP utils
    def test_dp_utils():
        from src.privacy.dp_utils import DPConfig, check_dp_compatibility
        config = {
            'differential_privacy': {
                'enabled': True,
                'target_epsilon': 8.0,
                'target_delta': 1e-5
            }
        }
        dp_cfg = DPConfig(config)
        assert dp_cfg.enabled
        assert dp_cfg.target_epsilon == 8.0
    
    runner.test("DP utilities", test_dp_utils)
    
    return runner


# ============================================================================
# TIER 3 TESTS
# ============================================================================

def test_tier3_components():
    """Test Tier 3 components."""
    print("\n" + "="*70)
    print("TIER 3: FEDERATED LEARNING & EVALUATION")
    print("="*70)
    
    runner = TestRunner()
    
    # Test FL client import
    def test_fl_client():
        from src.privacy.fl_client import FLClient
        assert FLClient is not None
    
    runner.test("FL client import", test_fl_client)
    
    # Test FL aggregators
    def test_fl_aggregators():
        from src.privacy.fl_aggregators import FedAvgAggregator, create_aggregator
        agg = create_aggregator('fedavg')
        assert agg is not None
    
    runner.test("FL aggregators", test_fl_aggregators)
    
    # Test FL server import
    def test_fl_server():
        from src.privacy.fl_server import FLServer
        assert FLServer is not None
    
    runner.test("FL server import", test_fl_server)
    
    # Test FL trainer import
    def test_fl_trainer():
        from src.training.trainers.fl_trainer import FLTrainer
        assert FLTrainer is not None
    
    runner.test("FL trainer import", test_fl_trainer)
    
    # Test FL+DP trainer import
    def test_fl_dp_trainer():
        from src.training.trainers.fl_dp_trainer import FLDPTrainer
        assert FLDPTrainer is not None
    
    runner.test("FL+DP trainer import", test_fl_dp_trainer)
    
    # Test metrics calculator
    def test_metrics():
        from src.evaluation.metrics import MetricsCalculator
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 1])
        calc = MetricsCalculator()
        metrics = calc.compute_metrics(y_true, y_pred)
        assert 'accuracy' in metrics
        assert 'f1_weighted' in metrics
    
    runner.test("Metrics calculator", test_metrics)
    
    # Test privacy-utility analyzer
    def test_analyzer():
        from src.evaluation.privacy_utility_analysis import PrivacyUtilityAnalyzer
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = PrivacyUtilityAnalyzer(tmpdir)
            assert analyzer is not None
    
    runner.test("Privacy-utility analyzer", test_analyzer)
    
    # Test model factory
    def test_model_factory():
        from src.models.model_factory import create_model
        config_sleep = {
            'dataset': {'name': 'sleep-edf', 'n_features': 24, 'n_classes': 5},
            'model': {'lstm_units': 128, 'lstm_layers': 2, 'dropout': 0.3, 'dense_layers': [64, 32]}
        }
        model = create_model('sleep-edf', config_sleep, device='cpu')
        assert model is not None
    
    runner.test("Model factory", test_model_factory)
    
    # Test data quality analyzer
    def test_quality_analyzer():
        from src.preprocessing.quality_report import DataQualityAnalyzer
        X = np.random.randn(100, 24)
        y = np.random.randint(0, 5, 100)
        report = DataQualityAnalyzer.analyze_dataset(X, y, 'test')
        assert 'n_samples' in report
        assert report['n_samples'] == 100
    
    runner.test("Data quality analyzer", test_quality_analyzer)
    
    return runner


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_end_to_end():
    """End-to-end integration test."""
    print("\n" + "="*70)
    print("END-TO-END INTEGRATION TEST")
    print("="*70)
    
    runner = TestRunner()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Complete training pipeline simulation
    def test_full_pipeline():
        from src.utils.seed_utils import set_all_seeds
        from src.models.sleep_edf_model import SleepEDFModel
        from src.training.trainers.baseline_trainer import BaselineTrainer
        from torch.utils.data import TensorDataset, DataLoader
        
        # Setup
        set_all_seeds(42, verbose=False)
        
        # Create config
        config = {
            'dataset': {
                'name': 'sleep-edf',
                'n_features': 24,
                'n_classes': 5,
                'sequence_length': 10,
                'class_names': ['W', 'N1', 'N2', 'N3', 'R']
            },
            'model': {
                'lstm_units': 64,
                'lstm_layers': 1,
                'dropout': 0.2,
                'dense_layers': [32, 16]
            },
            'training': {
                'batch_size': 16,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'gradient_clipping': True,
                'gradient_clip_norm': 1.0
            }
        }
        
        # Create data
        X_train = np.random.randn(100, 10, 24).astype(np.float32)
        y_train = np.random.randint(0, 5, 100)
        X_val = np.random.randn(20, 10, 24).astype(np.float32)
        y_val = np.random.randint(0, 5, 20)
        
        # Create dataloaders
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Create model
        model = SleepEDFModel(config, device=device)
        
        # Create trainer
        trainer = BaselineTrainer(model, config, device=device)
        
        # Train (just 2 epochs for testing)
        results = trainer.fit(
            train_loader,
            val_loader,
            epochs=2,
            patience=10
        )
        
        assert 'total_epochs' in results
        assert 'best_val_acc' in results
        assert results['total_epochs'] >= 1
    
    runner.test("Full training pipeline", test_full_pipeline)
    
    return runner


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all tests."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  COMPREHENSIVE TIER TEST SUITE".center(68) + "║")
    print("║" + "  Validating Tier 1, 2, and 3 Components".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    all_passed = True
    
    # Run all test suites
    all_passed &= test_tier1_imports().print_summary()
    all_passed &= test_tier1_functionality().print_summary()
    all_passed &= test_tier2_configs().print_summary()
    all_passed &= test_tier2_trainers().print_summary()
    all_passed &= test_tier3_components().print_summary()
    all_passed &= test_end_to_end().print_summary()
    
    # Final summary
    print("\n")
    print("╔" + "="*68 + "╗")
    if all_passed:
        print("║" + "  ✅ ALL TESTS PASSED - ARCHITECTURE IS FUNCTIONAL".center(68) + "║")
    else:
        print("║" + "  ❌ SOME TESTS FAILED - REVIEW ERRORS ABOVE".center(68) + "║")
    print("╚" + "="*68 + "╝")
    print("\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())