#!/usr/bin/env python3
"""
Comprehensive test suite for Tier 1 components.

Tests:
1. seed_utils - Seeding reproducibility
2. logging_utils - Logging setup
3. base_model - Abstract model class
4. sleep_edf_model - Sleep-EDF model
5. wesad_model - WESAD model
6. base_trainer - Training base class
7. training/utils - Progress bar, gradient monitor, etc
"""

import sys
import os
from pathlib import Path
import tempfile
import json

import torch
import torch.nn as nn
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed_utils import set_all_seeds, set_deterministic, RandomState
from src.utils.logging_utils import setup_logging, get_logger, log_config, log_metrics
from src.models.base_model import BaseModel
from src.models.sleep_edf_model import SleepEDFModel
from src.models.wesad_model import WESADModel
from src.training.utils import ProgressBar, GradientMonitor, LearningRateScheduler


class TestResults:
    """Track test results."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_pass(self, test_name: str):
        self.passed += 1
        print(f"  ✅ {test_name}")
    
    def add_fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append((test_name, error))
        print(f"  ❌ {test_name}: {error}")
    
    def print_summary(self):
        print(f"\n{'='*70}")
        print(f"TEST SUMMARY")
        print(f"{'='*70}")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        
        if self.errors:
            print(f"\nFailed tests:")
            for test_name, error in self.errors:
                print(f"  - {test_name}: {error}")
        
        print(f"{'='*70}\n")
        
        return self.failed == 0


# ============================================================================
# TEST 1: seed_utils
# ============================================================================

def test_seed_utils():
    """Test seed_utils reproducibility."""
    print("\n" + "="*70)
    print("TEST 1: seed_utils - Seeding Reproducibility")
    print("="*70)
    
    results = TestResults()
    
    try:
        # Test 1.1: Basic seeding
        set_all_seeds(42, verbose=False)
        x1 = torch.randn(10)
        
        set_all_seeds(42, verbose=False)
        x2 = torch.randn(10)
        
        if torch.allclose(x1, x2):
            results.add_pass("Basic seeding reproducibility")
        else:
            results.add_fail("Basic seeding", "Tensors not equal after same seed")
        
        # Test 1.2: NumPy seeding
        set_all_seeds(123, verbose=False)
        np_arr1 = np.random.randn(10)
        
        set_all_seeds(123, verbose=False)
        np_arr2 = np.random.randn(10)
        
        if np.allclose(np_arr1, np_arr2):
            results.add_pass("NumPy seeding reproducibility")
        else:
            results.add_fail("NumPy seeding", "Arrays not equal after same seed")
        
        # Test 1.3: RandomState context manager
        set_all_seeds(42, verbose=False)
        y1 = torch.randn(5)
        
        with RandomState(999):
            y_temp = torch.randn(5)
        
        y2 = torch.randn(5)
        
        # After context, should return to original seed state
        set_all_seeds(42, verbose=False)
        torch.randn(5)  # Skip first
        y3 = torch.randn(5)
        
        if torch.allclose(y2, y3):
            results.add_pass("RandomState context manager")
        else:
            results.add_fail("RandomState context", "State not restored after context")
        
        # Test 1.4: Deterministic mode (just check no error)
        try:
            set_deterministic('cuda' if torch.cuda.is_available() else 'cpu', verbose=False)
            results.add_pass("Deterministic mode setting")
        except Exception as e:
            results.add_fail("Deterministic mode", str(e))
        
    except Exception as e:
        results.add_fail("seed_utils general", str(e))
    
    results.print_summary()
    return results.failed == 0


# ============================================================================
# TEST 2: logging_utils
# ============================================================================

def test_logging_utils():
    """Test logging utilities."""
    print("\n" + "="*70)
    print("TEST 2: logging_utils - Logging Setup")
    print("="*70)
    
    results = TestResults()
    
    try:
        # Test 2.1: Basic logging setup
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logging(output_dir=tmpdir, level='DEBUG', verbose=False)
            
            if logger is not None:
                results.add_pass("Basic logging setup")
            else:
                results.add_fail("Basic logging setup", "Logger is None")
            
            # Test 2.2: Get logger
            logger2 = get_logger("test.module")
            if logger2 is not None:
                results.add_pass("Get logger")
            else:
                results.add_fail("Get logger", "Logger is None")
            
            # Test 2.3: Log config
            config = {'dataset': 'sleep-edf', 'model': {'type': 'lstm'}}
            try:
                log_config(config, logger)
                results.add_pass("Log config")
            except Exception as e:
                results.add_fail("Log config", str(e))
            
            # Test 2.4: Log metrics
            metrics = {'accuracy': 0.95, 'loss': 0.123}
            try:
                log_metrics(metrics, logger, step=1)
                results.add_pass("Log metrics")
            except Exception as e:
                results.add_fail("Log metrics", str(e))
            
    
    except Exception as e:
        results.add_fail("logging_utils general", str(e))
    
    results.print_summary()
    return results.failed == 0


# ============================================================================
# TEST 3: base_model
# ============================================================================

def test_base_model():
    """Test base model functionality."""
    print("\n" + "="*70)
    print("TEST 3: base_model - Abstract Model Class")
    print("="*70)
    
    results = TestResults()
    
    try:
        # Note: Can't directly instantiate abstract class, so we'll test through subclasses
        # But we can test that the class exists and has the right methods
        
        # Test 3.1: Check base model class exists
        if hasattr(BaseModel, 'forward'):
            results.add_pass("BaseModel has forward method")
        else:
            results.add_fail("BaseModel", "Missing forward method")
        
        # Test 3.2: Check key methods exist
        required_methods = ['to_device', 'get_model_info', 'save', 'load', 'freeze', 'unfreeze']
        for method in required_methods:
            if hasattr(BaseModel, method):
                results.add_pass(f"BaseModel has {method} method")
            else:
                results.add_fail(f"BaseModel {method}", f"Missing {method} method")
    
    except Exception as e:
        results.add_fail("base_model general", str(e))
    
    results.print_summary()
    return results.failed == 0


# ============================================================================
# TEST 4: sleep_edf_model
# ============================================================================

def test_sleep_edf_model():
    """Test Sleep-EDF model."""
    print("\n" + "="*70)
    print("TEST 4: sleep_edf_model - Sleep-EDF Model")
    print("="*70)
    
    results = TestResults()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Test 4.1: Model creation
        config = {
            'dataset': {
                'name': 'sleep-edf',
                'n_features': 24,
                'n_classes': 5,
                'sequence_length': 10
            },
            'model': {
                'lstm_units': 128,
                'lstm_layers': 2,
                'dropout': 0.3,
                'dense_layers': [64, 32]
            }
        }
        
        model = SleepEDFModel(config, device=device)
        results.add_pass("Model creation")
        
        # Test 4.2: Forward pass shape
        x = torch.randn(32, 10, 24).to(device)  # (batch, sequence, features)
        y = model(x)
        
        if y.shape == (32, 5):
            results.add_pass("Forward pass output shape")
        else:
            results.add_fail("Forward pass", f"Expected (32, 5), got {y.shape}")
        
        # Test 4.3: Get model info
        try:
            info = model.get_model_info()
            if 'total_parameters' in info and 'model_size_mb' in info:
                results.add_pass("Get model info")
            else:
                results.add_fail("Get model info", "Missing keys in info")
        except Exception as e:
            results.add_fail("Get model info", str(e))
        
        # Test 4.4: Model save/load
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'model.pth'
            
            try:
                model.save(str(save_path))
                results.add_pass("Model save")
            except Exception as e:
                results.add_fail("Model save", str(e))
            
            try:
                loaded_model = SleepEDFModel.load(str(save_path), device=device)
                results.add_pass("Model load")
            except Exception as e:
                results.add_fail("Model load", str(e))
            
            # Test 4.5: Loaded model forward pass
            try:
                y_loaded = loaded_model(x)
                if torch.allclose(y, y_loaded):
                    results.add_pass("Loaded model forward pass consistency")
                else:
                    results.add_fail("Loaded model", "Outputs not equal")
            except Exception as e:
                results.add_fail("Loaded model forward pass", str(e))
        
        # Test 4.6: Freeze/unfreeze
        try:
            model.freeze()
            frozen = not any(p.requires_grad for p in model.parameters())
            if frozen:
                results.add_pass("Model freeze")
            else:
                results.add_fail("Model freeze", "Parameters still require grad")
            
            model.unfreeze()
            unfrozen = any(p.requires_grad for p in model.parameters())
            if unfrozen:
                results.add_pass("Model unfreeze")
            else:
                results.add_fail("Model unfreeze", "Parameters don't require grad")
        except Exception as e:
            results.add_fail("Freeze/unfreeze", str(e))
    
    except Exception as e:
        results.add_fail("sleep_edf_model general", str(e))
    
    results.print_summary()
    return results.failed == 0


# ============================================================================
# TEST 5: wesad_model
# ============================================================================

def test_wesad_model():
    """Test WESAD model."""
    print("\n" + "="*70)
    print("TEST 5: wesad_model - WESAD Model")
    print("="*70)
    
    results = TestResults()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Test 5.1: Model creation
        config = {
            'dataset': {
                'name': 'wesad',
                'n_channels': 16,
                'n_classes': 2,
                'sequence_length': 1920
            },
            'model': {
                'lstm_units': 64,
                'lstm_layers': 2,
                'dropout': 0.3,
                'dense_layers': [64, 32]
            }
        }
        
        model = WESADModel(config, device=device)
        results.add_pass("Model creation")
        
        # Test 5.2: Forward pass shape
        x = torch.randn(64, 16, 1920).to(device)  # (batch, channels, timesteps)
        y = model(x)
        
        if y.shape == (64, 2):
            results.add_pass("Forward pass output shape")
        else:
            results.add_fail("Forward pass", f"Expected (64, 2), got {y.shape}")
        
        # Test 5.3: Get model info
        try:
            info = model.get_model_info()
            if 'total_parameters' in info and 'model_size_mb' in info:
                results.add_pass("Get model info")
            else:
                results.add_fail("Get model info", "Missing keys")
        except Exception as e:
            results.add_fail("Get model info", str(e))
        
        # Test 5.4: Model save/load
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'model.pth'
            
            try:
                model.save(str(save_path))
                results.add_pass("Model save")
            except Exception as e:
                results.add_fail("Model save", str(e))
            
            try:
                loaded_model = WESADModel.load(str(save_path), device=device)
                results.add_pass("Model load")
            except Exception as e:
                results.add_fail("Model load", str(e))
        
        # Test 5.5: GroupNorm usage (DP-safety)
        try:
            has_groupnorm = any(isinstance(m, nn.GroupNorm) for m in model.modules())
            if has_groupnorm:
                results.add_pass("Model uses GroupNorm (DP-safe)")
            else:
                results.add_fail("GroupNorm", "Model doesn't use GroupNorm")
        except Exception as e:
            results.add_fail("GroupNorm check", str(e))
    
    except Exception as e:
        results.add_fail("wesad_model general", str(e))
    
    results.print_summary()
    return results.failed == 0


# ============================================================================
# TEST 6: base_trainer
# ============================================================================

def test_base_trainer():
    """Test base trainer functionality."""
    print("\n" + "="*70)
    print("TEST 6: base_trainer - Base Trainer Class")
    print("="*70)
    
    results = TestResults()
    
    try:
        # Test 6.1: Check class exists and has required methods
        from src.training.base_trainer import BaseTrainer
        
        required_methods = ['fit', 'validate', 'save_checkpoint', 'load_checkpoint', 'save_results']
        for method in required_methods:
            if hasattr(BaseTrainer, method):
                results.add_pass(f"BaseTrainer has {method} method")
            else:
                results.add_fail(f"BaseTrainer {method}", f"Missing {method}")
    
    except Exception as e:
        results.add_fail("base_trainer general", str(e))
    
    results.print_summary()
    return results.failed == 0


# ============================================================================
# TEST 7: training/utils
# ============================================================================

def test_training_utils():
    """Test training utilities."""
    print("\n" + "="*70)
    print("TEST 7: training/utils - Progress Bar & Monitors")
    print("="*70)
    
    results = TestResults()
    
    try:
        # Test 7.1: ProgressBar
        try:
            pbar = ProgressBar(100, "Test")
            for i in range(100):
                pbar.update(1)
            pbar.finish()
            results.add_pass("ProgressBar basic usage")
        except Exception as e:
            results.add_fail("ProgressBar", str(e))
        
        # Test 7.2: GradientMonitor
        try:
            model = nn.Linear(10, 5)
            monitor = GradientMonitor(model)
            
            x = torch.randn(32, 10)
            y = torch.randint(0, 5, (32,))
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters())
            
            for _ in range(3):
                optimizer.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                
                grad_stats = monitor.log_gradients()
                if grad_stats is not None:
                    results.add_pass("GradientMonitor log_gradients")
                else:
                    results.add_fail("GradientMonitor", "No gradients logged")
                    break
                
                optimizer.step()
            
            # Test 7.3: Get summary
            try:
                summary = monitor.get_summary()
                if 'norm_mean' in summary:
                    results.add_pass("GradientMonitor get_summary")
                else:
                    results.add_fail("GradientMonitor summary", "Missing keys")
            except Exception as e:
                results.add_fail("GradientMonitor summary", str(e))
        
        except Exception as e:
            results.add_fail("GradientMonitor", str(e))
        
        # Test 7.4: LearningRateScheduler
        try:
            model = nn.Linear(10, 5)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = LearningRateScheduler(optimizer, initial_lr=0.001)
            
            # Test exponential decay
            lr1 = scheduler.step_exponential(0, decay_rate=0.5, decay_steps=5)
            lr2 = scheduler.step_exponential(1, decay_rate=0.5, decay_steps=5)
            
            if lr2 < lr1:
                results.add_pass("LearningRateScheduler exponential decay")
            else:
                results.add_fail("LearningRateScheduler", "Learning rate not decreasing")
            
            # Test linear decay
            lr_linear = scheduler.step_linear(5, 10)
            if 0 < lr_linear < 0.001:
                results.add_pass("LearningRateScheduler linear decay")
            else:
                results.add_fail("LearningRateScheduler linear", f"Unexpected lr: {lr_linear}")
            
            # Test cosine annealing
            lr_cosine = scheduler.step_cosine(0, 10)
            if 0 < lr_cosine <= 0.001:
                results.add_pass("LearningRateScheduler cosine annealing")
            else:
                results.add_fail("LearningRateScheduler cosine", f"Unexpected lr: {lr_cosine}")
        
        except Exception as e:
            results.add_fail("LearningRateScheduler", str(e))
    
    except Exception as e:
        results.add_fail("training_utils general", str(e))
    
    results.print_summary()
    return results.failed == 0


# ============================================================================
# INTEGRATION TEST
# ============================================================================

def test_integration():
    """Test integration of all components."""
    print("\n" + "="*70)
    print("INTEGRATION TEST: All Components Together")
    print("="*70)
    
    results = TestResults()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Test 1: Complete workflow
        # 1. Set seed
        set_all_seeds(42, verbose=False)
        
        # 2. Setup logging
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logging(output_dir=tmpdir, level='INFO', verbose=False)
            results.add_pass("Integration: Seeding + Logging")
            
            # 3. Create model
            config = {
                'dataset': {
                    'name': 'sleep-edf',
                    'n_features': 24,
                    'n_classes': 5,
                    'sequence_length': 10
                },
                'model': {
                    'lstm_units': 128,
                    'lstm_layers': 2,
                    'dropout': 0.3,
                    'dense_layers': [64, 32]
                }
            }
            model = SleepEDFModel(config, device=device)
            results.add_pass("Integration: Model creation")
            
            # 4. Training workflow
            x = torch.randn(32, 10, 24).to(device)
            y = torch.randint(0, 5, (32,))
            
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Simulate training
            for epoch in range(3):
                optimizer.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()
            
            results.add_pass("Integration: Training loop")
            
            # 5. Save results
            training_results = {
                'epochs': 3,
                'final_loss': float(loss.item()),
                'model_class': model.__class__.__name__
            }
            
            results_file = Path(tmpdir) / 'results.json'
            with open(results_file, 'w') as f:
                json.dump(training_results, f)
            
            results.add_pass("Integration: Results saving")
    
    except Exception as e:
        results.add_fail("Integration test", str(e))
    
    results.print_summary()
    return results.failed == 0


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all tests."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  TIER 1 COMPONENT TESTS - COMPREHENSIVE VALIDATION".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    all_passed = True
    
    # Run all tests
    all_passed &= test_seed_utils()
    all_passed &= test_logging_utils()
    all_passed &= test_base_model()
    all_passed &= test_sleep_edf_model()
    all_passed &= test_wesad_model()
    all_passed &= test_base_trainer()
    all_passed &= test_training_utils()
    all_passed &= test_integration()
    
    # Final summary
    print("\n")
    print("╔" + "="*68 + "╗")
    if all_passed:
        print("║" + "  ✅ ALL TESTS PASSED - TIER 1 READY FOR PRODUCTION".center(68) + "║")
    else:
        print("║" + "  ❌ SOME TESTS FAILED - REVIEW ERRORS ABOVE".center(68) + "║")
    print("╚" + "="*68 + "╝")
    print("\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())