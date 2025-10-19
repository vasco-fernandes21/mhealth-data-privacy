#!/usr/bin/env python3
"""
Comprehensive test suite for Tier 2 components.

Tests:
1. YAML configs loading
2. Config merging
3. Model creation with configs
4. BaselineTrainer workflow
5. DPTrainer workflow
6. DP utilities
7. Data loading
8. Integration tests
"""

import sys
import os
from pathlib import Path
import tempfile
import json
import yaml

import torch
import torch.nn as nn
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed_utils import set_all_seeds
from src.utils.logging_utils import setup_logging, get_logger
from src.models.sleep_edf_model import SleepEDFModel
from src.models.wesad_model import WESADModel
from src.training.trainers.baseline_trainer import BaselineTrainer
from src.training.trainers.dp_trainer import DPTrainer
from src.privacy.dp_utils import DPConfig, check_dp_compatibility


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
# TEST 1: YAML Configs
# ============================================================================

def test_yaml_configs():
    """Test YAML configuration loading."""
    print("\n" + "="*70)
    print("TEST 1: YAML Configuration Loading")
    print("="*70)
    
    results = TestResults()
    
    try:
        # Test 1.1: Create temp configs
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            
            # Create training defaults
            training_defaults = {
                'training': {
                    'epochs': 100,
                    'batch_size': 64,
                    'learning_rate': 0.001,
                    'early_stopping': True,
                    'gradient_clipping': True,
                    'gradient_clip_norm': 1.0
                }
            }
            
            with open(config_dir / 'training_defaults.yaml', 'w') as f:
                yaml.dump(training_defaults, f)
            
            results.add_pass("Create training_defaults.yaml")
            
            # Create privacy defaults
            privacy_defaults = {
                'differential_privacy': {
                    'enabled': False,
                    'target_epsilon': 8.0,
                    'target_delta': 1e-5,
                    'max_grad_norm': 1.0
                }
            }
            
            with open(config_dir / 'privacy_defaults.yaml', 'w') as f:
                yaml.dump(privacy_defaults, f)
            
            results.add_pass("Create privacy_defaults.yaml")
            
            # Create Sleep-EDF config
            sleep_edf_cfg = {
                'dataset': {
                    'name': 'sleep-edf',
                    'n_features': 24,
                    'n_classes': 5,
                    'class_names': ['W', 'N1', 'N2', 'N3', 'R']
                },
                'model': {
                    'architecture': 'lstm',
                    'lstm_units': 128,
                    'lstm_layers': 2,
                    'dropout': 0.3,
                    'dense_layers': [64, 32]
                }
            }
            
            with open(config_dir / 'sleep_edf.yaml', 'w') as f:
                yaml.dump(sleep_edf_cfg, f)
            
            results.add_pass("Create sleep_edf.yaml")
            
            # Create WESAD config
            wesad_cfg = {
                'dataset': {
                    'name': 'wesad',
                    'n_channels': 16,
                    'n_classes': 2,
                    'class_names': ['non-stress', 'stress']
                },
                'model': {
                    'architecture': 'lstm',
                    'lstm_units': 64,
                    'lstm_layers': 2,
                    'dropout': 0.3,
                    'dense_layers': [64, 32]
                }
            }
            
            with open(config_dir / 'wesad.yaml', 'w') as f:
                yaml.dump(wesad_cfg, f)
            
            results.add_pass("Create wesad.yaml")
            
            # Test 1.2: Load configs
            try:
                with open(config_dir / 'training_defaults.yaml') as f:
                    train_cfg = yaml.safe_load(f)
                
                if train_cfg['training']['epochs'] == 100:
                    results.add_pass("Load training_defaults.yaml")
                else:
                    results.add_fail("training_defaults.yaml", "Wrong values")
            except Exception as e:
                results.add_fail("Load training_defaults.yaml", str(e))
            
            # Test 1.3: Load Sleep-EDF config
            try:
                with open(config_dir / 'sleep_edf.yaml') as f:
                    cfg = yaml.safe_load(f)
                
                if cfg['dataset']['name'] == 'sleep-edf':
                    results.add_pass("Load sleep_edf.yaml")
                else:
                    results.add_fail("sleep_edf.yaml", "Wrong dataset name")
            except Exception as e:
                results.add_fail("Load sleep_edf.yaml", str(e))
            
            # Test 1.4: Load WESAD config
            try:
                with open(config_dir / 'wesad.yaml') as f:
                    cfg = yaml.safe_load(f)
                
                if cfg['dataset']['n_channels'] == 16:
                    results.add_pass("Load wesad.yaml")
                else:
                    results.add_fail("wesad.yaml", "Wrong n_channels")
            except Exception as e:
                results.add_fail("Load wesad.yaml", str(e))
    
    except Exception as e:
        results.add_fail("YAML configs general", str(e))
    
    results.print_summary()
    return results.failed == 0


# ============================================================================
# TEST 2: Config Merging
# ============================================================================

def test_config_merging():
    """Test configuration merging."""
    print("\n" + "="*70)
    print("TEST 2: Configuration Merging")
    print("="*70)
    
    results = TestResults()
    
    try:
        # Test 2.1: Simple merge
        cfg1 = {'a': 1, 'b': {'c': 2}}
        cfg2 = {'d': 3, 'b': {'e': 4}}
        
        def merge_configs(*configs):
            merged = {}
            for cfg in configs:
                for k, v in cfg.items():
                    if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                        merged[k] = {**merged[k], **v}
                    else:
                        merged[k] = v
            return merged
        
        merged = merge_configs(cfg1, cfg2)
        
        if merged['a'] == 1 and merged['d'] == 3 and 'c' in merged['b']:
            results.add_pass("Simple config merge")
        else:
            results.add_fail("Config merge", "Merge failed")
        
        # Test 2.2: Override values
        default_cfg = {'training': {'epochs': 100, 'batch_size': 64}}
        custom_cfg = {'training': {'epochs': 50}}
        
        merged = merge_configs(default_cfg, custom_cfg)
        
        if merged['training']['epochs'] == 50 and merged['training']['batch_size'] == 64:
            results.add_pass("Config override")
        else:
            results.add_fail("Config override", "Values not overridden correctly")
        
        # Test 2.3: Three-way merge
        cfg_a = {'model': {'type': 'lstm'}}
        cfg_b = {'training': {'lr': 0.001}}
        cfg_c = {'privacy': {'enabled': True}}
        
        merged = merge_configs(cfg_a, cfg_b, cfg_c)
        
        if 'model' in merged and 'training' in merged and 'privacy' in merged:
            results.add_pass("Three-way config merge")
        else:
            results.add_fail("Three-way merge", "Not all configs merged")
    
    except Exception as e:
        results.add_fail("Config merging general", str(e))
    
    results.print_summary()
    return results.failed == 0


# ============================================================================
# TEST 3: Model Creation with Config
# ============================================================================

def test_model_creation_with_config():
    """Test creating models from config."""
    print("\n" + "="*70)
    print("TEST 3: Model Creation with Configuration")
    print("="*70)
    
    results = TestResults()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Test 3.1: Sleep-EDF model from config
        sleep_edf_config = {
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
        
        try:
            model = SleepEDFModel(sleep_edf_config, device=device)
            results.add_pass("Create Sleep-EDF model from config")
        except Exception as e:
            results.add_fail("Sleep-EDF model creation", str(e))
        
        # Test 3.2: WESAD model from config
        wesad_config = {
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
        
        try:
            model = WESADModel(wesad_config, device=device)
            results.add_pass("Create WESAD model from config")
        except Exception as e:
            results.add_fail("WESAD model creation", str(e))
        
        # Test 3.3: Model forward pass with config
        try:
            model = SleepEDFModel(sleep_edf_config, device=device)
            x = torch.randn(32, 10, 24).to(device)
            y = model(x)
            
            if y.shape == (32, 5):
                results.add_pass("Sleep-EDF forward pass with config")
            else:
                results.add_fail("Sleep-EDF forward", f"Wrong shape: {y.shape}")
        except Exception as e:
            results.add_fail("Sleep-EDF forward pass", str(e))
    
    except Exception as e:
        results.add_fail("Model creation general", str(e))
    
    results.print_summary()
    return results.failed == 0


# ============================================================================
# TEST 4: BaselineTrainer
# ============================================================================

def test_baseline_trainer():
    """Test BaselineTrainer."""
    print("\n" + "="*70)
    print("TEST 4: BaselineTrainer")
    print("="*70)
    
    results = TestResults()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        set_all_seeds(42, verbose=False)
        
        # Test 4.1: Create trainer
        config = {
            'dataset': {
                'name': 'sleep-edf',
                'n_features': 24,
                'n_classes': 5
            },
            'model': {
                'lstm_units': 128,
                'lstm_layers': 2,
                'dropout': 0.3,
                'dense_layers': [64, 32]
            },
            'training': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'early_stopping': True,
                'gradient_clipping': True,
                'gradient_clip_norm': 1.0,
                'use_class_weights': False,
                'label_smoothing': 0.0,
                'weight_decay': 1e-4,
                'num_workers': 0,
                'optimizer': 'adam',
                'loss': 'cross_entropy'
            }
        }
        
        model = SleepEDFModel(config, device=device)
        trainer = BaselineTrainer(model, config, device=device)
        
        results.add_pass("Create BaselineTrainer")
        
        # Test 4.2: Setup optimizer and loss
        try:
            trainer.setup_optimizer_and_loss()
            
            if trainer.optimizer is not None and trainer.criterion is not None:
                results.add_pass("Setup optimizer and loss")
            else:
                results.add_fail("Optimizer/Loss setup", "None values")
        except Exception as e:
            results.add_fail("Setup optimizer/loss", str(e))
        
        # Test 4.3: Training epoch
        try:
            from torch.utils.data import TensorDataset, DataLoader
            
            X_train = np.random.randn(100, 10, 24).astype(np.float32)
            y_train = np.random.randint(0, 5, 100)
            
            train_dataset = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long)
            )
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            loss, acc = trainer.train_epoch(train_loader)
            
            if isinstance(loss, float) and isinstance(acc, float):
                results.add_pass("Train epoch")
            else:
                results.add_fail("Train epoch", "Invalid return types")
        except Exception as e:
            results.add_fail("Train epoch", str(e))
        
        # Test 4.4: Validation
        try:
            X_val = np.random.randn(50, 10, 24).astype(np.float32)
            y_val = np.random.randint(0, 5, 50)
            
            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long)
            )
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            val_loss, val_acc = trainer.validate(val_loader)
            
            if isinstance(val_loss, float) and isinstance(val_acc, float):
                results.add_pass("Validation")
            else:
                results.add_fail("Validation", "Invalid return types")
        except Exception as e:
            results.add_fail("Validation", str(e))
        
        # Test 4.5: Checkpoint save/load
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_path = Path(tmpdir) / 'checkpoint.pth'
                trainer.save_checkpoint(str(checkpoint_path))
                
                if checkpoint_path.exists():
                    results.add_pass("Save checkpoint")
                else:
                    results.add_fail("Save checkpoint", "File not created")
                
                # Try to load
                try:
                    trainer.load_checkpoint(str(checkpoint_path))
                    results.add_pass("Load checkpoint")
                except Exception as e:
                    results.add_fail("Load checkpoint", str(e))
        except Exception as e:
            results.add_fail("Checkpoint operations", str(e))
    
    except Exception as e:
        results.add_fail("BaselineTrainer general", str(e))
    
    results.print_summary()
    return results.failed == 0


# ============================================================================
# TEST 5: DPTrainer
# ============================================================================

def test_dp_trainer():
    """Test DPTrainer."""
    print("\n" + "="*70)
    print("TEST 5: DPTrainer")
    print("="*70)
    
    results = TestResults()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        set_all_seeds(42, verbose=False)
        
        # Test 5.1: Create DP trainer
        config = {
            'dataset': {
                'name': 'sleep-edf',
                'n_features': 24,
                'n_classes': 5
            },
            'model': {
                'lstm_units': 128,
                'lstm_layers': 2,
                'dropout': 0.3,
                'dense_layers': [64, 32]
            },
            'training': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'early_stopping': True,
                'gradient_clipping': True,
                'gradient_clip_norm': 1.0,
                'label_smoothing': 0.0,
                'weight_decay': 1e-4,
                'num_workers': 0,
                'optimizer': 'adam',
                'loss': 'cross_entropy'
            },
            'differential_privacy': {
                'enabled': True,
                'target_epsilon': 8.0,
                'target_delta': 1e-5,
                'max_grad_norm': 1.0,
                'noise_multiplier': 0.9
            }
        }
        
        model = SleepEDFModel(config, device=device)
        trainer = DPTrainer(model, config, device=device)
        
        results.add_pass("Create DPTrainer")
        
        # Test 5.2: DP config
        if trainer.dp_config.enabled:
            results.add_pass("DP config enabled")
        else:
            results.add_fail("DP config", "Not enabled")
        
        # Test 5.3: Setup optimizer
        try:
            trainer.setup_optimizer_and_loss()
            results.add_pass("Setup optimizer (DP)")
        except Exception as e:
            results.add_fail("Setup optimizer DP", str(e))
        
        # Test 5.4: Privacy budget history
        if hasattr(trainer, 'privacy_budget_history'):
            results.add_pass("Privacy budget history attribute")
        else:
            results.add_fail("Privacy history", "Missing attribute")
    
    except Exception as e:
        results.add_fail("DPTrainer general", str(e))
    
    results.print_summary()
    return results.failed == 0


# ============================================================================
# TEST 6: DP Utilities
# ============================================================================

def test_dp_utilities():
    """Test DP utilities."""
    print("\n" + "="*70)
    print("TEST 6: DP Utilities")
    print("="*70)
    
    results = TestResults()
    
    try:
        # Test 6.1: DPConfig
        config = {
            'differential_privacy': {
                'enabled': True,
                'target_epsilon': 8.0,
                'target_delta': 1e-5,
                'max_grad_norm': 1.0,
                'noise_multiplier': 0.9
            }
        }
        
        dp_config = DPConfig(config)
        
        if dp_config.enabled and dp_config.target_epsilon == 8.0:
            results.add_pass("DPConfig creation and values")
        else:
            results.add_fail("DPConfig", "Wrong values")
        
        # Test 6.2: DPConfig repr
        try:
            repr_str = repr(dp_config)
            if 'epsilon' in repr_str.lower():
                results.add_pass("DPConfig repr")
            else:
                results.add_fail("DPConfig repr", "Missing epsilon info")
        except Exception as e:
            results.add_fail("DPConfig repr", str(e))
        
        # Test 6.3: DP compatibility check
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        try:
            is_compatible, incompatible = check_dp_compatibility(model)
            
            if is_compatible:
                results.add_pass("DP compatibility check (compatible model)")
            else:
                results.add_fail("DP compatibility", "Should be compatible")
        except Exception as e:
            results.add_fail("DP compatibility check", str(e))
        
        # Test 6.4: DP compatibility with BatchNorm (should fail)
        model_with_bn = nn.Sequential(
            nn.Linear(10, 20),
            nn.BatchNorm1d(20),
            nn.Linear(20, 5)
        )
        
        try:
            is_compatible, incompatible = check_dp_compatibility(model_with_bn)
            
            if not is_compatible and len(incompatible) > 0:
                results.add_pass("DP compatibility check (detects BatchNorm)")
            else:
                results.add_fail("DP compatibility", "Should detect BatchNorm as incompatible")
        except Exception as e:
            results.add_fail("DP compatibility with BN", str(e))
    
    except Exception as e:
        results.add_fail("DP utilities general", str(e))
    
    results.print_summary()
    return results.failed == 0


# ============================================================================
# TEST 7: Integration Test
# ============================================================================

def test_integration():
    """Test integration of Tier 2 components."""
    print("\n" + "="*70)
    print("TEST 7: Integration Test - Full Workflow")
    print("="*70)
    
    results = TestResults()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            
            # Step 1: Create configs
            training_defaults = {
                'training': {
                    'epochs': 2,  # Short for testing
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'early_stopping': True,
                    'early_stopping_patience': 1,
                    'gradient_clipping': True,
                    'gradient_clip_norm': 1.0,
                    'label_smoothing': 0.0,
                    'weight_decay': 1e-4,
                    'num_workers': 0,
                    'optimizer': 'adam',
                    'loss': 'cross_entropy'
                }
            }
            
            sleep_edf_cfg = {
                'dataset': {
                    'name': 'sleep-edf',
                    'n_features': 24,
                    'n_classes': 5,
                    'sequence_length': 10,
                    'class_names': ['W', 'N1', 'N2', 'N3', 'R']
                },
                'model': {
                    'lstm_units': 64,  # Smaller for speed
                    'lstm_layers': 1,
                    'dropout': 0.3,
                    'dense_layers': [32]
                }
            }
            
            with open(config_dir / 'training_defaults.yaml', 'w') as f:
                yaml.dump(training_defaults, f)
            
            with open(config_dir / 'sleep_edf.yaml', 'w') as f:
                yaml.dump(sleep_edf_cfg, f)
            
            # Step 2: Load and merge configs
            with open(config_dir / 'training_defaults.yaml') as f:
                train_cfg = yaml.safe_load(f)
            
            with open(config_dir / 'sleep_edf.yaml') as f:
                dataset_cfg = yaml.safe_load(f)
            
            config = {**train_cfg, **dataset_cfg}
            results.add_pass("Integration: Config loading and merging")
            
            # Step 3: Create model
            model = SleepEDFModel(config, device=device)
            results.add_pass("Integration: Model creation")
            
            # Step 4: Create trainer
            trainer = BaselineTrainer(model, config, device=device)
            results.add_pass("Integration: Trainer creation")
            
            # Step 5: Create data
            from torch.utils.data import TensorDataset, DataLoader
            
            X_train = np.random.randn(64, 10, 24).astype(np.float32)
            y_train = np.random.randint(0, 5, 64)
            X_val = np.random.randn(32, 10, 24).astype(np.float32)
            y_val = np.random.randint(0, 5, 32)
            
            train_dataset = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long)
            )
            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            results.add_pass("Integration: Data loading")
            
            # Step 6: Training loop (minimal)
            try:
                results_dict = trainer.fit(
                    train_loader,
                    val_loader,
                    epochs=2,
                    patience=1,
                    output_dir=str(Path(tmpdir) / 'checkpoints')
                )
                
                if 'total_epochs' in results_dict and results_dict['total_epochs'] > 0:
                    results.add_pass("Integration: Training completed")
                else:
                    results.add_fail("Integration: Training", "Missing results")
            except Exception as e:
                results.add_fail("Integration: Training", str(e))
            
            # Step 7: Save and load results
            try:
                output_file = Path(tmpdir) / 'results.json'
                trainer.save_results(results_dict, str(Path(tmpdir)))
                
                if output_file.exists():
                    results.add_pass("Integration: Results saving")
                else:
                    results.add_fail("Integration: Save", "File not created")
            except Exception as e:
                results.add_fail("Integration: Save results", str(e))
    
    except Exception as e:
        results.add_fail("Integration test general", str(e))
    
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
    print("║" + "  TIER 2 COMPONENT TESTS - COMPREHENSIVE VALIDATION".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    all_passed = True
    
    # Run all tests
    all_passed &= test_yaml_configs()
    all_passed &= test_config_merging()
    all_passed &= test_model_creation_with_config()
    all_passed &= test_baseline_trainer()
    all_passed &= test_dp_trainer()
    all_passed &= test_dp_utilities()
    all_passed &= test_integration()
    
    # Final summary
    print("\n")
    print("╔" + "="*68 + "╗")
    if all_passed:
        print("║" + "  ✅ ALL TESTS PASSED - TIER 2 READY FOR PRODUCTION".center(68) + "║")
    else:
        print("║" + "  ❌ SOME TESTS FAILED - REVIEW ERRORS ABOVE".center(68) + "║")
    print("╚" + "="*68 + "╝")
    print("\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())