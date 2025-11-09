#!/usr/bin/env python3
"""
Centralized random seeding for reproducibility.

Handles:
- Python random
- NumPy random
- PyTorch random (CPU, CUDA, MPS)
- CUDA backend options (deterministic, benchmark)

Usage:
    from src.utils.seed_utils import set_all_seeds, set_deterministic
    
    set_all_seeds(seed=42)
    set_deterministic(device='cuda')  # Optional: force determinism
"""

import os
import random
import numpy as np
import torch
from typing import Literal
import warnings


def set_all_seeds(seed: int = 42, verbose: bool = True) -> None:
    """
    Set all random seeds for complete reproducibility.
    
    Args:
        seed: Random seed value (default: 42)
        verbose: Print confirmation (default: True)
    
    Notes:
        - Sets seeds for: Python, NumPy, PyTorch (CPU, CUDA, MPS)
        - Does NOT set deterministic mode (see set_deterministic)
        - Safe to call multiple times
    """
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # PyTorch MPS (Apple Silicon)
    if hasattr(torch, 'mps') and torch.mps.is_available():
        torch.mps.manual_seed(seed)
    
    # Also set environment variable for some libraries
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if verbose:
        print(f"✅ All random seeds set to {seed}")


def set_deterministic(device: Literal['cuda', 'cpu', 'mps', 'auto'] = 'auto',
                      verbose: bool = True) -> None:
    """
    Force deterministic behavior (for maximum reproducibility).
    
    ⚠️ WARNING: Can significantly slow down training!
    - CUDA: ~10-15% slower
    - CPU: Minimal impact
    
    Args:
        device: Device type ('cuda', 'cpu', 'mps', or 'auto' to detect)
        verbose: Print confirmation
    
    Notes:
        - Only affects CUDA backend
        - CPU and MPS are deterministic by default
        - Some operations still may not be deterministic
    """
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    if device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if verbose:
            print(f"✅ CUDA deterministic mode enabled (slower but reproducible)")
    
    elif device == 'mps':
        # MPS doesn't have deterministic/benchmark options
        if verbose:
            print(f"ℹ️  MPS backend is deterministic by default")
    
    elif device == 'cpu':
        if verbose:
            print(f"ℹ️  CPU backend is deterministic by default")


def set_reproducible(seed: int = 42, device: str = 'auto', 
                     verbose: bool = True) -> None:
    """
    Convenience function: Set both seeds AND deterministic mode.
    
    Args:
        seed: Random seed value
        device: Device for deterministic mode
        verbose: Print confirmation
    
    Example:
        set_reproducible(seed=42, device='cuda')
    """
    set_all_seeds(seed, verbose=verbose)
    set_deterministic(device, verbose=verbose)


def set_benchmark_mode(enable: bool = True, verbose: bool = True) -> None:
    """
    Enable/disable CUDA benchmark mode for performance optimization.
    
    Use benchmark=True for:
    - Fixed architecture & input sizes
    - Multiple epochs with same shapes
    - When speed > reproducibility
    
    Use benchmark=False for:
    - Variable input sizes
    - First experiments (need reproducibility)
    
    Args:
        enable: Enable (True) or disable (False) benchmark
        verbose: Print confirmation
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = enable
        mode = "enabled" if enable else "disabled"
        if verbose:
            print(f"✅ CUDA benchmark mode {mode}")
    else:
        if verbose:
            print(f"ℹ️  CUDA not available - benchmark mode has no effect")


class RandomState:
    """Context manager for temporary seed changes."""
    
    def __init__(self, seed: int):
        """
        Args:
            seed: Temporary seed value
        
        Example:
            with RandomState(42):
                # Code here uses seed 42
                x = torch.randn(10)
            # After context, random state is restored
        """
        self.seed = seed
        self.python_state = None
        self.np_state = None
        self.torch_state = None
        self.cuda_state = None
    
    def __enter__(self):
        # Save current states
        self.python_state = random.getstate()
        self.np_state = np.random.get_state()
        self.torch_state = torch.get_rng_state()
        if torch.cuda.is_available():
            self.cuda_state = torch.cuda.get_rng_state_all()
        
        # Set new seed
        set_all_seeds(self.seed, verbose=False)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore states
        random.setstate(self.python_state)
        np.random.set_state(self.np_state)
        torch.set_rng_state(self.torch_state)
        if torch.cuda.is_available() and self.cuda_state is not None:
            torch.cuda.set_rng_state_all(self.cuda_state)


if __name__ == "__main__":
    # Test script
    print("Testing seed utilities...\n")
    
    # Test 1: Set seeds
    set_all_seeds(42)
    x1 = torch.randn(5)
    
    set_all_seeds(42)
    x2 = torch.randn(5)
    
    print(f"Test 1 - Reproducibility:")
    print(f"  x1: {x1}")
    print(f"  x2: {x2}")
    print(f"  Equal: {torch.allclose(x1, x2)}\n")
    
    # Test 2: Deterministic mode
    print(f"Test 2 - Deterministic mode:")
    set_deterministic('cuda')
    
    # Test 3: Benchmark mode
    print(f"\nTest 3 - Benchmark mode:")
    set_benchmark_mode(True)
    
    # Test 4: Context manager
    print(f"\nTest 4 - Random state context manager:")
    set_all_seeds(42, verbose=False)
    y1 = torch.randn(3)
    print(f"  Before context: {y1}")
    
    with RandomState(123):
        y_temp = torch.randn(3)
        print(f"  Inside context (seed=123): {y_temp}")
    
    y2 = torch.randn(3)
    print(f"  After context: {y2}")
    print(f"  y1 == y2: {torch.allclose(y1, y2)}")
    
    print("\n✅ All tests completed!")