#!/usr/bin/env python3
"""Random seeding utilities for reproducibility."""

import os
import random
import numpy as np
import torch
from typing import Literal


def set_all_seeds(seed: int = 42, verbose: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_deterministic(device: Literal['cuda', 'cpu', 'auto'] = 'auto') -> None:
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_reproducible(seed: int = 42, device: str = 'auto') -> None:
    set_all_seeds(seed, verbose=False)
    set_deterministic(device)