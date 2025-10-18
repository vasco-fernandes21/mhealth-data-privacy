"""
mhealth_privacy - Privacy-Preserving Health Data Analysis

This package provides tools for analyzing health data with privacy-preserving techniques
including Differential Privacy (DP) and Federated Learning (FL).
"""

__version__ = "0.1.0"
__author__ = "Vasco"

# Make submodules easily accessible
from . import preprocessing
from . import device_utils

__all__ = ["preprocessing", "device_utils"]

