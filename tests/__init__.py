#!/usr/bin/env python3
"""
Test modules for mHealth Privacy package.

Test structure:
- test_tier1.py: Tests for Tier 1 components
- test_preprocessing.py: Tests for preprocessing (Tier 2+)
- test_models.py: Tests for models (Tier 2+)
- test_trainers.py: Tests for trainers (Tier 2+)
- test_privacy.py: Tests for privacy modules (Tier 2+)
- test_integration.py: Integration tests (Tier 2+)

To run tests:
    pytest tests/ -v
    pytest tests/ -v --cov=src --cov-report=html
    pytest tests/test_tier1.py -v  # Tier 1 only
"""

__all__ = []