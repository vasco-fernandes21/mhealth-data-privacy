#!/usr/bin/env python3
"""
Centralized logging configuration.

Provides:
- Consistent log formatting
- Multiple handlers (console, file)
- Experiment-specific logging

Usage:
    from src.utils.logging_utils import setup_logging, get_logger
    
    # Setup once at program start
    setup_logging(output_dir='./logs', level='INFO')
    
    # Get logger in any module
    logger = get_logger(__name__)
    logger.info("Training started")
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Literal


# Global logger instance
_LOGGER = None


def setup_logging(
    output_dir: Optional[str] = None,
    level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO',
    name: str = 'mhealth-privacy',
    verbose: bool = True
) -> logging.Logger:
    """
    Setup global logging configuration.
    
    Args:
        output_dir: Directory to save log files (optional)
        level: Logging level
        name: Logger name
        verbose: Print setup info
    
    Returns:
        Configured logger instance
    
    Example:
        logger = setup_logging(output_dir='./logs', level='DEBUG')
    """
    global _LOGGER
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if output_dir provided)
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = output_path / f'log_{timestamp}.txt'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        if verbose:
            logger.info(f"Logging to: {log_file}")
    
    _LOGGER = logger
    return logger


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get logger instance.
    
    If no logger has been setup, creates a basic one.
    
    Args:
        name: Module name (typically __name__)
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If no handlers, setup basic logging
    if not logger.handlers:
        if _LOGGER is None:
            setup_logging(level='INFO', verbose=False)
        logger = logging.getLogger(name)
    
    return logger


class ExperimentLogger:
    """Context manager for experiment-specific logging."""
    
    def __init__(self, 
                 experiment_name: str,
                 output_dir: str = './logs',
                 level: str = 'INFO'):
        """
        Args:
            experiment_name: Name of experiment
            output_dir: Directory to save logs
            level: Logging level
        
        Example:
            with ExperimentLogger('baseline_sleep_edf') as logger:
                logger.info("Starting training")
                # ... training code ...
                logger.info("Training completed")
        """
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.level = level
        self.logger = None
        self.log_file = None
    
    def __enter__(self) -> logging.Logger:
        # Create experiment directory
        exp_dir = Path(self.output_dir) / self.experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(getattr(logging, self.level))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = exp_dir / f'experiment_{timestamp}.log'
        
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Experiment '{self.experiment_name}' started")
        self.logger.info(f"Log file: {self.log_file}")
        
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.logger.error(f"Experiment failed with {exc_type.__name__}: {exc_val}")
        else:
            self.logger.info(f"Experiment completed successfully")


def log_config(config: dict, logger: logging.Logger) -> None:
    """
    Log configuration dictionary nicely.
    
    Args:
        config: Configuration dict
        logger: Logger instance
    """
    logger.info("Configuration:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for k, v in value.items():
                logger.info(f"    {k}: {v}")
        else:
            logger.info(f"  {key}: {value}")


def log_metrics(metrics: dict, logger: logging.Logger, step: Optional[int] = None) -> None:
    """
    Log metrics dictionary nicely.
    
    Args:
        metrics: Metrics dict
        logger: Logger instance
        step: Training step (optional)
    """
    prefix = f"[Step {step}]" if step is not None else ""
    logger.info(f"Metrics {prefix}:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    # Test script
    print("Testing logging utilities...\n")
    
    # Test 1: Basic setup
    print("Test 1 - Basic setup:")
    logger = setup_logging(level='DEBUG')
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    
    # Test 2: Get logger
    print("\nTest 2 - Get logger:")
    logger2 = get_logger("test.module")
    logger2.info("Message from module logger")
    
    # Test 3: Log config
    print("\nTest 3 - Log config:")
    config = {
        'dataset': 'sleep-edf',
        'model': {'type': 'lstm', 'hidden_size': 128},
        'training': {'epochs': 100, 'batch_size': 32}
    }
    log_config(config, logger)
    
    # Test 4: Log metrics
    print("\nTest 4 - Log metrics:")
    metrics = {'accuracy': 0.9234, 'f1_score': 0.8912, 'loss': 0.2341}
    log_metrics(metrics, logger, step=10)
    
    # Test 5: Experiment logger
    print("\nTest 5 - Experiment logger:")
    with ExperimentLogger('test_experiment') as exp_logger:
        exp_logger.info("Experiment started")
        exp_logger.info("Training...")
        exp_logger.info("Experiment finished")
    
    print("\n✅ All tests completed!")