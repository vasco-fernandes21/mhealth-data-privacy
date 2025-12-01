#!/usr/bin/env python3
"""Centralized logging configuration."""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Literal

_LOGGER = None


def setup_logging(
    output_dir: Optional[str] = None,
    level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO',
    name: str = 'mhealth-privacy',
    verbose: bool = False
) -> logging.Logger:
    global _LOGGER
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = output_path / f'log_{timestamp}.txt'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    _LOGGER = logger
    return logger


def get_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        if _LOGGER is None:
            setup_logging(level='INFO', verbose=False)
        logger = logging.getLogger(name)
    
    return logger