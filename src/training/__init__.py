#!/usr/bin/env python3
"""
Training modules for model training.

Submodules:
- base_trainer: Abstract base class for all trainers
- utils: Training utilities (ProgressBar, GradientMonitor, etc)
- trainers/: Dataset and privacy-specific trainers (Tier 2+)

Usage:
    from src.training import BaseTrainer, ProgressBar, GradientMonitor
    from src.training.trainers import BaselineTrainer  # Tier 2+
    
    # Create trainer
    trainer = BaselineTrainer(model, config, device='cuda')
    
    # Train
    results = trainer.fit(
        train_loader, val_loader,
        epochs=100,
        patience=8,
        output_dir='./checkpoints'
    )
    
    # Save results
    trainer.save_results(results, './results')
"""

from .base_trainer import BaseTrainer

__all__ = [
    # base_trainer
    'BaseTrainer',
]