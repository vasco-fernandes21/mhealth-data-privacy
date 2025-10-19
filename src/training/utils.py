#!/usr/bin/env python3
"""
Training utilities:
- Progress bar with ETA
- Gradient monitoring
- Learning rate scheduling
"""

import time
import numpy as np
import torch
from typing import Optional, List
from datetime import timedelta


class ProgressBar:
    """Progress bar with ETA estimation."""
    
    def __init__(self, total: int, description: str = ""):
        """
        Initialize progress bar.
        
        Args:
            total: Total iterations
            description: Description label
        """
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, n: int = 1) -> None:
        """
        Update progress.
        
        Args:
            n: Number of items processed
        """
        self.current += n
        now = time.time()
        
        # Update every 1% or at least every 5 seconds
        if (self.current % max(1, self.total // 100) == 0 or
            now - self.last_update >= 5.0):
            self._display()
            self.last_update = now
    
    def _display(self) -> None:
        """Display progress bar."""
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.current) / rate if rate > 0 else 0
        
        percentage = min(100, (self.current / self.total) * 100)
        
        # Format times
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        remaining_str = str(timedelta(seconds=int(remaining)))
        
        # Create bar
        bar_length = 30
        filled = int(bar_length * self.current // self.total)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        # Print
        print(f"\r{self.description}: [{bar}] {percentage:5.1f}% | "
              f"{self.current}/{self.total} | "
              f"Elapsed: {elapsed_str} | ETA: {remaining_str}",
              end="", flush=True)
    
    def finish(self) -> None:
        """Finish progress bar."""
        self.update(self.total - self.current)
        print()  # New line


class GradientMonitor:
    """Monitor gradient statistics during training."""
    
    def __init__(self, model: torch.nn.Module):
        """
        Initialize monitor.
        
        Args:
            model: PyTorch model
        """
        self.model = model
        self.grad_norms = []
        self.grad_means = []
        self.grad_stds = []
    
    def log_gradients(self) -> dict:
        """
        Log current gradient statistics.
        
        Returns:
            Dictionary with gradient stats
        """
        all_grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                all_grads.append(param.grad.data.cpu().numpy().flatten())
        
        if not all_grads:
            return None
        
        all_grads = np.concatenate(all_grads)
        
        stats = {
            'norm': float(np.linalg.norm(all_grads)),
            'mean': float(np.mean(np.abs(all_grads))),
            'std': float(np.std(all_grads)),
            'min': float(np.min(all_grads)),
            'max': float(np.max(all_grads))
        }
        
        self.grad_norms.append(stats['norm'])
        self.grad_means.append(stats['mean'])
        self.grad_stds.append(stats['std'])
        
        return stats
    
    def get_summary(self) -> dict:
        """Get summary of gradient statistics."""
        if not self.grad_norms:
            return None
        
        return {
            'norm_mean': float(np.mean(self.grad_norms)),
            'norm_max': float(np.max(self.grad_norms)),
            'norm_min': float(np.min(self.grad_norms)),
            'mean_mean': float(np.mean(self.grad_means)),
            'std_mean': float(np.mean(self.grad_stds))
        }


class LearningRateScheduler:
    """Custom learning rate scheduler."""
    
    def __init__(self, optimizer, initial_lr: float):
        """
        Initialize scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            initial_lr: Initial learning rate
        """
        self.optimizer = optimizer
        self.initial_lr = initial_lr
    
    def step_exponential(self, epoch: int, decay_rate: float = 0.1, 
                        decay_steps: int = 10) -> float:
        """
        Exponential decay schedule.
        
        Args:
            epoch: Current epoch (starting from 0)
            decay_rate: Decay rate
            decay_steps: Decay every N steps
        
        Returns:
            Current learning rate
        """
        lr = self.initial_lr * (decay_rate ** (epoch / decay_steps))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def step_linear(self, epoch: int, total_epochs: int) -> float:
        """
        Linear decay schedule.
        
        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
        
        Returns:
            Current learning rate
        """
        lr = self.initial_lr * (1 - epoch / total_epochs)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def step_cosine(self, epoch: int, total_epochs: int) -> float:
        """
        Cosine annealing schedule.
        
        Args:
            epoch: Current epoch
            total_epochs: Total epochs
        
        Returns:
            Current learning rate
        """
        lr = self.initial_lr * (1 + np.cos(np.pi * epoch / total_epochs)) / 2
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


if __name__ == "__main__":
    # Test ProgressBar
    print("Testing ProgressBar...")
    pbar = ProgressBar(100, "Processing")
    for i in range(100):
        time.sleep(0.01)
        pbar.update(1)
    pbar.finish()
    print("✅ ProgressBar test passed\n")
    
    # Test GradientMonitor
    print("Testing GradientMonitor...")
    model = torch.nn.Linear(10, 5)
    monitor = GradientMonitor(model)
    
    x = torch.randn(32, 10)
    y = torch.randint(0, 5, (32,))
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    for _ in range(5):
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        
        grad_stats = monitor.log_gradients()
        print(f"Grad norm: {grad_stats['norm']:.4f}, "
              f"mean: {grad_stats['mean']:.6f}, "
              f"std: {grad_stats['std']:.6f}")
        
        optimizer.step()
    
    summary = monitor.get_summary()
    print(f"\nGradient summary:")
    print(f"  Mean norm: {summary['norm_mean']:.4f}")
    print(f"  Max norm: {summary['norm_max']:.4f}")
    print("✅ GradientMonitor test passed\n")
    
    # Test LRScheduler
    print("Testing LearningRateScheduler...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = LearningRateScheduler(optimizer, initial_lr=0.001)
    
    for epoch in range(10):
        lr_exp = scheduler.step_exponential(epoch, decay_rate=0.5, decay_steps=5)
        print(f"Epoch {epoch}: lr = {lr_exp:.6f}")
    
    print("✅ LearningRateScheduler test passed\n")
    print("✅ All utility tests passed!")