#!/usr/bin/env python3
"""
Custom learning rate schedulers.

All DP/FL compatible - no gradient computations.
Supports serialization for checkpoint saving/loading.
"""

import math
from typing import List, Dict, Any


class CosineAnnealingWarmRestarts:
    """
    Cosine annealing with warm restarts.

    Decreases LR following cosine function with periodic restarts.
    Compatible with DP/FL and serializable.

    Reference: Loshchilov & Hutter (2016)
    https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_0: int = 10, T_mult: float = 2,
                 eta_min: float = 1e-6, last_epoch: int = -1):
        """
        Args:
            optimizer: Wrapped optimizer
            T_0: Number of iterations for first restart period (> 0)
            T_mult: Multiplier for period length after restart (> 0)
            eta_min: Minimum learning rate (>= 0)
            last_epoch: The index of last epoch
        """
        # Validation
        if T_0 <= 0:
            raise ValueError(f"T_0 must be > 0, got {T_0}")
        if T_mult <= 0:
            raise ValueError(f"T_mult must be > 0, got {T_mult}")
        if eta_min < 0:
            raise ValueError(f"eta_min must be >= 0, got {eta_min}")

        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.last_epoch = last_epoch

        # Initialize base_lrs
        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        self.base_lrs = [
            group['initial_lr'] for group in self.optimizer.param_groups
        ]

        # Track current period info
        self._current_period = 0
        self._period_start_epoch = 0
        self._T_i = T_0

        # Initial step
        self.step(last_epoch + 1)

    def step(self, epoch: int = None):
        """
        Step scheduler.

        Args:
            epoch: Current epoch (if None, increments from last_epoch)
        """
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch

        # Calculate which period we're in
        self._update_period(epoch)

        # Calculate position within current period
        t_cur = epoch - self._period_start_epoch

        # Update learning rates
        for param_group, base_lr in zip(
            self.optimizer.param_groups, self.base_lrs
        ):
            lr = self.eta_min + (base_lr - self.eta_min) * \
                 (1 + math.cos(math.pi * t_cur / self._T_i)) / 2
            param_group['lr'] = lr

    def _update_period(self, epoch: int) -> None:
        """Update current period based on epoch."""
        # Find which period epoch belongs to
        epoch_in_period = 0
        period = 0
        T_i = self.T_0

        while epoch_in_period + T_i <= epoch:
            epoch_in_period += T_i
            T_i = int(T_i * self.T_mult)
            period += 1

        self._current_period = period
        self._period_start_epoch = epoch_in_period
        self._T_i = T_i

    def get_last_lr(self) -> List[float]:
        """Return last learning rate."""
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self) -> Dict[str, Any]:
        """Return scheduler state for checkpointing."""
        return {
            'last_epoch': self.last_epoch,
            'current_period': self._current_period,
            'period_start_epoch': self._period_start_epoch,
            'T_i': self._T_i,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state from checkpoint."""
        self.last_epoch = state_dict.get('last_epoch', self.last_epoch)
        self._current_period = state_dict.get('current_period', 0)
        self._period_start_epoch = state_dict.get('period_start_epoch', 0)
        self._T_i = state_dict.get('T_i', self.T_0)


class WarmupCosineAnnealingScheduler:
    """
    Combines warmup phase with cosine annealing.

    Warmup: linear increase from eta_min to base_lr
    Annealing: cosine decay from base_lr to eta_min

    Compatible with DP/FL and serializable.
    """

    def __init__(self, optimizer, warmup_epochs: int = 5,
                 total_epochs: int = 100, eta_min: float = 1e-6,
                 last_epoch: int = -1):
        """
        Args:
            optimizer: Wrapped optimizer
            warmup_epochs: Number of warmup epochs (>= 0)
            total_epochs: Total training epochs (> warmup_epochs)
            eta_min: Minimum learning rate (>= 0)
            last_epoch: The index of last epoch
        """
        # Validation
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        if total_epochs <= warmup_epochs:
            raise ValueError(
                f"total_epochs ({total_epochs}) must be > "
                f"warmup_epochs ({warmup_epochs})"
            )
        if eta_min < 0:
            raise ValueError(f"eta_min must be >= 0, got {eta_min}")

        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.last_epoch = last_epoch

        # Initialize base_lrs
        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        self.base_lrs = [
            group['initial_lr'] for group in self.optimizer.param_groups
        ]

        # Initial step
        self.step(last_epoch + 1)

    def step(self, epoch: int = None):
        """
        Step scheduler.

        Args:
            epoch: Current epoch (if None, increments from last_epoch)
        """
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch

        for param_group, base_lr in zip(
            self.optimizer.param_groups, self.base_lrs
        ):
            if epoch < self.warmup_epochs:
                # Warmup: linear increase
                if self.warmup_epochs > 0:
                    lr = self.eta_min + (base_lr - self.eta_min) * \
                         (epoch / self.warmup_epochs)
                else:
                    lr = base_lr
            else:
                # Cosine annealing after warmup
                progress = (epoch - self.warmup_epochs) / \
                          (self.total_epochs - self.warmup_epochs)
                # Clamp progress to [0, 1] in case epoch > total_epochs
                progress = min(progress, 1.0)
                lr = self.eta_min + (base_lr - self.eta_min) * \
                     (1 + math.cos(math.pi * progress)) / 2

            param_group['lr'] = max(lr, self.eta_min)

    def get_last_lr(self) -> List[float]:
        """Return last learning rate."""
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self) -> Dict[str, Any]:
        """Return scheduler state for checkpointing."""
        return {
            'last_epoch': self.last_epoch,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state from checkpoint."""
        self.last_epoch = state_dict.get('last_epoch', self.last_epoch)


class ConstantLRScheduler:
    """
    No-op scheduler (constant LR).

    Useful for testing or when you want to disable scheduling.
    Compatible with DP/FL.
    """

    def __init__(self, optimizer, last_epoch: int = -1):
        """
        Args:
            optimizer: Wrapped optimizer
            last_epoch: The index of last epoch
        """
        self.optimizer = optimizer
        self.last_epoch = last_epoch

        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        self.base_lrs = [
            group['initial_lr'] for group in self.optimizer.param_groups
        ]

    def step(self, epoch: int = None):
        """Step scheduler (no-op)."""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

    def get_last_lr(self) -> List[float]:
        """Return last learning rate."""
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self) -> Dict[str, Any]:
        """Return scheduler state."""
        return {'last_epoch': self.last_epoch}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state."""
        self.last_epoch = state_dict.get('last_epoch', self.last_epoch)


# Utility function for creating scheduler
def create_scheduler(optimizer, scheduler_name: str = 'cosine',
                    **kwargs) -> Any:
    """
    Create scheduler by name.

    Args:
        optimizer: Wrapped optimizer
        scheduler_name: 'cosine', 'warmup_cosine', 'constant'
        **kwargs: Additional arguments for scheduler

    Returns:
        Scheduler instance
    """
    scheduler_name = scheduler_name.lower()

    if scheduler_name == 'cosine':
        return CosineAnnealingWarmRestarts(optimizer, **kwargs)
    elif scheduler_name == 'warmup_cosine':
        return WarmupCosineAnnealingScheduler(optimizer, **kwargs)
    elif scheduler_name == 'constant':
        return ConstantLRScheduler(optimizer, **kwargs)
    else:
        raise ValueError(
            f"Unknown scheduler: {scheduler_name}. "
            f"Choose from: 'cosine', 'warmup_cosine', 'constant'"
        )