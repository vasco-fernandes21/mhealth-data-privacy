#!/usr/bin/env python3
"""
Custom learning rate schedulers.

All DP/FL compatible - no gradient computations.
"""

import math


class CosineAnnealingWarmRestarts:
    """
    Cosine annealing with warm restarts.
    
    Decreases LR following the curve of cosine function with periodic restarts.
    Compatible with DP/FL.
    
    Reference: Loshchilov & Hutter (2016)
    https://arxiv.org/abs/1608.03983
    """
    
    def __init__(self, optimizer, T_0: int = 10, T_mult: int = 2, 
                 eta_min: float = 1e-6, last_epoch: int = -1):
        """
        Args:
            optimizer: Wrapped optimizer
            T_0: Number of iterations for first restart period
            T_mult: Used to calculate the number of iterations in the i-th period
            eta_min: Minimum learning rate
            last_epoch: The index of last epoch
        """
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.T_i = T_0
        self.last_epoch = last_epoch
        
        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        
        self.base_lrs = [group['initial_lr'] for group in self.optimizer.param_groups]
    
    def step(self, epoch: int = None):
        """Step scheduler."""
        if epoch is None:
            epoch = self.last_epoch + 1
        
        self.last_epoch = epoch
        self.T_cur = epoch % self.T_i
        
        # Check if we need to restart
        if self.T_cur == 0 and epoch > 0:
            self.T_i = int(self.T_i * self.T_mult)
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = self.eta_min + (base_lr - self.eta_min) * \
                              (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
    
    def get_last_lr(self):
        """Return last learning rate."""
        return [group['lr'] for group in self.optimizer.param_groups]


class WarmupCosineAnnealingScheduler:
    """
    Combines warmup phase with cosine annealing.
    
    Warmup: linear increase from eta_min to base_lr
    Annealing: cosine decay from base_lr to eta_min
    """
    
    def __init__(self, optimizer, warmup_epochs: int = 5, total_epochs: int = 100,
                 eta_min: float = 1e-6, last_epoch: int = -1):
        """
        Args:
            optimizer: Wrapped optimizer
            warmup_epochs: Number of warmup epochs
            total_epochs: Total training epochs
            eta_min: Minimum learning rate
            last_epoch: The index of last epoch
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        
        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        
        self.base_lrs = [group['initial_lr'] for group in self.optimizer.param_groups]
    
    def step(self, epoch: int = None):
        """Step scheduler."""
        if epoch is None:
            epoch = self.last_epoch + 1
        
        self.last_epoch = epoch
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if epoch < self.warmup_epochs:
                # Warmup: linear increase
                lr = self.eta_min + (base_lr - self.eta_min) * \
                    (epoch / self.warmup_epochs)
            else:
                # Cosine annealing
                progress = (epoch - self.warmup_epochs) / \
                          (self.total_epochs - self.warmup_epochs)
                lr = self.eta_min + (base_lr - self.eta_min) * \
                    (1 + math.cos(math.pi * progress)) / 2
            
            param_group['lr'] = lr
    
    def get_last_lr(self):
        """Return last learning rate."""
        return [group['lr'] for group in self.optimizer.param_groups]