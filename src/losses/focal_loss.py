#!/usr/bin/env python3
"""
Focal Loss for handling class imbalance.

Reference: Lin et al. (2017) - Focal Loss for Dense Object Detection
https://arxiv.org/abs/1708.02002

Replicable in DP/FL - no special operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Focuses on hard examples by down-weighting easy examples.
    Compatible with Differential Privacy and Federated Learning.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, 
                 reduction: str = 'mean'):
        """
        Initialize FocalLoss.
        
        Args:
            alpha: Weighting factor in range (0,1) to balance
                   positive vs negative examples or a list of weights
                   for each class. Default: 0.25
            gamma: Exponent of the modulating factor (1 - p_t)^gamma.
                   Default: 2.0
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Model outputs (logits), shape (N, C) or (N, C, ...)
            targets: Target labels, shape (N,) or (N, ...)
        
        Returns:
            Focal loss value
        """
        # Get probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get class probabilities
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Focal loss
        focal_loss = self.alpha * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """
    Weighted Focal Loss for multi-class with class weights.
    """
    
    def __init__(self, weights: torch.Tensor = None, gamma: float = 2.0,
                 reduction: str = 'mean'):
        """
        Args:
            weights: Tensor of shape (C,) with weight for each class
            gamma: Focusing parameter
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.weights = weights
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted focal loss."""
        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        focal_weight = (1 - p_t) ** self.gamma
        
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weights, 
                                  reduction='none')
        
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss