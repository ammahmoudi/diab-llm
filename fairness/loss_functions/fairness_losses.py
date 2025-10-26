"""
Fairness-Aware Loss Functions
============================

This module implements various fairness-aware loss functions that can be used
to train models with better fairness properties.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class FairnessAwareLoss(nn.Module):
    """Base class for fairness-aware loss functions."""
    
    def __init__(self, base_loss: nn.Module = None, fairness_weight: float = 1.0):
        """Initialize fairness-aware loss.
        
        Args:
            base_loss: Base loss function (e.g., MSELoss, CrossEntropyLoss)
            fairness_weight: Weight for the fairness penalty term
        """
        super().__init__()
        self.base_loss = base_loss or nn.MSELoss()
        self.fairness_weight = fairness_weight
    
    def forward(self, predictions, targets, group_labels):
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError


class DemographicParityLoss(FairnessAwareLoss):
    """Loss function that enforces demographic parity."""
    
    def __init__(self, base_loss: nn.Module = None, fairness_weight: float = 1.0):
        super().__init__(base_loss, fairness_weight)
    
    def forward(self, predictions: torch.Tensor, 
                targets: torch.Tensor, 
                group_labels: torch.Tensor) -> torch.Tensor:
        """Calculate loss with demographic parity constraint.
        
        Args:
            predictions: Model predictions [batch_size, ...]
            targets: True targets [batch_size, ...]
            group_labels: Group membership indicators [batch_size]
            
        Returns:
            Combined loss with fairness penalty
        """
        # Base loss
        base_loss_value = self.base_loss(predictions, targets)
        
        # Demographic parity penalty
        unique_groups = torch.unique(group_labels)
        if len(unique_groups) < 2:
            return base_loss_value
        
        group_means = []
        for group in unique_groups:
            group_mask = (group_labels == group)
            if group_mask.sum() > 0:
                group_predictions = predictions[group_mask]
                group_mean = torch.mean(group_predictions)
                group_means.append(group_mean)
        
        if len(group_means) >= 2:
            # Penalty is the variance of group means
            group_means_tensor = torch.stack(group_means)
            fairness_penalty = torch.var(group_means_tensor)
        else:
            fairness_penalty = torch.tensor(0.0, device=predictions.device)
        
        return base_loss_value + self.fairness_weight * fairness_penalty


class EqualizedOddsLoss(FairnessAwareLoss):
    """Loss function that enforces equalized odds."""
    
    def __init__(self, base_loss: nn.Module = None, fairness_weight: float = 1.0):
        super().__init__(base_loss, fairness_weight)
    
    def forward(self, predictions: torch.Tensor, 
                targets: torch.Tensor, 
                group_labels: torch.Tensor) -> torch.Tensor:
        """Calculate loss with equalized odds constraint.
        
        Args:
            predictions: Model predictions [batch_size, ...]
            targets: True targets [batch_size, ...]
            group_labels: Group membership indicators [batch_size]
            
        Returns:
            Combined loss with fairness penalty
        """
        # Base loss
        base_loss_value = self.base_loss(predictions, targets)
        
        # Convert to binary classification for equalized odds
        pred_binary = (predictions > 0.5).float()
        target_binary = (targets > torch.median(targets)).float()
        
        unique_groups = torch.unique(group_labels)
        if len(unique_groups) != 2:
            return base_loss_value
        
        tpr_diff = 0.0
        fpr_diff = 0.0
        
        tprs = []
        fprs = []
        
        for group in unique_groups:
            group_mask = (group_labels == group)
            if group_mask.sum() == 0:
                continue
                
            group_pred = pred_binary[group_mask]
            group_target = target_binary[group_mask]
            
            # True Positive Rate
            tp = torch.sum((group_target == 1) & (group_pred == 1)).float()
            fn = torch.sum((group_target == 1) & (group_pred == 0)).float()
            tpr = tp / (tp + fn + 1e-8)
            tprs.append(tpr)
            
            # False Positive Rate
            fp = torch.sum((group_target == 0) & (group_pred == 1)).float()
            tn = torch.sum((group_target == 0) & (group_pred == 0)).float()
            fpr = fp / (fp + tn + 1e-8)
            fprs.append(fpr)
        
        if len(tprs) == 2 and len(fprs) == 2:
            tpr_diff = torch.abs(tprs[0] - tprs[1])
            fpr_diff = torch.abs(fprs[0] - fprs[1])
            fairness_penalty = tpr_diff + fpr_diff
        else:
            fairness_penalty = torch.tensor(0.0, device=predictions.device)
        
        return base_loss_value + self.fairness_weight * fairness_penalty


class GroupRegularizedLoss(FairnessAwareLoss):
    """Loss function with group-wise regularization."""
    
    def __init__(self, base_loss: nn.Module = None, 
                 fairness_weight: float = 1.0,
                 regularization_type: str = 'mse_difference'):
        """Initialize group regularized loss.
        
        Args:
            base_loss: Base loss function
            fairness_weight: Weight for fairness penalty
            regularization_type: Type of regularization ('mse_difference', 'performance_variance')
        """
        super().__init__(base_loss, fairness_weight)
        self.regularization_type = regularization_type
    
    def forward(self, predictions: torch.Tensor, 
                targets: torch.Tensor, 
                group_labels: torch.Tensor) -> torch.Tensor:
        """Calculate loss with group regularization.
        
        Args:
            predictions: Model predictions [batch_size, ...]
            targets: True targets [batch_size, ...]
            group_labels: Group membership indicators [batch_size]
            
        Returns:
            Combined loss with fairness penalty
        """
        # Base loss
        base_loss_value = self.base_loss(predictions, targets)
        
        unique_groups = torch.unique(group_labels)
        if len(unique_groups) < 2:
            return base_loss_value
        
        if self.regularization_type == 'mse_difference':
            group_mses = []
            for group in unique_groups:
                group_mask = (group_labels == group)
                if group_mask.sum() > 0:
                    group_pred = predictions[group_mask]
                    group_target = targets[group_mask]
                    group_mse = F.mse_loss(group_pred, group_target)
                    group_mses.append(group_mse)
            
            if len(group_mses) >= 2:
                # Penalty is the difference between group MSEs
                group_mses_tensor = torch.stack(group_mses)
                fairness_penalty = torch.var(group_mses_tensor)
            else:
                fairness_penalty = torch.tensor(0.0, device=predictions.device)
                
        elif self.regularization_type == 'performance_variance':
            group_performances = []
            for group in unique_groups:
                group_mask = (group_labels == group)
                if group_mask.sum() > 0:
                    group_pred = predictions[group_mask]
                    group_target = targets[group_mask]
                    # Use negative MSE as performance (higher is better)
                    group_performance = -F.mse_loss(group_pred, group_target)
                    group_performances.append(group_performance)
            
            if len(group_performances) >= 2:
                group_performances_tensor = torch.stack(group_performances)
                fairness_penalty = torch.var(group_performances_tensor)
            else:
                fairness_penalty = torch.tensor(0.0, device=predictions.device)
        
        else:
            raise ValueError(f"Unknown regularization type: {self.regularization_type}")
        
        return base_loss_value + self.fairness_weight * fairness_penalty


class AdversarialFairnessLoss(nn.Module):
    """Adversarial fairness loss function."""
    
    def __init__(self, base_loss: nn.Module = None, 
                 adversarial_weight: float = 1.0):
        """Initialize adversarial fairness loss.
        
        Args:
            base_loss: Base loss function
            adversarial_weight: Weight for adversarial loss
        """
        super().__init__()
        self.base_loss = base_loss or nn.MSELoss()
        self.adversarial_weight = adversarial_weight
    
    def forward(self, predictions: torch.Tensor, 
                targets: torch.Tensor,
                group_predictions: torch.Tensor,
                group_labels: torch.Tensor) -> torch.Tensor:
        """Calculate adversarial fairness loss.
        
        Args:
            predictions: Model predictions [batch_size, ...]
            targets: True targets [batch_size, ...]
            group_predictions: Adversary's group predictions [batch_size, num_groups]
            group_labels: True group labels [batch_size]
            
        Returns:
            Combined loss for the main model
        """
        # Base prediction loss
        base_loss_value = self.base_loss(predictions, targets)
        
        # Adversarial loss (we want to fool the adversary)
        # Convert group_labels to one-hot if necessary
        if len(group_labels.shape) == 1:
            num_groups = len(torch.unique(group_labels))
            group_labels_onehot = F.one_hot(group_labels, num_groups).float()
        else:
            group_labels_onehot = group_labels
        
        # We want to minimize the adversary's ability to predict group membership
        # So we maximize the cross-entropy loss of the adversary
        adversarial_loss = -F.cross_entropy(group_predictions, group_labels_onehot.argmax(dim=1))
        
        return base_loss_value + self.adversarial_weight * adversarial_loss


class FairnessLossFactory:
    """Factory class for creating fairness-aware loss functions."""
    
    @staticmethod
    def create_loss(loss_type: str, 
                   base_loss: nn.Module = None,
                   fairness_weight: float = 1.0,
                   **kwargs) -> nn.Module:
        """Create a fairness-aware loss function.
        
        Args:
            loss_type: Type of fairness loss ('demographic_parity', 'equalized_odds', 
                      'group_regularized', 'adversarial')
            base_loss: Base loss function
            fairness_weight: Weight for fairness penalty
            **kwargs: Additional arguments for specific loss types
            
        Returns:
            Fairness-aware loss function
        """
        if loss_type == 'demographic_parity':
            return DemographicParityLoss(base_loss, fairness_weight)
        elif loss_type == 'equalized_odds':
            return EqualizedOddsLoss(base_loss, fairness_weight)
        elif loss_type == 'group_regularized':
            return GroupRegularizedLoss(base_loss, fairness_weight, **kwargs)
        elif loss_type == 'adversarial':
            return AdversarialFairnessLoss(base_loss, fairness_weight)
        else:
            raise ValueError(f"Unknown fairness loss type: {loss_type}")
    
    @staticmethod
    def get_available_losses() -> List[str]:
        """Get list of available fairness loss types."""
        return ['demographic_parity', 'equalized_odds', 'group_regularized', 'adversarial']


# Example usage and testing
if __name__ == "__main__":
    # Test fairness loss functions
    torch.manual_seed(42)
    
    batch_size = 32
    predictions = torch.randn(batch_size, 1)
    targets = torch.randn(batch_size, 1)
    group_labels = torch.randint(0, 2, (batch_size,))
    
    # Test different loss functions
    base_loss = nn.MSELoss()
    
    print("Testing Fairness Loss Functions")
    print("=" * 40)
    
    # Demographic Parity Loss
    dp_loss = DemographicParityLoss(base_loss, fairness_weight=0.5)
    dp_loss_value = dp_loss(predictions, targets, group_labels)
    print(f"Demographic Parity Loss: {dp_loss_value.item():.4f}")
    
    # Equalized Odds Loss
    eo_loss = EqualizedOddsLoss(base_loss, fairness_weight=0.5)
    eo_loss_value = eo_loss(predictions, targets, group_labels)
    print(f"Equalized Odds Loss: {eo_loss_value.item():.4f}")
    
    # Group Regularized Loss
    gr_loss = GroupRegularizedLoss(base_loss, fairness_weight=0.5)
    gr_loss_value = gr_loss(predictions, targets, group_labels)
    print(f"Group Regularized Loss: {gr_loss_value.item():.4f}")
    
    # Base loss for comparison
    base_loss_value = base_loss(predictions, targets)
    print(f"Base Loss (MSE): {base_loss_value.item():.4f}")
    
    print("\nFairness penalties successfully added to base loss!")