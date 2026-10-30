"""
Loss functions for hybrid intrusion detection system training.

Hybrid ML-Based Intrusion Detection System
Role: Nancy (Machine Learning & ML Backend)
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss for addressing severe tabular class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights well-classified benign traffic and focuses optimization on rare attack signatures.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes focal loss between logits and integer class targets.

        Args:
            logits: Tensor of shape (batch_size, num_classes)
            targets: Tensor of shape (batch_size,) with ground-truth class indices
        Returns:
            Computed scalar or per-sample loss tensor
        """
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        targets_col = targets.view(-1, 1)
        target_log_probs = log_probs.gather(1, targets_col).squeeze(-1)
        target_probs = probs.gather(1, targets_col).squeeze(-1)

        focal_weight = torch.pow(1.0 - target_probs, self.gamma)
        loss = -focal_weight * target_log_probs

        if self.alpha is not None:
            alpha_tensor = self.alpha.to(logits.device)
            loss = loss * alpha_tensor[targets]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class AutoencoderReconstructionLoss(nn.Module):
    """
    Reconstruction loss combining Mean Squared Error (MSE) and directional cosine penalty.
    """

    def __init__(self, cosine_weight: float = 0.05):
        super().__init__()
        self.mse = nn.MSELoss()
        self.cosine_weight = cosine_weight

    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(reconstructed, original)
        if self.cosine_weight > 0.0:
            cosine_similarity = F.cosine_similarity(reconstructed, original, dim=-1).mean()
            return mse_loss + self.cosine_weight * (1.0 - cosine_similarity)
        return mse_loss
