"""
FT-Transformer (Feature Tokenizer Transformer) for Tabular Network Flow Data.

Hybrid ML-Based Intrusion Detection System
Role: Nancy (Machine Learning & ML Backend)
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureTokenizer(nn.Module):
    """
    Transforms continuous tabular numerical features into dense feature tokens.

    Each continuous feature x_j is projected to a d-dimensional vector:
        e_j = x_j * W_j + b_j
    A learnable [CLS] token is prepended to the sequence of feature tokens.
    """

    def __init__(self, num_features: int, embedding_dim: int = 64):
        super().__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim

        # Per-feature weight and bias: (num_features, embedding_dim)
        self.weights = nn.Parameter(torch.empty(num_features, embedding_dim))
        self.biases = nn.Parameter(torch.empty(num_features, embedding_dim))
        nn.init.kaiming_uniform_(self.weights, a=5**0.5)
        nn.init.zeros_(self.biases)

        # Learnable [CLS] token prepended to the sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, num_features)
        Returns:
            Tensor of shape (batch_size, num_features + 1, embedding_dim)
        """
        batch_size = x.size(0)
        # Reshape x to (batch_size, num_features, 1)
        x_expanded = x.unsqueeze(-1)
        # Linear projection per feature
        tokens = x_expanded * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        return torch.cat([cls_tokens, tokens], dim=1)


class FTTransformer(nn.Module):
    """
    Supervised Tabular Transformer Classifier for Network Traffic.

    Combines Feature Tokenizer with multi-head self-attention transformer blocks
    and a classification head on top of the [CLS] representation.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        ffn_factor: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.tokenizer = FeatureTokenizer(num_features, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * ffn_factor,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning unnormalized class logits.

        Args:
            x: Numerical flow features of shape (batch_size, num_features)
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # (batch_size, num_features + 1, embedding_dim)
        tokens = self.tokenizer(x)
        # Transformer encoder
        encoded = self.transformer(tokens)
        # Extract [CLS] token representation
        cls_rep = self.norm(encoded[:, 0, :])
        # Classification logits
        return self.head(cls_rep)

    @torch.no_grad()
    def predict_with_confidence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference helper returning predicted class index and confidence probability.

        Returns:
            Tuple of (pred_class_indices, confidence_probabilities)
        """
        self.eval()
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        confidence, preds = torch.max(probs, dim=-1)
        return preds, confidence


class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss for handling severe class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
