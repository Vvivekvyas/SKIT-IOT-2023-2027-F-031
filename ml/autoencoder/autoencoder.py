"""
Deep Autoencoder for Reconstruction-Based Anomaly Scoring.

Hybrid ML-Based Intrusion Detection System
Role: Nancy (Machine Learning & ML Backend)
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import numpy as np


class DeepAutoencoder(nn.Module):
    """
    Symmetric Deep Autoencoder for Network Traffic Anomaly Detection.

    Trained strictly on nominal (benign) network traffic.
    Calculates reconstruction error to flag anomalous, unseen, or novel flow patterns.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dims: List[int] = None,
        dropout: float = 0.05,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.anomaly_threshold = 1.0

        # Build Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.LayerNorm(h_dim))
            encoder_layers.append(nn.GELU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build Decoder (symmetric)
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.LayerNorm(h_dim))
            decoder_layers.append(nn.GELU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor of shape (batch_size, input_dim)
        Returns:
            Reconstructed feature tensor of shape (batch_size, input_dim)
        """
        z = self.encoder(x)
        return self.decoder(z)

    @torch.no_grad()
    def compute_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes per-sample Mean Squared Error (MSE) reconstruction error.

        Args:
            x: Feature tensor of shape (batch_size, input_dim)
        Returns:
            Tensor of shape (batch_size,) containing per-sample reconstruction MSE.
        """
        self.eval()
        x_recon = self.forward(x)
        # MSE per sample across features
        mse = torch.mean((x - x_recon) ** 2, dim=-1)
        return mse

    @torch.no_grad()
    def compute_anomaly_score(
        self,
        x: torch.Tensor,
        threshold: float = None,
    ) -> torch.Tensor:
        """
        Computes normalized anomaly score scaled between [0.0, 1.0].
        Scores exceeding 1.0 are mapped smoothly via hyperbolic tangent.

        Args:
            x: Feature tensor of shape (batch_size, input_dim)
            threshold: Calibrated anomaly threshold (default: self.anomaly_threshold)
        Returns:
            Tensor of shape (batch_size,) with values in [0.0, 1.0].
        """
        if threshold is None:
            threshold = self.anomaly_threshold

        mse = self.compute_reconstruction_error(x)
        # Scaled anomaly score
        normalized_score = torch.tanh(mse / max(threshold, 1e-6))
        return normalized_score

    def calibrate_threshold(
        self,
        benign_val_tensor: torch.Tensor,
        percentile: float = 99.0,
    ) -> float:
        """
        Calibrates the anomaly threshold using nominal benign validation data.

        Args:
            benign_val_tensor: Tensor of benign flows
            percentile: Percentile cutoff (e.g. 95.0 or 99.0)
        Returns:
            Calibrated threshold value.
        """
        errors = self.compute_reconstruction_error(benign_val_tensor).cpu().numpy()
        self.anomaly_threshold = float(np.percentile(errors, percentile))
        return self.anomaly_threshold
