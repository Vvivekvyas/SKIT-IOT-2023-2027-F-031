"""
Training and Orchestration Engine for Hybrid IDS Models.
Integrates FT-Transformer supervised classification and Deep Autoencoder anomaly detection.

Hybrid ML-Based Intrusion Detection System
Role: Nancy (Machine Learning & ML Backend)
"""
from typing import Any, Dict, List, Optional
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.autoencoder.autoencoder import DeepAutoencoder
from ml.inference.hybrid_predictor import HybridPredictor
from ml.models.ft_transformer import FTTransformer
from ml.training.losses import AutoencoderReconstructionLoss, FocalLoss


class HybridTrainer:
    """Orchestrates training for supervised FT-Transformer and unsupervised Autoencoder."""

    def __init__(
        self,
        classifier: FTTransformer,
        autoencoder: DeepAutoencoder,
        feature_names: List[str],
        class_names: List[str],
        classifier_config: Optional[Dict[str, Any]] = None,
        autoencoder_config: Optional[Dict[str, Any]] = None,
        confidence_threshold: float = 0.85,
        anomaly_flag_threshold: float = 0.70,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.classifier = classifier.to(self.device)
        self.autoencoder = autoencoder.to(self.device)
        self.feature_names = feature_names
        self.class_names = class_names
        self.classifier_config = classifier_config or {}
        self.autoencoder_config = autoencoder_config or {}
        self.confidence_threshold = confidence_threshold
        self.anomaly_flag_threshold = anomaly_flag_threshold

    def train_classifier(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, List[float]]:
        """Trains the FT-Transformer classifier using Focal Loss and AdamW."""
        criterion = FocalLoss(gamma=gamma, alpha=class_weights)
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=lr, weight_decay=weight_decay)
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(epochs):
            self.classifier.train()
            total_loss = 0.0
            total_samples = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).long()
                optimizer.zero_grad()
                logits = self.classifier(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_x.size(0)
                total_samples += batch_x.size(0)

            history["train_loss"].append(round(total_loss / max(total_samples, 1), 5))
            if val_loader is not None:
                val_loss, val_acc = self.evaluate_classifier(val_loader, criterion)
                history["val_loss"].append(round(val_loss, 5))
                history["val_acc"].append(round(val_acc, 4))
        return history

    @torch.no_grad()
    def evaluate_classifier(
        self,
        loader: DataLoader,
        criterion: Optional[nn.Module] = None,
    ) -> tuple[float, float]:
        """Evaluates classifier loss and accuracy on a given data loader."""
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        self.classifier.eval()
        total_loss, correct, total_samples = 0.0, 0, 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device).float()
            batch_y = batch_y.to(self.device).long()
            logits = self.classifier(batch_x)
            loss = criterion(logits, batch_y)
            preds = torch.argmax(logits, dim=-1)
            total_loss += loss.item() * batch_x.size(0)
            correct += (preds == batch_y).sum().item()
            total_samples += batch_x.size(0)
        return total_loss / max(total_samples, 1), correct / max(total_samples, 1)

    def train_autoencoder(
        self,
        benign_train_loader: DataLoader,
        benign_val_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ) -> Dict[str, List[float]]:
        """Trains the Deep Autoencoder strictly on nominal benign network traffic."""
        criterion = AutoencoderReconstructionLoss(cosine_weight=0.05)
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr, weight_decay=weight_decay)
        history: Dict[str, List[float]] = {"train_recon_loss": [], "val_recon_loss": []}

        for epoch in range(epochs):
            self.autoencoder.train()
            total_loss = 0.0
            total_samples = 0
            for batch in benign_train_loader:
                batch_x = batch[0] if isinstance(batch, (list, tuple)) else batch
                batch_x = batch_x.to(self.device).float()
                optimizer.zero_grad()
                reconstruction = self.autoencoder(batch_x)
                loss = criterion(reconstruction, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_x.size(0)
                total_samples += batch_x.size(0)

            history["train_recon_loss"].append(round(total_loss / max(total_samples, 1), 5))
            if benign_val_loader is not None:
                val_loss = self.evaluate_autoencoder(benign_val_loader, criterion)
                history["val_recon_loss"].append(round(val_loss, 5))
        return history

    @torch.no_grad()
    def evaluate_autoencoder(
        self,
        loader: DataLoader,
        criterion: Optional[nn.Module] = None,
    ) -> float:
        """Evaluates autoencoder reconstruction loss on nominal traffic."""
        if criterion is None:
            criterion = nn.MSELoss()
        self.autoencoder.eval()
        total_loss, total_samples = 0.0, 0
        for batch in loader:
            batch_x = batch[0] if isinstance(batch, (list, tuple)) else batch
            batch_x = batch_x.to(self.device).float()
            reconstruction = self.autoencoder(batch_x)
            loss = criterion(reconstruction, batch_x)
            total_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)
        return total_loss / max(total_samples, 1)

    def calibrate_threshold(
        self,
        benign_val_tensor: torch.Tensor,
        percentile: float = 99.0,
    ) -> float:
        """Calibrates anomaly threshold on benign validation flows."""
        tensor = benign_val_tensor.to(self.device).float()
        return self.autoencoder.calibrate_threshold(tensor, percentile=percentile)

    def export_checkpoint(
        self,
        export_path: str,
        scaler_center: Optional[np.ndarray] = None,
        scaler_scale: Optional[np.ndarray] = None,
        model_version: str = "v1.0.0",
    ) -> Dict[str, Any]:
        """Exports a production-ready checkpoint bundle conforming to backend predictor contract."""
        os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
        center_list = scaler_center.tolist() if isinstance(scaler_center, np.ndarray) else scaler_center
        scale_list = scaler_scale.tolist() if isinstance(scaler_scale, np.ndarray) else scaler_scale
        checkpoint_data = {
            "feature_names": self.feature_names,
            "class_names": self.class_names,
            "classifier_config": self.classifier_config,
            "classifier_state": self.classifier.state_dict(),
            "autoencoder_config": self.autoencoder_config,
            "autoencoder_state": self.autoencoder.state_dict(),
            "anomaly_threshold": float(self.autoencoder.anomaly_threshold),
            "confidence_threshold": float(self.confidence_threshold),
            "anomaly_flag_threshold": float(self.anomaly_flag_threshold),
            "scaler_center": center_list,
            "scaler_scale": scale_list,
            "model_version": model_version,
        }
        torch.save(checkpoint_data, export_path)
        return checkpoint_data

    def build_predictor(self) -> HybridPredictor:
        """Assembles and returns an active HybridPredictor instance."""
        return HybridPredictor(
            classifier=self.classifier,
            autoencoder=self.autoencoder,
            class_names=self.class_names,
            confidence_threshold=self.confidence_threshold,
            anomaly_flag_threshold=self.anomaly_flag_threshold,
            feature_names=self.feature_names,
        )
