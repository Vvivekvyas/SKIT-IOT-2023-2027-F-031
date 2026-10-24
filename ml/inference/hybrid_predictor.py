"""
Hybrid ML Inference Predictor combining FT-Transformer and Autoencoder.

Hybrid ML-Based Intrusion Detection System
Role: Nancy (Machine Learning & ML Backend)
"""

from typing import Dict, List, Union, Any, Optional
import torch
import numpy as np
import pandas as pd

from ml.models.ft_transformer import FTTransformer
from ml.autoencoder.autoencoder import DeepAutoencoder


class HybridPredictor:
    """
    Combines supervised FT-Transformer classification with unsupervised
    Autoencoder anomaly reconstruction scoring.

    Expected Output Interface Contract:
    {
        "prediction": str,
        "confidence": float,
        "anomaly_score": float
    }
    """

    def __init__(
        self,
        classifier: FTTransformer,
        autoencoder: DeepAutoencoder,
        class_names: List[str],
        confidence_threshold: float = 0.85,
        anomaly_flag_threshold: float = 0.70,
        feature_names: Optional[List[str]] = None,
    ):
        self.classifier = classifier
        self.autoencoder = autoencoder
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.anomaly_flag_threshold = anomaly_flag_threshold
        self.feature_names = feature_names

        self.classifier.eval()
        self.autoencoder.eval()

    def _prepare_tensor(self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor]) -> torch.Tensor:
        """Converts input data into a float32 torch.Tensor."""
        if isinstance(x, pd.DataFrame):
            if self.feature_names is not None:
                # Align features to expected schema
                missing = [col for col in self.feature_names if col not in x.columns]
                if missing:
                    raise ValueError(f"Input DataFrame is missing required features: {missing[:5]}...")
                x = x[self.feature_names]
            tensor = torch.tensor(x.to_numpy(), dtype=torch.float32)
        elif isinstance(x, np.ndarray):
            tensor = torch.tensor(x, dtype=torch.float32)
        elif isinstance(x, torch.Tensor):
            tensor = x.float()
        else:
            raise TypeError(f"Unsupported input type: {type(x)}")

        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    @torch.no_grad()
    def predict(
        self,
        features: Union[np.ndarray, pd.DataFrame, torch.Tensor],
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid inference on a batch of network flow feature vectors.

        Args:
            features: 2D array, DataFrame, or Tensor of scaled network features.
        Returns:
            List of prediction dictionaries conforming to interface specification:
            [
                {
                    "prediction": "DDoS",
                    "confidence": 0.96,
                    "anomaly_score": 0.12
                },
                ...
            ]
        """
        x = self._prepare_tensor(features)

        # 1. Supervised Classification via FT-Transformer
        logits = self.classifier(x)
        probs = torch.softmax(logits, dim=-1)
        confidences, pred_indices = torch.max(probs, dim=-1)

        # 2. Reconstruction Anomaly Scoring via Deep Autoencoder
        anomaly_scores = self.autoencoder.compute_anomaly_score(x)

        results = []
        for i in range(x.size(0)):
            idx = int(pred_indices[i].item())
            raw_pred_name = self.class_names[idx] if idx < len(self.class_names) else f"Class_{idx}"
            conf = round(float(confidences[i].item()), 4)
            anomaly_sc = round(float(anomaly_scores[i].item()), 4)

            # Hybrid Decision Triage:
            # If classifier predicts BENIGN with high confidence, BUT the Autoencoder
            # flags an unusually high reconstruction anomaly, flag as Suspected_Anomaly
            final_prediction = raw_pred_name
            is_benign = raw_pred_name.upper() in {"BENIGN", "BENIGN_FINAL", "NORMAL"}
            if is_benign and anomaly_sc >= self.anomaly_flag_threshold:
                final_prediction = "Suspected_Anomaly"

            results.append({
                "prediction": final_prediction,
                "confidence": conf,
                "anomaly_score": anomaly_sc,
            })

        return results

    def predict_single(
        self,
        flow_vector: Union[np.ndarray, pd.Series, List[float]],
    ) -> Dict[str, Any]:
        """Convenience method for single-flow inference."""
        if isinstance(flow_vector, list):
            flow_vector = np.array(flow_vector, dtype=np.float32)
        elif isinstance(flow_vector, pd.Series):
            flow_vector = flow_vector.to_numpy(dtype=np.float32)

        batch_res = self.predict(flow_vector)
        return batch_res[0]
