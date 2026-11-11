"""
Secure Model-Service Interface for Hybrid IDS Inference.
Provides cryptographic model verification, input feature sanitization,
adversarial boundary enforcement, and zero-day threat triage assessment.

Role: Nancy (Machine Learning & ML Backend)
"""
import hashlib
import json
import os
from typing import Any, Dict, List, Optional
import numpy as np

from ml.inference.hybrid_predictor import HybridPredictor


class ModelIntegrityVerifier:
    """Verifies cryptographic hashes of saved model checkpoints to prevent tampering."""

    @staticmethod
    def compute_sha256(filepath: str) -> str:
        """Computes the SHA-256 hash of a file on disk."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        return sha256.hexdigest()

    @classmethod
    def verify_checkpoint(cls, filepath: str, expected_hash: str) -> bool:
        """Verifies checkpoint SHA-256 against the expected hash."""
        actual_hash = cls.compute_sha256(filepath)
        if actual_hash.lower() != expected_hash.strip().lower():
            raise ValueError(
                f"Model integrity verification failed for {filepath}. Expected {expected_hash}, got {actual_hash}"
            )
        return True

    @classmethod
    def generate_manifest(cls, checkpoint_path: str, manifest_path: str, version: str = "v1.0.0") -> Dict[str, Any]:
        """Generates a cryptographic manifest JSON for an exported model checkpoint."""
        sha256_hash = cls.compute_sha256(checkpoint_path)
        file_size = os.path.getsize(checkpoint_path)
        manifest = {
            "model_version": version,
            "filename": os.path.basename(checkpoint_path),
            "sha256": sha256_hash,
            "size_bytes": file_size,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return manifest


class FeatureSanitizer:
    """Validates and sanitizes input feature matrices against adversarial out-of-bound values."""

    def __init__(
        self,
        num_features: int,
        feature_names: Optional[List[str]] = None,
        min_bound: float = -10.0,
        max_bound: float = 10.0,
    ):
        self.num_features = num_features
        self.feature_names = feature_names
        self.min_bound = min_bound
        self.max_bound = max_bound

    def sanitize(self, features: np.ndarray) -> np.ndarray:
        """Checks dimensions, replaces NaNs/Infs, and clips extreme outlier values."""
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.num_features:
            raise ValueError(f"Feature shape mismatch: expected {self.num_features} columns, got {arr.shape[1]}")
        arr = np.nan_to_num(arr, nan=0.0, posinf=self.max_bound, neginf=self.min_bound)
        return np.clip(arr, self.min_bound, self.max_bound)


class ThreatTriageEngine:
    """Calculates threat severity levels, uncertainty entropy, and zero-day alert classifications."""

    BENIGN_LABELS = {"BENIGN", "BENIGN_FINAL", "NORMAL"}

    def __init__(self, anomaly_threshold: float = 0.70, confidence_threshold: float = 0.85):
        self.anomaly_threshold = anomaly_threshold
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def compute_entropy(probabilities: np.ndarray) -> float:
        """Calculates normalized Shannon entropy to gauge prediction uncertainty."""
        p = np.clip(probabilities, 1e-12, 1.0)
        raw_entropy = -float(np.sum(p * np.log(p)))
        max_entropy = np.log(len(p)) if len(p) > 1 else 1.0
        return round(float(raw_entropy / max_entropy), 4)

    def triage(
        self,
        prediction: str,
        confidence: float,
        anomaly_score: float,
        probabilities: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Triages inference output into threat severity, zero-day flags, and action recommendations."""
        is_classified_benign = prediction.upper() in self.BENIGN_LABELS
        is_anomalous = anomaly_score >= self.anomaly_threshold
        is_high_conf = confidence >= self.confidence_threshold
        entropy = self.compute_entropy(probabilities) if probabilities is not None else 0.0

        if is_classified_benign and is_anomalous:
            severity = "ZERO_DAY_SUSPECT"
            action = "QUARANTINE_FOR_ANALYSIS"
            final_label = "Suspected_Zero_Day"
        elif not is_classified_benign and is_high_conf:
            severity = "CRITICAL" if anomaly_score >= 0.50 else "HIGH"
            action = "BLOCK_IMMEDIATE"
            final_label = prediction
        elif not is_classified_benign:
            severity = "MEDIUM"
            action = "INSPECT_PACKET_TRACE"
            final_label = prediction
        else:
            severity = "LOW"
            action = "ALLOW"
            final_label = prediction

        return {
            "classification": final_label,
            "raw_prediction": prediction,
            "confidence": round(float(confidence), 4),
            "anomaly_score": round(float(anomaly_score), 4),
            "entropy": entropy,
            "severity": severity,
            "action": action,
            "is_alert": severity in {"CRITICAL", "HIGH", "ZERO_DAY_SUSPECT", "MEDIUM"},
        }


class SecureModelInterface:
    """Hardened interface combining feature sanitization, hybrid inference, and threat triage."""

    def __init__(
        self,
        predictor: HybridPredictor,
        sanitizer: Optional[FeatureSanitizer] = None,
        triage_engine: Optional[ThreatTriageEngine] = None,
    ):
        self.predictor = predictor
        num_features = len(predictor.feature_names) if predictor.feature_names else predictor.classifier.num_features
        self.sanitizer = sanitizer or FeatureSanitizer(num_features=num_features, feature_names=predictor.feature_names)
        self.triage_engine = triage_engine or ThreatTriageEngine(
            anomaly_threshold=predictor.anomaly_flag_threshold,
            confidence_threshold=predictor.confidence_threshold,
        )

    def predict_secure(self, raw_features: np.ndarray) -> List[Dict[str, Any]]:
        """Executes sanitized input validation, model inference, and threat triage assessment."""
        clean_features = self.sanitizer.sanitize(raw_features)
        raw_results = self.predictor.predict(clean_features)
        triaged = []
        for res in raw_results:
            assessment = self.triage_engine.triage(
                prediction=res["prediction"],
                confidence=res["confidence"],
                anomaly_score=res["anomaly_score"],
            )
            triaged.append(assessment)
        return triaged
