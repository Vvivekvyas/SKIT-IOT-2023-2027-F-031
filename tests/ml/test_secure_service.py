"""
Unit tests for Secure Model-Service Interface.
Validates model integrity hashing, feature bounds sanitization, and zero-day threat triage.

Role: Nancy (Machine Learning & ML Backend)
"""
import json
import os
import tempfile
import numpy as np
import pytest
import torch

from ml.autoencoder.autoencoder import DeepAutoencoder
from ml.inference.hybrid_predictor import HybridPredictor
from ml.inference.secure_service import (
    FeatureSanitizer,
    ModelIntegrityVerifier,
    SecureModelInterface,
    ThreatTriageEngine,
)
from ml.models.ft_transformer import FTTransformer


def test_model_integrity_verification_and_manifest():
    """Verifies SHA-256 calculation, tampering detection, and manifest generation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_file = os.path.join(tmp_dir, "test_model.pt")
        manifest_file = os.path.join(tmp_dir, "test_manifest.json")
        with open(model_file, "wb") as f:
            f.write(b"dummy_model_binary_payload_for_verification")

        expected_hash = ModelIntegrityVerifier.compute_sha256(model_file)
        assert len(expected_hash) == 64
        assert ModelIntegrityVerifier.verify_checkpoint(model_file, expected_hash) is True

        manifest = ModelIntegrityVerifier.generate_manifest(model_file, manifest_file, version="v1.0.0")
        assert os.path.exists(manifest_file)
        assert manifest["sha256"] == expected_hash

        with pytest.raises(ValueError, match="integrity verification failed"):
            ModelIntegrityVerifier.verify_checkpoint(model_file, "invalid_corrupt_hash_000000000000")


def test_feature_sanitizer_dimension_and_bounds():
    """Verifies feature shape checking, NaN/Inf scrubbing, and clipping boundaries."""
    sanitizer = FeatureSanitizer(num_features=4, min_bound=-5.0, max_bound=5.0)
    with pytest.raises(ValueError, match="Feature shape mismatch"):
        sanitizer.sanitize(np.ones((2, 6)))

    dirty_input = np.array([[np.nan, np.inf, -np.inf, 100.0]], dtype=np.float32)
    cleaned = sanitizer.sanitize(dirty_input)
    assert not np.isnan(cleaned).any()
    assert not np.isinf(cleaned).any()
    assert cleaned[0, 0] == 0.0
    assert cleaned[0, 1] == 5.0
    assert cleaned[0, 2] == -5.0
    assert cleaned[0, 3] == 5.0


def test_threat_triage_decision_matrix_and_entropy():
    """Tests threat triage policy and Shannon entropy calculation across scenarios."""
    triage = ThreatTriageEngine(anomaly_threshold=0.70, confidence_threshold=0.85)

    probs = np.array([0.95, 0.03, 0.02])
    entropy = triage.compute_entropy(probs)
    assert 0.0 <= entropy <= 1.0

    benign_res = triage.triage(prediction="BENIGN", confidence=0.95, anomaly_score=0.10, probabilities=probs)
    assert benign_res["severity"] == "LOW"
    assert benign_res["action"] == "ALLOW"
    assert benign_res["is_alert"] is False
    assert benign_res["entropy"] == entropy

    attack_res = triage.triage(prediction="DDoS", confidence=0.92, anomaly_score=0.80)
    assert attack_res["severity"] == "CRITICAL"
    assert attack_res["action"] == "BLOCK_IMMEDIATE"
    assert attack_res["is_alert"] is True

    zero_day_res = triage.triage(prediction="BENIGN", confidence=0.90, anomaly_score=0.88)
    assert zero_day_res["severity"] == "ZERO_DAY_SUSPECT"
    assert zero_day_res["action"] == "QUARANTINE_FOR_ANALYSIS"
    assert zero_day_res["classification"] == "Suspected_Zero_Day"
    assert zero_day_res["is_alert"] is True


def test_secure_model_interface_end_to_end():
    """Verifies end-to-end execution of sanitized input through secure model inference."""
    torch.manual_seed(42)
    names = ["flow_dur", "tot_fwd_pkts", "tot_bwd_pkts"]
    classes = ["BENIGN", "PortScan"]

    clf = FTTransformer(num_features=3, num_classes=2, embedding_dim=8, num_heads=2, num_layers=1)
    ae = DeepAutoencoder(input_dim=3, latent_dim=2, hidden_dims=[4])
    predictor = HybridPredictor(clf, ae, classes, feature_names=names)

    interface = SecureModelInterface(predictor=predictor)
    raw_flows = np.array([[1.2, np.nan, 3.4], [0.5, 100.0, -20.0]], dtype=np.float32)

    results = interface.predict_secure(raw_flows)
    assert len(results) == 2
    for item in results:
        assert "classification" in item
        assert "severity" in item
        assert "action" in item
        assert "is_alert" in item
