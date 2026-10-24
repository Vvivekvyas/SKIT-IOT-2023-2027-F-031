"""
Unit tests for initial ML model groundwork (Week 6 - Nancy).

Verifies:
- FT-Transformer Tokenizer and Classifier forward pass and output shapes
- Focal Loss computation
- Deep Autoencoder reconstruction and anomaly score scaling
- HybridPredictor interface output contract and hybrid triage logic
"""

import pytest
import torch
import numpy as np
import pandas as pd

from ml.models.ft_transformer import FeatureTokenizer, FTTransformer, FocalLoss
from ml.autoencoder.autoencoder import DeepAutoencoder
from ml.inference.hybrid_predictor import HybridPredictor


class TestFTTransformer:
    """Verifies FT-Transformer architecture components and shapes."""

    def test_feature_tokenizer_shape(self):
        batch_size = 16
        num_features = 78
        embedding_dim = 32

        tokenizer = FeatureTokenizer(num_features=num_features, embedding_dim=embedding_dim)
        x = torch.randn(batch_size, num_features)
        tokens = tokenizer(x)

        # Expected shape: (batch_size, num_features + 1, embedding_dim)
        assert tokens.shape == (batch_size, num_features + 1, embedding_dim)

    def test_ft_transformer_forward_and_confidence(self):
        batch_size = 8
        num_features = 39
        num_classes = 15
        embedding_dim = 32

        model = FTTransformer(
            num_features=num_features,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            num_heads=2,
            num_layers=2,
        )
        x = torch.randn(batch_size, num_features)

        # Test forward logits
        logits = model(x)
        assert logits.shape == (batch_size, num_classes)

        # Test prediction helper
        preds, conf = model.predict_with_confidence(x)
        assert preds.shape == (batch_size,)
        assert conf.shape == (batch_size,)
        assert (conf >= 0.0).all() and (conf <= 1.0).all()

    def test_focal_loss_computation(self):
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(10, 5)
        targets = torch.randint(0, 5, (10,))
        loss = loss_fn(logits, targets)

        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert loss.item() >= 0.0


class TestDeepAutoencoder:
    """Verifies Deep Autoencoder reconstruction and anomaly scoring."""

    def test_autoencoder_reconstruction_shape(self):
        batch_size = 12
        input_dim = 78
        model = DeepAutoencoder(input_dim=input_dim, latent_dim=16, hidden_dims=[64, 32])
        x = torch.randn(batch_size, input_dim)

        recon = model(x)
        assert recon.shape == (batch_size, input_dim)

    def test_reconstruction_error_and_anomaly_score(self):
        batch_size = 10
        input_dim = 39
        model = DeepAutoencoder(input_dim=input_dim, latent_dim=8, hidden_dims=[32, 16])
        x = torch.randn(batch_size, input_dim)

        mse = model.compute_reconstruction_error(x)
        assert mse.shape == (batch_size,)
        assert (mse >= 0.0).all()

        scores = model.compute_anomaly_score(x, threshold=1.0)
        assert scores.shape == (batch_size,)
        assert (scores >= 0.0).all() and (scores <= 1.0).all()

    def test_threshold_calibration(self):
        input_dim = 20
        model = DeepAutoencoder(input_dim=input_dim, latent_dim=8)
        benign_val = torch.randn(50, input_dim)

        thresh = model.calibrate_threshold(benign_val, percentile=95.0)
        assert isinstance(thresh, float)
        assert thresh > 0.0


class TestHybridPredictor:
    """Verifies HybridPredictor inference output contract and triage logic."""

    @pytest.fixture
    def setup_hybrid(self):
        num_features = 20
        class_names = ["BENIGN", "DDoS", "PortScan"]

        classifier = FTTransformer(
            num_features=num_features,
            num_classes=len(class_names),
            embedding_dim=16,
            num_heads=2,
            num_layers=1,
        )
        autoencoder = DeepAutoencoder(
            input_dim=num_features,
            latent_dim=8,
            hidden_dims=[16],
        )
        autoencoder.anomaly_threshold = 0.5

        predictor = HybridPredictor(
            classifier=classifier,
            autoencoder=autoencoder,
            class_names=class_names,
            confidence_threshold=0.85,
            anomaly_flag_threshold=0.60,
        )
        return predictor, num_features

    def test_prediction_contract(self, setup_hybrid):
        predictor, num_features = setup_hybrid
        x = np.random.randn(5, num_features).astype(np.float32)

        results = predictor.predict(x)
        assert len(results) == 5

        for item in results:
            assert "prediction" in item
            assert "confidence" in item
            assert "anomaly_score" in item
            assert isinstance(item["prediction"], str)
            assert isinstance(item["confidence"], float)
            assert isinstance(item["anomaly_score"], float)
            assert 0.0 <= item["confidence"] <= 1.0
            assert 0.0 <= item["anomaly_score"] <= 1.0

    def test_single_flow_prediction(self, setup_hybrid):
        predictor, num_features = setup_hybrid
        single_flow = [0.5] * num_features

        res = predictor.predict_single(single_flow)
        assert isinstance(res, dict)
        assert "prediction" in res
        assert "confidence" in res
        assert "anomaly_score" in res
