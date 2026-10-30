"""
Unit and Integration Tests for Hybrid IDS Training Engine and Loss Functions.
Validates Focal Loss, Autoencoder reconstruction training, threshold calibration,
and backend checkpoint export compatibility.

Role: Nancy (Machine Learning & ML Backend)
"""
import os
import tempfile
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from ml.autoencoder.autoencoder import DeepAutoencoder
from ml.models.ft_transformer import FTTransformer
from ml.training.losses import AutoencoderReconstructionLoss, FocalLoss
from ml.training.trainer import HybridTrainer


@pytest.fixture
def dummy_dataset():
    """Generates synthetic tabular flow records for multi-class classification."""
    torch.manual_seed(42)
    num_samples, num_features, num_classes = 64, 8, 3
    x = torch.randn(num_samples, num_features)
    y = torch.randint(0, num_classes, (num_samples,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)
    return loader, num_features, num_classes


@pytest.fixture
def dummy_trainer(dummy_dataset):
    """Instantiates a lightweight HybridTrainer for testing."""
    _, num_features, num_classes = dummy_dataset
    feature_names = [f"f_{i}" for i in range(num_features)]
    class_names = ["BENIGN", "DDoS", "PortScan"]
    clf_config = {"embedding_dim": 16, "num_heads": 2, "num_layers": 1}
    ae_config = {"latent_dim": 4, "hidden_dims": [8]}

    clf = FTTransformer(num_features=num_features, num_classes=num_classes, **clf_config)
    ae = DeepAutoencoder(input_dim=num_features, **ae_config)
    return HybridTrainer(
        classifier=clf,
        autoencoder=ae,
        feature_names=feature_names,
        class_names=class_names,
        classifier_config=clf_config,
        autoencoder_config=ae_config,
        device="cpu",
    )


def test_focal_loss_forward_and_gradients():
    """Tests Focal Loss forward pass calculation and gradient backpropagation."""
    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.randn(8, 4, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3, 1, 0, 2, 1], dtype=torch.long)
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0 and loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None and not torch.isnan(logits.grad).any()


def test_focal_loss_with_alpha_weights():
    """Tests Focal Loss weighting behavior with class-specific alpha weights."""
    alpha = torch.tensor([0.1, 1.0, 5.0])
    loss_fn = FocalLoss(gamma=1.5, alpha=alpha)
    logits = torch.zeros(3, 3)
    targets = torch.tensor([0, 1, 2], dtype=torch.long)
    loss_weighted = loss_fn(logits, targets)
    assert loss_weighted.item() > 0.0


def test_reconstruction_loss_computation():
    """Tests AutoencoderReconstructionLoss forward calculation and cosine penalty."""
    criterion = AutoencoderReconstructionLoss(cosine_weight=0.1)
    orig = torch.ones(4, 8)
    perfect_loss = criterion(orig, orig)
    assert abs(perfect_loss.item()) < 1e-5
    imperfect_loss = criterion(orig * 0.5, orig)
    assert imperfect_loss.item() > perfect_loss.item()


def test_train_classifier_epoch_loop(dummy_trainer, dummy_dataset):
    """Tests FT-Transformer training loop execution and validation history tracking."""
    loader, _, _ = dummy_dataset
    history = dummy_trainer.train_classifier(
        train_loader=loader,
        val_loader=loader,
        epochs=2,
        lr=1e-3,
    )
    assert "train_loss" in history and "val_loss" in history and "val_acc" in history
    assert len(history["train_loss"]) == 2
    assert all(loss >= 0.0 for loss in history["train_loss"])


def test_train_autoencoder_nominal_traffic(dummy_trainer, dummy_dataset):
    """Tests unsupervised Autoencoder training loop on nominal network flows."""
    loader, _, _ = dummy_dataset
    history = dummy_trainer.train_autoencoder(
        benign_train_loader=loader,
        benign_val_loader=loader,
        epochs=2,
        lr=1e-3,
    )
    assert "train_recon_loss" in history and "val_recon_loss" in history
    assert len(history["train_recon_loss"]) == 2


def test_calibrate_threshold(dummy_trainer):
    """Verifies empirical threshold calibration on benign validation flow records."""
    benign_flows = torch.randn(50, len(dummy_trainer.feature_names))
    threshold = dummy_trainer.calibrate_threshold(benign_flows, percentile=95.0)
    assert threshold > 0.0
    assert dummy_trainer.autoencoder.anomaly_threshold == threshold


def test_export_checkpoint_and_load_contract(dummy_trainer):
    """Verifies exported model bundle conforms strictly to backend prediction service schema."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "test_hybrid_checkpoint.pt")
        center = np.zeros(len(dummy_trainer.feature_names), dtype=np.float32)
        scale = np.ones(len(dummy_trainer.feature_names), dtype=np.float32)

        dummy_trainer.export_checkpoint(
            export_path=ckpt_path,
            scaler_center=center,
            scaler_scale=scale,
            model_version="test-v1",
        )
        assert os.path.exists(ckpt_path)
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        expected_keys = {
            "feature_names", "class_names", "classifier_config", "classifier_state",
            "autoencoder_config", "autoencoder_state", "anomaly_threshold",
            "confidence_threshold", "anomaly_flag_threshold", "scaler_center",
            "scaler_scale", "model_version",
        }
        assert expected_keys.issubset(loaded.keys())
        assert loaded["model_version"] == "test-v1"


def test_build_predictor_and_run_inference(dummy_trainer):
    """Verifies that HybridPredictor assembled from HybridTrainer produces valid predictions."""
    predictor = dummy_trainer.build_predictor()
    dummy_input = np.random.randn(2, len(dummy_trainer.feature_names)).astype(np.float32)
    predictions = predictor.predict(dummy_input)
    assert len(predictions) == 2
    for pred in predictions:
        assert "prediction" in pred and "confidence" in pred and "anomaly_score" in pred
        assert 0.0 <= pred["confidence"] <= 1.0
        assert 0.0 <= pred["anomaly_score"] <= 1.0
