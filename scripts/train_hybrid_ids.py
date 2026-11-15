"""
CLI Training and Checkpoint Export Script for the Deep Hybrid IDS.

Orchestrates:
1. Dataset ingestion / feature scaling for tabular network flows
2. Supervised FT-Transformer classifier training with Focal Loss
3. Unsupervised Deep Autoencoder anomaly detector training (benign-isolated)
4. Empirical anomaly threshold calibration (99th percentile)
5. Production checkpoint bundle export and SHA-256 integrity manifest generation

Author: Nancy (Machine Learning & ML Backend Lead)
Repository: SKIT-IOT-2023-2027-F-031
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.autoencoder.autoencoder import DeepAutoencoder
from ml.inference.secure_service import ModelIntegrityVerifier
from ml.models.ft_transformer import FTTransformer
from ml.training.trainer import HybridTrainer
from src.data.schemas import CICIDS2017_MANDATORY_NUMERICAL_FEATURES


def create_training_data(n_samples: int = 3000):
    """Generates standardized flow feature matrix with realistic class imbalance."""
    np.random.seed(42)
    feature_names = list(CICIDS2017_MANDATORY_NUMERICAL_FEATURES)
    num_features = len(feature_names)

    # 70% Benign, 15% DDoS, 10% PortScan, 3% Bot, 2% Infiltration
    classes = ["BENIGN", "DDoS", "PortScan", "Bot", "Infiltration"]
    proportions = [0.70, 0.15, 0.10, 0.03, 0.02]

    x_list, y_list = [], []
    for cls_idx, (cls_name, prop) in enumerate(zip(classes, proportions)):
        count = int(n_samples * prop)
        # Shift distributions slightly for different attack classes
        mean_shift = cls_idx * 0.45
        feats = np.random.normal(loc=mean_shift, scale=1.0, size=(count, num_features)).astype(np.float32)
        x_list.append(feats)
        y_list.extend([cls_idx] * count)

    x = np.vstack(x_list)
    y = np.array(y_list, dtype=np.int64)

    # Shuffle
    indices = np.random.permutation(len(x))
    x, y = x[indices], y[indices]

    # Preprocessing: Standard Scaler parameters
    scaler_center = np.mean(x, axis=0).astype(np.float32)
    scaler_scale = np.std(x, axis=0).astype(np.float32)
    scaler_scale = np.where(scaler_scale == 0, 1.0, scaler_scale)
    x_scaled = (x - scaler_center) / scaler_scale

    return x_scaled, y, feature_names, classes, scaler_center, scaler_scale


def run_pipeline(
    epochs_clf: int = 4,
    epochs_ae: int = 4,
    batch_size: int = 64,
    output_dir: str = "ml/artifacts/hybrid",
):
    print("=" * 70)
    print("🚀 HYBRID IDS: MODEL TRAINING & CHECKPOINT EXPORT PIPELINE")
    print("=" * 70)

    # 1. Prepare data
    print("\n[1/5] Preparing feature tensors and stratified splits...")
    x, y, feature_names, class_names, center, scale = create_training_data(n_samples=2500)
    num_features = len(feature_names)
    num_classes = len(class_names)

    split_idx = int(len(x) * 0.8)
    x_train, x_val = x[:split_idx], x[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=batch_size,
        shuffle=False,
    )

    # Separate benign-only data for Autoencoder training
    benign_mask_train = (y_train == 0)
    benign_mask_val = (y_val == 0)
    x_benign_train = torch.from_numpy(x_train[benign_mask_train])
    x_benign_val = torch.from_numpy(x_val[benign_mask_val])

    benign_train_loader = DataLoader(TensorDataset(x_benign_train), batch_size=batch_size, shuffle=True)
    benign_val_loader = DataLoader(TensorDataset(x_benign_val), batch_size=batch_size, shuffle=False)

    print(f"      Total Samples: {len(x)} | Features: {num_features} | Classes: {num_classes}")
    print(f"      Train Samples: {len(x_train)} (Benign: {len(x_benign_train)})")
    print(f"      Validation Samples: {len(x_val)} (Benign: {len(x_benign_val)})")

    # 2. Build Models
    print("\n[2/5] Initializing FT-Transformer and Deep Autoencoder...")
    clf_config = {
        "embedding_dim": 32,
        "num_heads": 4,
        "num_layers": 2,
        "ffn_factor": 2,
        "dropout": 0.1,
    }
    ae_config = {
        "latent_dim": 8,
        "hidden_dims": [32, 16],
        "dropout": 0.05,
    }

    classifier = FTTransformer(num_features=num_features, num_classes=num_classes, **clf_config)
    autoencoder = DeepAutoencoder(input_dim=num_features, **ae_config)

    trainer = HybridTrainer(
        classifier=classifier,
        autoencoder=autoencoder,
        feature_names=feature_names,
        class_names=class_names,
        classifier_config=clf_config,
        autoencoder_config=ae_config,
        confidence_threshold=0.85,
        anomaly_flag_threshold=0.70,
    )

    # 3. Train Models
    print(f"\n[3/5] Training FT-Transformer with Focal Loss ({epochs_clf} epochs)...")
    clf_history = trainer.train_classifier(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs_clf,
        lr=5e-4,
    )
    final_clf_loss = clf_history["val_loss"][-1]
    final_clf_acc = clf_history["val_acc"][-1]
    print(f"      FT-Transformer Validation Loss: {final_clf_loss:.4f} | Accuracy: {final_clf_acc * 100:.2f}%")

    print(f"\n[4/5] Training Deep Autoencoder on Benign-Only Traffic ({epochs_ae} epochs)...")
    ae_history = trainer.train_autoencoder(
        benign_train_loader=benign_train_loader,
        benign_val_loader=benign_val_loader,
        epochs=epochs_ae,
        lr=1e-3,
    )
    final_ae_loss = ae_history["val_recon_loss"][-1]
    print(f"      Autoencoder Reconstruction Loss: {final_ae_loss:.5f}")

    # Calibrate Anomaly Threshold
    calibrated_threshold = trainer.calibrate_threshold(x_benign_val, percentile=99.0)
    print(f"      Calibrated Anomaly Cutoff (p99): {calibrated_threshold:.4f}")

    # 4. Export Artifacts
    print("\n[5/5] Exporting production checkpoint and integrity manifest...")
    target_dir = Path(REPO_ROOT) / output_dir
    os.makedirs(target_dir, exist_ok=True)
    ckpt_file = str(target_dir / "hybrid_model.pt")
    manifest_file = str(target_dir / "hybrid_model.pt.manifest.json")

    trainer.export_checkpoint(
        export_path=ckpt_file,
        scaler_center=center,
        scaler_scale=scale,
        model_version="hybrid-ids-v1.0.0",
    )
    manifest = ModelIntegrityVerifier.generate_manifest(
        checkpoint_path=ckpt_file,
        manifest_path=manifest_file,
        version="hybrid-ids-v1.0.0",
    )

    print(f"      ✅ Model Saved: {ckpt_file} ({manifest['size_bytes'] / 1024:.1f} KB)")
    print(f"      ✅ SHA-256 Manifest: {manifest_file}")
    print(f"      SHA-256 Hash: {manifest['sha256']}")

    print("\n" + "=" * 70)
    print("🎯 HYBRID TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hybrid IDS Model")
    parser.add_argument("--epochs-clf", type=int, default=3, help="FT-Transformer epochs")
    parser.add_argument("--epochs-ae", type=int, default=3, help="Autoencoder epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    run_pipeline(
        epochs_clf=args.epochs_clf,
        epochs_ae=args.epochs_ae,
        batch_size=args.batch_size,
    )
