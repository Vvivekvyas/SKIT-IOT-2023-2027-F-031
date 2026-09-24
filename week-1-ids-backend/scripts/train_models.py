"""
Placeholder for Module 2 (Model Training & Evaluation).

Expected contract with the backend:
  1. Load preprocessed data — now wired to read app/data/processed/train.csv
     and test.csv, produced by scripts/run_data_pipeline.py (the Dataset
     Loading Pipeline task). Falls back to dummy random data if the
     pipeline hasn't been run yet, so this script is always runnable.
  2. Train RF / SVM / Decision Tree / XGBoost / hybrid FT-Transformer+Autoencoder.
  3. Compare on accuracy / precision / recall / F1.
  4. Save the winning model with joblib to app/ml/artifacts/best_model.pkl
  5. feature_columns.json is already produced by the data pipeline — this
     script should not need to touch it, just make sure the model was
     trained on those exact columns in that exact order.

Once best_model.pkl + feature_columns.json exist, the API in
app/ml/model_registry.py picks them up automatically — no backend code
changes needed.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

TRAIN_PATH = Path("app/data/processed/train.csv")
FEATURE_COLUMNS_PATH = Path("app/ml/artifacts/feature_columns.json")


def load_training_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    if TRAIN_PATH.exists() and FEATURE_COLUMNS_PATH.exists():
        df = pd.read_csv(TRAIN_PATH)
        with open(FEATURE_COLUMNS_PATH) as f:
            feature_columns = json.load(f)
        X = df[feature_columns].to_numpy()
        y = df["label"].to_numpy()
        print(f"Loaded real preprocessed data: {len(df)} rows, {len(feature_columns)} features.")
        return X, y, feature_columns

    print(
        "No processed data found at app/data/processed/train.csv — "
        "run `python scripts/run_data_pipeline.py` first. "
        "Falling back to dummy data so this script stays runnable."
    )
    feature_columns = ["duration", "protocol_type", "src_bytes", "dst_bytes"]
    X = np.random.rand(100, len(feature_columns))
    y = np.random.choice(["Normal", "DoS", "Probe"], size=100)
    return X, y, feature_columns


def main() -> None:
    X, y, feature_columns = load_training_data()

    # TODO (Module 2): train + compare RF, SVM, Decision Tree, XGBoost,
    # and the hybrid FT-Transformer+Autoencoder here; this is just RF as
    # a working baseline so the pipeline is testable end-to-end today.
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    Path("app/ml/artifacts").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, "app/ml/artifacts/best_model.pkl")

    # feature_columns.json is normally already written by the data pipeline;
    # only (re)write it here in the dummy-data fallback case.
    if not FEATURE_COLUMNS_PATH.exists():
        with open(FEATURE_COLUMNS_PATH, "w") as f:
            json.dump(feature_columns, f)

    print("Model saved to app/ml/artifacts/best_model.pkl")


if __name__ == "__main__":
    main()