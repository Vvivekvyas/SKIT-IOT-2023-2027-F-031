"""
Placeholder for Module 2 (Model Training & Evaluation).

Expected contract with the backend:
  1. Load preprocessed data (output of the Dataset Loading Pipeline).
  2. Train RF / SVM / Decision Tree / XGBoost / hybrid FT-Transformer+Autoencoder.
  3. Compare on accuracy / precision / recall / F1.
  4. Save the winning model with joblib to app/ml/artifacts/best_model.pkl
  5. Save the ordered feature column list to app/ml/artifacts/feature_columns.json

Once those two files exist, the API in app/ml/model_registry.py picks them
up automatically — no backend code changes needed.
"""
import json

import joblib
from sklearn.ensemble import RandomForestClassifier

FEATURE_COLUMNS = ["duration", "protocol_type", "src_bytes", "dst_bytes"]  # placeholder


def main() -> None:
    # TODO: replace with real preprocessed dataset from the data pipeline
    import numpy as np

    X_dummy = np.random.rand(100, len(FEATURE_COLUMNS))
    y_dummy = np.random.choice(["Normal", "DoS", "Probe"], size=100)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_dummy, y_dummy)

    joblib.dump(model, "app/ml/artifacts/best_model.pkl")
    with open("app/ml/artifacts/feature_columns.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f)

    print("Dummy model saved. Replace this script's data loading with the real pipeline.")


if __name__ == "__main__":
    main()
