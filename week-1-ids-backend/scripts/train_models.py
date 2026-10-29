"""
Module 2: Model Training, Evaluation & Comparative Benchmarking.

Expected contract with the backend:
  1. Load preprocessed data — reads app/data/processed/train.csv
     and test.csv, produced by scripts/run_data_pipeline.py. Falls back
     to dummy random data if the pipeline hasn't been run yet.
  2. Train and compare models:
     - Random Forest (Bagging Ensemble)
     - Decision Tree (CART)
     - Gradient Boosting Classifier
     - Linear SVM (LinearSVC)
     - Logistic Regression
     - Gaussian Naive Bayes
     - (XGBoost if installed)
  3. Compare on accuracy / precision / recall / F1-macro.
  4. Save the winning model with joblib to app/ml/artifacts/best_model.pkl.
  5. feature_columns.json is produced by the data pipeline — this script ensures
     alignment with the serialized contract.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

TRAIN_PATH = Path("app/data/processed/train.csv")
TEST_PATH = Path("app/data/processed/test.csv")
FEATURE_COLUMNS_PATH = Path("app/ml/artifacts/feature_columns.json")


def load_training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Loads training and test data partitions."""
    if TRAIN_PATH.exists() and FEATURE_COLUMNS_PATH.exists():
        train_df = pd.read_csv(TRAIN_PATH)
        with open(FEATURE_COLUMNS_PATH) as f:
            schema = json.load(f)
        feature_columns = schema["feature_columns"] if isinstance(schema, dict) else schema
        X_train = train_df[feature_columns].to_numpy()
        y_train = train_df["label"].to_numpy()

        if TEST_PATH.exists():
            test_df = pd.read_csv(TEST_PATH)
            X_test = test_df[feature_columns].to_numpy()
            y_test = test_df["label"].to_numpy()
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )

        print(f"Loaded preprocessed data: {len(train_df)} train rows, {len(feature_columns)} features.")
        return X_train, y_train, X_test, y_test, feature_columns

    print(
        "No processed data found at app/data/processed/train.csv — "
        "Falling back to synthetic data so this script stays runnable."
    )
    feature_columns = ["duration", "protocol_type", "src_bytes", "dst_bytes"]
    X = np.random.rand(200, len(feature_columns))
    y = np.random.choice(["Normal", "DoS", "Probe"], size=200)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, y_train, X_test, y_test, feature_columns


def main() -> None:
    X_train, y_train, X_test, y_test, feature_columns = load_training_data()

    candidate_models = {
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=15, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42),
        "Linear SVM": LinearSVC(random_state=42, dual="auto", max_iter=1000),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Gaussian NB": GaussianNB(),
    }

    try:
        import xgboost as xgb
        candidate_models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=50, max_depth=5, random_state=42, eval_metric="mlogloss"
        )
    except ImportError:
        pass

    results = []
    trained_models = {}

    print("\n--- Training and Evaluating Candidate Models ---")
    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)

        results.append({
            "Model": name,
            "Accuracy": f"{acc * 100:.2f}%",
            "F1-Macro": f"{f1_macro * 100:.2f}%",
            "Precision": f"{precision_macro * 100:.2f}%",
            "Recall": f"{recall_macro * 100:.2f}%",
            "_raw_f1": f1_macro,
        })
        trained_models[name] = model

    comparison_df = pd.DataFrame(results).sort_values(by="_raw_f1", ascending=False)
    print("\n" + "=" * 65)
    print("MODEL COMPARISON REPORT")
    print("=" * 65)
    display_df = comparison_df.drop(columns=["_raw_f1"])
    print(display_df.to_string(index=False))

    # Pick winning model
    winning_model_name = comparison_df.iloc[0]["Model"]
    winning_model = trained_models[winning_model_name]
    print(f"\nWinning Model: {winning_model_name} (Highest F1-Macro: {comparison_df.iloc[0]['F1-Macro']})")

    Path("app/ml/artifacts").mkdir(parents=True, exist_ok=True)
    joblib.dump(winning_model, "app/ml/artifacts/best_model.pkl")

    if not FEATURE_COLUMNS_PATH.exists():
        with open(FEATURE_COLUMNS_PATH, "w") as f:
            json.dump(feature_columns, f)

    print(f"Winning model saved to app/ml/artifacts/best_model.pkl")


if __name__ == "__main__":
    main()