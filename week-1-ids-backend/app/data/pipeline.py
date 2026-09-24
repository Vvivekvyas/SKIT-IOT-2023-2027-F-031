"""
End-to-end dataset loading pipeline. Ties together loaders.py +
preprocessing.py and produces exactly the artifacts the rest of the
project expects:

  app/data/processed/train.csv         -> for scripts/train_models.py
  app/data/processed/test.csv          -> held-out normal test split
  app/data/processed/zero_day_test.csv -> attack categories excluded from
                                           training entirely (Proposed
                                           Methodology's "Zero-day setup" step)
  app/ml/artifacts/feature_columns.json -> read by app/ml/model_registry.py
  app/ml/artifacts/scaler.pkl           -> StandardScaler fit on train only
  app/ml/artifacts/label_encoders.pkl   -> per-column encoders for any
                                            categorical features

Run it with:
  python scripts/run_data_pipeline.py --dataset both --k-features 30 \\
      --zero-day-categories U2R
"""
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from app.data.loaders import load_combined
from app.data.preprocessing import clean_dataframe, encode_categoricals, select_top_k_features, scale_features

PROCESSED_DIR = Path("app/data/processed")
ARTIFACTS_DIR = Path("app/ml/artifacts")


@dataclass
class PipelineResult:
    n_train: int
    n_test: int
    n_zero_day: int
    feature_columns: list[str]
    class_balance: dict[str, int]


def run_pipeline(
    dataset: str = "both",              # "cicids2017" | "ciciot2023" | "both"
    k_features: int = 30,
    zero_day_categories: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> PipelineResult:
    zero_day_categories = zero_day_categories or []
    include = ("cicids2017", "ciciot2023") if dataset == "both" else (dataset,)

    # 1) Load + 2) clean
    df = load_combined(include=include)
    df = clean_dataframe(df)

    # 3) Zero-day setup — remove chosen attack categories from the pool used
    #    for training/normal-testing entirely; they're only ever seen at
    #    zero-day evaluation time, simulating genuinely unseen attacks.
    zero_day_mask = df["label"].isin(zero_day_categories)
    zero_day_df = df[zero_day_mask].copy()
    df = df[~zero_day_mask].copy()

    # 4) Encode categorical features
    df, encoders = encode_categoricals(df)
    zero_day_df, _ = encode_categoricals(zero_day_df) if len(zero_day_df) else (zero_day_df, {})

    feature_cols_all = [c for c in df.columns if c not in ("label", "raw_label", "dataset_source")]
    X, y = df[feature_cols_all], df["label"]

    # 5) Feature selection
    selected_features = select_top_k_features(X, y, k=k_features)
    X = X[selected_features]

    # 6) Train/test split (stratified so rare attack classes stay represented in both)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 7) Scale (fit on train only)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Assemble output frames
    train_df = pd.DataFrame(X_train_scaled, columns=selected_features)
    train_df["label"] = y_train.reset_index(drop=True)

    test_df = pd.DataFrame(X_test_scaled, columns=selected_features)
    test_df["label"] = y_test.reset_index(drop=True)

    if len(zero_day_df):
        zd_features = zero_day_df.reindex(columns=selected_features, fill_value=0)
        zd_scaled = scaler.transform(zd_features)
        zero_day_out = pd.DataFrame(zd_scaled, columns=selected_features)
        zero_day_out["label"] = zero_day_df["label"].reset_index(drop=True)
    else:
        zero_day_out = pd.DataFrame(columns=selected_features + ["label"])

    # Persist everything
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    zero_day_out.to_csv(PROCESSED_DIR / "zero_day_test.csv", index=False)

    with open(ARTIFACTS_DIR / "feature_columns.json", "w") as f:
        json.dump(selected_features, f, indent=2)

    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.pkl")
    joblib.dump(encoders, ARTIFACTS_DIR / "label_encoders.pkl")

    return PipelineResult(
        n_train=len(train_df),
        n_test=len(test_df),
        n_zero_day=len(zero_day_out),
        feature_columns=selected_features,
        class_balance=y.value_counts().to_dict(),
    )