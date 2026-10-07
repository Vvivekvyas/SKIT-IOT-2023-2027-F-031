"""
Runs Week 6's security-oriented feature analysis.

Combines:
  - feature importance (RandomForest, computed fresh on app/data/processed/train.csv)
  - spoofability classification (rule-based on feature names)
  - perturbation robustness (only if app/ml/artifacts/best_model.pkl exists)
  - zero-day distribution shift (only if app/data/processed/zero_day_test.csv
    has rows — i.e. the pipeline was run with --zero-day-categories)

Run after scripts/run_data_pipeline.py (and ideally scripts/train_models.py
so the perturbation check has a real model to test):

    python scripts/run_security_feature_analysis.py

Output: app/data/processed/security_feature_analysis_report.md
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from app.data.security_feature_analysis import (  # noqa: E402
    perturbation_robustness,
    rank_security_risk,
    zero_day_distribution_shift,
)
from app.data.security_feature_analysis_report import to_markdown  # noqa: E402

TRAIN_PATH = Path("app/data/processed/train.csv")
ZERO_DAY_PATH = Path("app/data/processed/zero_day_test.csv")
FEATURE_COLUMNS_PATH = Path("app/ml/artifacts/feature_columns.json")
MODEL_PATH = Path("app/ml/artifacts/best_model.pkl")
REPORT_PATH = Path("app/data/processed/security_feature_analysis_report.md")


def compute_importance(X: pd.DataFrame, y: pd.Series) -> list[tuple[str, float]]:
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    ranked = sorted(zip(X.columns, model.feature_importances_), key=lambda x: -x[1])
    return [(name, round(float(score), 4)) for name, score in ranked]


def main() -> None:
    if not TRAIN_PATH.exists() or not FEATURE_COLUMNS_PATH.exists():
        raise SystemExit(
            "Missing app/data/processed/train.csv or feature_columns.json.\n"
            "Run `python scripts/run_data_pipeline.py` first."
        )

    train_df = pd.read_csv(TRAIN_PATH)
    with open(FEATURE_COLUMNS_PATH) as f:
        feature_columns = json.load(f)

    importance_ranking = compute_importance(train_df[feature_columns], train_df["label"])

    flip_rates = None
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        flip_rates = perturbation_robustness(model, train_df, feature_columns)
    else:
        print(
            "No trained model found at app/ml/artifacts/best_model.pkl — "
            "skipping perturbation robustness check (run scripts/train_models.py first for that)."
        )

    distribution_shifts = None
    if ZERO_DAY_PATH.exists():
        zero_day_df = pd.read_csv(ZERO_DAY_PATH)
        if len(zero_day_df) > 0:
            distribution_shifts = zero_day_distribution_shift(train_df, zero_day_df, feature_columns)
        else:
            print(
                "zero_day_test.csv is empty — skipping distribution-shift check "
                "(re-run the pipeline with --zero-day-categories to populate it)."
            )

    rankings = rank_security_risk(importance_ranking, flip_rates, distribution_shifts)
    markdown = to_markdown(rankings)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(markdown)

    print(markdown)
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
    