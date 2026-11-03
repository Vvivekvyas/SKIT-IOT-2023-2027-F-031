"""
Model-agnostic feature reliance check (works for SVM, XGBoost, pipelines, etc.),
for when the baseline has no `feature_importances_`.

Place in: scripts/baseline_permutation_importance.py
Run:      python scripts/baseline_permutation_importance.py
Writes:   docs/baseline_permutation_importance.csv
"""

import argparse
import json
import os
import re

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

LABEL_COL = "label"  # change if your processed CSVs use a different target column name
IDENTITY_PATTERN = re.compile(
    r"(^|_)(ip|port|flow.?id|mac|timestamp|src.?ip|dst.?ip)($|_)", re.IGNORECASE
)


def run(artifacts_dir, processed_dir, docs_dir, n_repeats, max_rows, seed):
    model = joblib.load(os.path.join(artifacts_dir, "best_model.pkl"))
    with open(os.path.join(artifacts_dir, "feature_columns.json")) as f:
        schema = json.load(f)
    cols = schema["feature_columns"] if isinstance(schema, dict) else schema

    df = pd.read_csv(os.path.join(processed_dir, "test.csv"))
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed)

    result = permutation_importance(
        model, df[cols], df[LABEL_COL].astype(str),
        n_repeats=n_repeats, random_state=seed, n_jobs=-1, scoring="f1_weighted",
    )
    out = pd.DataFrame({
        "feature": cols,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    out["identity_style"] = out["feature"].map(lambda f: bool(IDENTITY_PATTERN.search(f)))

    os.makedirs(docs_dir, exist_ok=True)
    path = os.path.join(docs_dir, "baseline_permutation_importance.csv")
    out.to_csv(path, index=False)
    print(out.head(15).to_string(index=False))
    flagged = out[out["identity_style"] & (out["importance_mean"] > 0)]
    if len(flagged):
        print(f"\n[WARN] identity-style features with non-zero importance: {', '.join(flagged['feature'])}")
    print(f"\n[save] wrote {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts-dir", default="app/ml/artifacts")
    p.add_argument("--processed-dir", default="app/data/processed")
    p.add_argument("--docs-dir", default="docs")
    p.add_argument("--n-repeats", type=int, default=5)
    p.add_argument("--max-rows", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    run(a.artifacts_dir, a.processed_dir, a.docs_dir, a.n_repeats, a.max_rows, a.seed)