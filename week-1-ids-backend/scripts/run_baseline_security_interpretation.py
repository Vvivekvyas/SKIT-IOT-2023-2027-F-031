"""
Baseline Model Security Interpretation — Week 8
ML-Based Intrusion Detection System

Depends on Week 7/8's baseline model (app/ml/artifacts/best_model.pkl, per
scripts/train_models.py's contract) and the processed pipeline output
(app/data/processed/{train,test,zero_day_test}.csv).

Doesn't just report accuracy — looks at the baseline from a *security*
angle: which attack categories does it actually miss (dangerous false
negatives into "Normal"), does it rely on the identity-style features
Week 4's feature_quality analysis already flagged as leak-prone, and how
does it do on the U2R zero-day holdout (the real test of whether it
generalizes to unseen attack behavior vs. memorized the training traffic).

Reads:
  - app/ml/artifacts/best_model.pkl
  - app/ml/artifacts/feature_columns.json
  - app/data/processed/test.csv
  - app/data/processed/zero_day_test.csv   (optional — skipped with a warning if missing)
  - app/data/processed/feature_quality_report.md   (optional — used to cross-check leaky features)

Writes:
  - docs/baseline_security_interpretation.md

Usage:
    python scripts/run_baseline_security_interpretation.py
"""

import argparse
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

LABEL_COL = "label"  # adjust if your processed CSVs name the target column differently
IDENTITY_PATTERN = re.compile(
    r"(^|_)(ip|port|flow.?id|mac|timestamp|src.?ip|dst.?ip)($|_)", re.IGNORECASE
)


# --------------------------------------------------------------------------
# Load artifacts
# --------------------------------------------------------------------------
def load_artifacts(artifacts_dir: str, processed_dir: str):
    model = joblib.load(os.path.join(artifacts_dir, "best_model.pkl"))

    with open(os.path.join(artifacts_dir, "feature_columns.json")) as f:
        schema = json.load(f)
    feature_cols = schema["feature_columns"] if isinstance(schema, dict) else schema

    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))

    zero_day_path = os.path.join(processed_dir, "zero_day_test.csv")
    zero_day_df = pd.read_csv(zero_day_path) if os.path.exists(zero_day_path) else None
    if zero_day_df is None:
        print(f"[WARN] {zero_day_path} not found — skipping zero-day section")

    return model, feature_cols, test_df, zero_day_df


# --------------------------------------------------------------------------
# Per-category false-negative analysis
# --------------------------------------------------------------------------
def analyze_false_negatives(model, feature_cols, test_df: pd.DataFrame):
    X = test_df[feature_cols]
    y_true = test_df[LABEL_COL].astype(str)
    y_pred = pd.Series(model.predict(X), index=test_df.index).astype(str)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # The security-critical failure mode: attack traffic predicted as Normal
    dangerous_misses = {}
    if "Normal" in cm_df.columns:
        for attack_label in labels:
            if attack_label == "Normal":
                continue
            total = cm_df.loc[attack_label].sum()
            missed_as_normal = cm_df.loc[attack_label, "Normal"] if "Normal" in cm_df.columns else 0
            if total > 0:
                dangerous_misses[attack_label] = {
                    "total_samples": int(total),
                    "missed_as_normal": int(missed_as_normal),
                    "miss_rate_percent": round(100 * missed_as_normal / total, 2),
                }

    return report, cm_df, dangerous_misses


# --------------------------------------------------------------------------
# Zero-day (U2R) generalization check
# --------------------------------------------------------------------------
def analyze_zero_day(model, feature_cols, zero_day_df: pd.DataFrame):
    if zero_day_df is None or len(zero_day_df) == 0:
        return None

    X = zero_day_df[feature_cols]
    y_true = zero_day_df[LABEL_COL].astype(str)
    y_pred = pd.Series(model.predict(X), index=zero_day_df.index).astype(str)

    detected_as_attack = (y_pred != "Normal").sum()
    total = len(y_true)
    return {
        "total_zero_day_samples": int(total),
        "flagged_as_any_attack": int(detected_as_attack),
        "flagged_as_normal_missed": int(total - detected_as_attack),
        "zero_day_detection_rate_percent": round(100 * detected_as_attack / total, 2) if total else 0,
        "predicted_label_distribution": y_pred.value_counts().to_dict(),
    }


# --------------------------------------------------------------------------
# Feature importance + leak-risk cross-check
# --------------------------------------------------------------------------
def analyze_feature_reliance(model, feature_cols, top_k: int = 15):
    if not hasattr(model, "feature_importances_"):
        return None, []

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top = importances.head(top_k)

    flagged = [f for f in top.index if IDENTITY_PATTERN.search(f)]
    return top, flagged


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def write_report(report, cm_df, dangerous_misses, zero_day_result, top_importances, flagged_identity_features, docs_dir: str):
    os.makedirs(docs_dir, exist_ok=True)
    lines = ["# Baseline Model — Security Interpretation (Week 8)\n"]

    lines.append("## Overall test-set performance\n")
    overall = report.get("weighted avg", {})
    lines.append(f"- Weighted precision: {overall.get('precision', 0):.3f}")
    lines.append(f"- Weighted recall: {overall.get('recall', 0):.3f}")
    lines.append(f"- Weighted F1: {overall.get('f1-score', 0):.3f}\n")

    lines.append("## Dangerous false negatives (attack traffic predicted as Normal)\n")
    lines.append("This is the metric that actually matters for an IDS — overall accuracy can look")
    lines.append("fine while specific attack categories slip through undetected.\n")
    if dangerous_misses:
        lines.append("| Attack category | Samples | Missed as Normal | Miss rate |")
        lines.append("|---|---|---|---|")
        for cat, info in sorted(dangerous_misses.items(), key=lambda x: -x[1]["miss_rate_percent"]):
            lines.append(f"| {cat} | {info['total_samples']} | {info['missed_as_normal']} | {info['miss_rate_percent']}% |")
        worst = max(dangerous_misses.items(), key=lambda x: x[1]["miss_rate_percent"])
        lines.append(f"\n**Highest-risk category: `{worst[0]}`** — {worst[1]['miss_rate_percent']}% of its "
                      f"traffic is classified as Normal. If this is R2L or U2R, flag it explicitly: those are "
                      f"historically the hardest categories for flow-based IDS models (low sample count,")
        lines.append("subtle behavioral signal) and typically need class-weighting or resampling, not just a")
        lines.append("better model.\n")
    else:
        lines.append("No \"Normal\" column found in the confusion matrix, or no attack samples in test set.\n")

    lines.append("## Zero-day holdout (U2R) generalization\n")
    if zero_day_result:
        lines.append(f"- Zero-day samples: {zero_day_result['total_zero_day_samples']}")
        lines.append(f"- Flagged as *some* attack (not necessarily U2R): "
                      f"{zero_day_result['flagged_as_any_attack']} "
                      f"({zero_day_result['zero_day_detection_rate_percent']}%)")
        lines.append(f"- Missed entirely (predicted Normal): {zero_day_result['flagged_as_normal_missed']}\n")
        lines.append("A low detection rate here means the baseline isn't generalizing to unseen attack")
        lines.append("behavior — it's pattern-matching the attack categories it was trained on. Worth stating")
        lines.append("plainly in the report rather than only citing the headline accuracy number.\n")
    else:
        lines.append("`zero_day_test.csv` not found — run the data pipeline with "
                      "`--zero-day-categories U2R` first.\n")

    lines.append("## Feature reliance vs. known leak-risk features\n")
    if top_importances is not None:
        lines.append("Top features the baseline actually relies on:\n")
        lines.append("| Feature | Importance |")
        lines.append("|---|---|")
        for feat, imp in top_importances.items():
            flag = " ⚠️ identity-style" if feat in flagged_identity_features else ""
            lines.append(f"| {feat}{flag} | {imp:.4f} |")
        if flagged_identity_features:
            lines.append(f"\n**⚠️ {len(flagged_identity_features)} of the top features "
                          f"({', '.join(flagged_identity_features)}) match the identity-style pattern "
                          f"(IP/port/flow-ID/MAC/timestamp) Week 4's feature-quality analysis already warned "
                          f"about.** If any of these are driving predictions, the model may be memorizing hosts")
            lines.append("from the lab capture (which IPs attacked, which didn't) instead of learning general")
            lines.append("attack *behavior* — it would look great on this test set and fail on any new network.")
            lines.append("Cross-check against `app/data/processed/feature_quality_report.md` before trusting this baseline.\n")
        else:
            lines.append("\nNone of the top features match the identity-style leak pattern — reassuring sign "
                          "that the model is learning from behavioral features, not host identity.\n")
    else:
        lines.append("Model has no `feature_importances_` attribute (not a tree-based model) — "
                      "use permutation importance or SHAP instead for this section.\n")

    with open(os.path.join(docs_dir, "baseline_security_interpretation.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"[save] wrote docs/baseline_security_interpretation.md")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def run(artifacts_dir: str, processed_dir: str, docs_dir: str):
    model, feature_cols, test_df, zero_day_df = load_artifacts(artifacts_dir, processed_dir)

    report, cm_df, dangerous_misses = analyze_false_negatives(model, feature_cols, test_df)
    zero_day_result = analyze_zero_day(model, feature_cols, zero_day_df)
    top_importances, flagged_identity_features = analyze_feature_reliance(model, feature_cols)

    write_report(report, cm_df, dangerous_misses, zero_day_result,
                 top_importances, flagged_identity_features, docs_dir)

    print("\n[SUCCESS] Baseline security interpretation complete.")
    if dangerous_misses:
        worst = max(dangerous_misses.items(), key=lambda x: x[1]["miss_rate_percent"])
        print(f"  highest-risk category: {worst[0]} ({worst[1]['miss_rate_percent']}% missed as Normal)")
    if flagged_identity_features:
        print(f"  ⚠️ {len(flagged_identity_features)} identity-style feature(s) in top importances")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Security-focused interpretation of the baseline IDS model")
    parser.add_argument("--artifacts-dir", default="app/ml/artifacts")
    parser.add_argument("--processed-dir", default="app/data/processed")
    parser.add_argument("--docs-dir", default="docs")
    args = parser.parse_args()

    run(args.artifacts_dir, args.processed_dir, args.docs_dir)