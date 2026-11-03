"""
Baseline Model Security Interpretation — Week 8
Reads:  app/ml/artifacts/best_model.pkl, feature_columns.json,
        app/data/processed/test.csv, zero_day_test.csv (optional)
Writes: docs/baseline_security_interpretation.md
Run:    python scripts/run_baseline_security_interpretation.py
"""

import argparse
import json
import os
import re

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

LABEL_COL = "label"  # change if your processed CSVs use a different target column name
IDENTITY_PATTERN = re.compile(
    r"(^|_)(ip|port|flow.?id|mac|timestamp|src.?ip|dst.?ip)($|_)", re.IGNORECASE
)


def load_artifacts(artifacts_dir, processed_dir):
    model = joblib.load(os.path.join(artifacts_dir, "best_model.pkl"))

    with open(os.path.join(artifacts_dir, "feature_columns.json")) as f:
        schema = json.load(f)
    feature_cols = schema["feature_columns"] if isinstance(schema, dict) else schema

    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))

    zero_day_path = os.path.join(processed_dir, "zero_day_test.csv")
    zero_day_df = pd.read_csv(zero_day_path) if os.path.exists(zero_day_path) else None
    if zero_day_df is None:
        print(f"[WARN] {zero_day_path} not found - skipping zero-day section")

    return model, feature_cols, test_df, zero_day_df


def analyze_false_negatives(model, feature_cols, test_df):
    X = test_df[feature_cols]
    y_true = test_df[LABEL_COL].astype(str)
    y_pred = pd.Series(model.predict(X), index=test_df.index).astype(str)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    dangerous_misses = {}
    if "Normal" in cm_df.columns:
        for attack_label in sorted(y_true.unique()):
            if attack_label == "Normal":
                continue
            total = cm_df.loc[attack_label].sum()
            missed = cm_df.loc[attack_label, "Normal"]
            if total > 0:
                dangerous_misses[attack_label] = {
                    "total_samples": int(total),
                    "missed_as_normal": int(missed),
                    "miss_rate_percent": round(100 * missed / total, 2),
                }

    return report, cm_df, dangerous_misses


def analyze_zero_day(model, feature_cols, zero_day_df):
    if zero_day_df is None or len(zero_day_df) == 0:
        return None

    X = zero_day_df[feature_cols]
    y_pred = pd.Series(model.predict(X), index=zero_day_df.index).astype(str)

    total = len(y_pred)
    detected = int((y_pred != "Normal").sum())
    return {
        "total_zero_day_samples": total,
        "flagged_as_any_attack": detected,
        "flagged_as_normal_missed": total - detected,
        "zero_day_detection_rate_percent": round(100 * detected / total, 2),
        "predicted_label_distribution": y_pred.value_counts().to_dict(),
    }


def analyze_feature_reliance(model, feature_cols, top_k=15):
    if not hasattr(model, "feature_importances_"):
        return None, []
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top = importances.head(top_k)
    flagged = [f for f in top.index if IDENTITY_PATTERN.search(f)]
    return top, flagged


def write_report(report, dangerous_misses, zero_day_result, top_importances, flagged, docs_dir):
    os.makedirs(docs_dir, exist_ok=True)
    L = ["# Baseline Model - Security Interpretation (Week 8)\n"]

    overall = report.get("weighted avg", {})
    L += ["## Overall test-set performance\n",
          f"- Weighted precision: {overall.get('precision', 0):.3f}",
          f"- Weighted recall: {overall.get('recall', 0):.3f}",
          f"- Weighted F1: {overall.get('f1-score', 0):.3f}\n"]

    L += ["## Dangerous false negatives (attack traffic predicted as Normal)\n",
          "Overall accuracy can look fine while specific attack categories slip through undetected.\n"]
    if dangerous_misses:
        L += ["| Attack category | Samples | Missed as Normal | Miss rate |", "|---|---|---|---|"]
        for cat, i in sorted(dangerous_misses.items(), key=lambda x: -x[1]["miss_rate_percent"]):
            L.append(f"| {cat} | {i['total_samples']} | {i['missed_as_normal']} | {i['miss_rate_percent']}% |")
        worst = max(dangerous_misses.items(), key=lambda x: x[1]["miss_rate_percent"])
        L.append(f"\n**Highest-risk category: `{worst[0]}`** - {worst[1]['miss_rate_percent']}% of its "
                 f"traffic is classified as Normal. R2L/U2R are typically hardest for flow-based IDS "
                 f"(low sample count, subtle signal) and usually need class-weighting or resampling.\n")
    else:
        L.append("No 'Normal' column in the confusion matrix, or no attack samples in test set.\n")

    L.append("## Zero-day holdout (U2R) generalization\n")
    if zero_day_result:
        L += [f"- Zero-day samples: {zero_day_result['total_zero_day_samples']}",
              f"- Flagged as some attack: {zero_day_result['flagged_as_any_attack']} "
              f"({zero_day_result['zero_day_detection_rate_percent']}%)",
              f"- Missed (predicted Normal): {zero_day_result['flagged_as_normal_missed']}\n",
              "A low detection rate means the baseline is pattern-matching known categories, "
              "not generalizing to unseen attack behavior.\n"]
    else:
        L.append("`zero_day_test.csv` not found - run the pipeline with `--zero-day-categories U2R` first.\n")

    L.append("## Feature reliance vs. known leak-risk features\n")
    if top_importances is not None:
        L += ["| Feature | Importance |", "|---|---|"]
        for feat, imp in top_importances.items():
            tag = " (identity-style)" if feat in flagged else ""
            L.append(f"| {feat}{tag} | {imp:.4f} |")
        if flagged:
            L.append(f"\n**{len(flagged)} top feature(s) match the identity-style pattern "
                     f"({', '.join(flagged)}).** The model may be memorizing hosts from the lab capture "
                     f"instead of attack behavior. Cross-check with feature_quality_report.md.\n")
        else:
            L.append("\nNo top feature matches the identity-style leak pattern.\n")
    else:
        L.append("Model has no `feature_importances_` (not tree-based) - use permutation importance or SHAP.\n")

    with open(os.path.join(docs_dir, "baseline_security_interpretation.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("[save] wrote docs/baseline_security_interpretation.md")


def run(artifacts_dir, processed_dir, docs_dir):
    model, feature_cols, test_df, zero_day_df = load_artifacts(artifacts_dir, processed_dir)
    report, cm_df, dangerous_misses = analyze_false_negatives(model, feature_cols, test_df)
    zero_day_result = analyze_zero_day(model, feature_cols, zero_day_df)
    top, flagged = analyze_feature_reliance(model, feature_cols)
    write_report(report, dangerous_misses, zero_day_result, top, flagged, docs_dir)
    print("[SUCCESS] Baseline security interpretation complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts-dir", default="app/ml/artifacts")
    p.add_argument("--processed-dir", default="app/data/processed")
    p.add_argument("--docs-dir", default="docs")
    a = p.parse_args()
    run(a.artifacts_dir, a.processed_dir, a.docs_dir)