"""
Attack Classes + Feature Analysis — Week 3
Hybrid ML Based Intrusion Detection System

Consumes the artifacts written by Week 2's data_pipeline.py
(data/processed/*.csv, configs/feature_columns.json,
configs/label_encoder.joblib) and produces:

  1. Attack class distribution (counts + %) -> configs/class_distribution.json
     and a bar chart -> docs/class_distribution.png
  2. Feature importance ranking (Random Forest + mutual information)
     -> configs/feature_importance.json and docs/feature_importance.png
  3. A feature correlation heatmap (to flag redundant/highly-correlated
     features) -> docs/feature_correlation.png
  4. A short markdown summary -> docs/attack_feature_analysis.md

Usage:
    python attack_feature_analysis.py \
        --processed-dir data/processed \
        --config-dir configs \
        --docs-dir docs \
        --top-k 20
"""

import argparse
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

RANDOM_STATE = 42


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_artifacts(processed_dir: str, config_dir: str):
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train_multiclass.csv"))["label"].values

    with open(os.path.join(config_dir, "feature_columns.json")) as f:
        schema = json.load(f)

    label_encoder = joblib.load(os.path.join(config_dir, "label_encoder.joblib"))

    return X_train, y_train, schema, label_encoder


# --------------------------------------------------------------------------
# 1. Attack class distribution
# --------------------------------------------------------------------------
def analyze_class_distribution(y_train, label_encoder, docs_dir: str, config_dir: str):
    classes, counts = np.unique(y_train, return_counts=True)
    class_names = [label_encoder.classes_[c] for c in classes]
    total = counts.sum()

    dist = [
        {"class": name, "count": int(cnt), "pct": round(100 * cnt / total, 3)}
        for name, cnt in sorted(zip(class_names, counts), key=lambda t: -t[1])
    ]

    with open(os.path.join(config_dir, "class_distribution.json"), "w") as f:
        json.dump({"total_samples": int(total), "classes": dist}, f, indent=2)

    # bar chart, log scale since IDS datasets are heavily imbalanced
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(dist))))
    names = [d["class"] for d in dist]
    vals = [d["count"] for d in dist]
    ax.barh(names[::-1], vals[::-1], color="#2563EB")
    ax.set_xscale("log")
    ax.set_xlabel("Sample count (log scale)")
    ax.set_title("Attack Class Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, "class_distribution.png"), dpi=150)
    plt.close()

    print(f"[classes] {len(dist)} classes, most common = {dist[0]['class']} "
          f"({dist[0]['pct']}%), rarest = {dist[-1]['class']} ({dist[-1]['pct']}%)")
    return dist


# --------------------------------------------------------------------------
# 2. Feature importance
# --------------------------------------------------------------------------
def analyze_feature_importance(X_train, y_train, feature_cols, top_k, docs_dir, config_dir):
    # Random Forest importance (fast, robust baseline for tabular flow features)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    rf_importance = rf.feature_importances_

    # Mutual information as a second, non-linear-agnostic signal
    mi = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE, discrete_features=False)

    df_imp = pd.DataFrame({
        "feature": feature_cols,
        "rf_importance": rf_importance,
        "mutual_info": mi,
    })
    df_imp["rf_rank"] = df_imp["rf_importance"].rank(ascending=False)
    df_imp["mi_rank"] = df_imp["mutual_info"].rank(ascending=False)
    df_imp["combined_rank"] = (df_imp["rf_rank"] + df_imp["mi_rank"]) / 2
    df_imp = df_imp.sort_values("combined_rank")

    df_imp.to_json(os.path.join(config_dir, "feature_importance.json"), orient="records", indent=2)

    top = df_imp.head(top_k)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * top_k)))
    ax.barh(top["feature"][::-1], top["rf_importance"][::-1], color="#059669")
    ax.set_xlabel("Random Forest importance")
    ax.set_title(f"Top {top_k} Features by Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, "feature_importance.png"), dpi=150)
    plt.close()

    print(f"[features] top-{top_k} by combined rank: {list(top['feature'][:10])}")
    return df_imp


# --------------------------------------------------------------------------
# 3. Correlation heatmap (redundancy check)
# --------------------------------------------------------------------------
def analyze_correlation(X_train, top_features, docs_dir, threshold=0.9):
    corr = X_train[top_features].corr()

    fig, ax = plt.subplots(figsize=(0.5 * len(top_features) + 2, 0.5 * len(top_features) + 2))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(top_features)))
    ax.set_yticks(range(len(top_features)))
    ax.set_xticklabels(top_features, rotation=90, fontsize=7)
    ax.set_yticklabels(top_features, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature Correlation (top features)")
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, "feature_correlation.png"), dpi=150)
    plt.close()

    # flag redundant pairs above threshold
    redundant = []
    for i in range(len(top_features)):
        for j in range(i + 1, len(top_features)):
            if abs(corr.iloc[i, j]) >= threshold:
                redundant.append((top_features[i], top_features[j], round(float(corr.iloc[i, j]), 3)))

    print(f"[correlation] {len(redundant)} pairs with |corr| >= {threshold}")
    return redundant


# --------------------------------------------------------------------------
# 4. Markdown summary
# --------------------------------------------------------------------------
def write_summary(dist, df_imp, redundant, top_k, docs_dir):
    lines = ["# Attack Classes & Feature Analysis\n"]

    lines.append("## Attack class distribution\n")
    lines.append("| Class | Count | % |\n|---|---|---|")
    for d in dist:
        lines.append(f"| {d['class']} | {d['count']} | {d['pct']}% |")
    lines.append("\n![class distribution](class_distribution.png)\n")

    lines.append(f"## Top {top_k} features (Random Forest + mutual information)\n")
    lines.append("| Feature | RF importance | Mutual info |\n|---|---|---|")
    for _, row in df_imp.head(top_k).iterrows():
        lines.append(f"| {row['feature']} | {row['rf_importance']:.4f} | {row['mutual_info']:.4f} |")
    lines.append("\n![feature importance](feature_importance.png)\n")

    lines.append("## Redundant feature pairs (|corr| >= 0.9)\n")
    if redundant:
        lines.append("| Feature A | Feature B | Corr |\n|---|---|---|")
        for a, b, c in redundant:
            lines.append(f"| {a} | {b} | {c} |")
    else:
        lines.append("_None found above threshold among the top features analyzed._")
    lines.append("\n![feature correlation](feature_correlation.png)\n")

    out_path = os.path.join(docs_dir, "attack_feature_analysis.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[save] wrote {out_path}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def run(processed_dir, config_dir, docs_dir, top_k):
    os.makedirs(docs_dir, exist_ok=True)

    X_train, y_train, schema, label_encoder = load_artifacts(processed_dir, config_dir)
    feature_cols = schema["feature_columns"]

    dist = analyze_class_distribution(y_train, label_encoder, docs_dir, config_dir)
    df_imp = analyze_feature_importance(X_train, y_train, feature_cols, top_k, docs_dir, config_dir)
    redundant = analyze_correlation(X_train, list(df_imp.head(top_k)["feature"]), docs_dir)
    write_summary(dist, df_imp, redundant, top_k, docs_dir)

    print("\n[SUCCESS] Attack class + feature analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attack class distribution + feature importance/correlation analysis")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    run(args.processed_dir, args.config_dir, args.docs_dir, args.top_k)
