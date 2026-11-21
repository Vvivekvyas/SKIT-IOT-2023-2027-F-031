import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def numeric_summary(df, prep):
    rows = []
    for c in prep.numeric_cols:
        col = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        rows.append(
            {
                "feature": c,
                "type": "numeric",
                "missing_pct": round(float(col.isna().mean() * 100), 3),
                "unique": int(col.nunique(dropna=True)),
                "mean": float(col.mean()),
                "std": float(col.std()),
                "min": float(col.min()),
                "max": float(col.max()),
                "skew": float(col.skew()),
            }
        )
    return rows


def categorical_summary(df, prep):
    rows = []
    for c in prep.categorical_cols:
        col = df[c]
        top = col.astype(str).value_counts(normalize=True)
        rows.append(
            {
                "feature": c,
                "type": "categorical",
                "missing_pct": round(float(col.isna().mean() * 100), 3),
                "unique": int(col.nunique(dropna=True)),
                "top_category": str(top.index[0]) if len(top) else None,
                "top_share_pct": round(float(top.iloc[0] * 100), 3) if len(top) else None,
            }
        )
    return rows


def correlation_pairs(df, prep, threshold=0.8):
    if len(prep.numeric_cols) < 2:
        return []
    numeric = df[prep.numeric_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    corr = numeric.corr().abs()
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if pd.notna(value) and value >= threshold:
                pairs.append({"a": cols[i], "b": cols[j], "abs_corr": round(float(value), 4)})
    return sorted(pairs, key=lambda p: -p["abs_corr"])


def mutual_information(x_num, x_cat, y, prep):
    n_num = x_num.shape[1]
    x = np.concatenate([x_num.numpy(), x_cat.numpy().astype(np.float32)], axis=1)
    if x.shape[1] == 0:
        return []
    discrete = np.array([False] * n_num + [True] * x_cat.shape[1])
    if prep.task == "classification":
        scores = mutual_info_classif(x, y.numpy(), discrete_features=discrete, random_state=prep.seed)
    else:
        scores = mutual_info_regression(x, y.numpy(), discrete_features=discrete, random_state=prep.seed)
    names = prep.feature_cols
    ranked = sorted(zip(names, scores), key=lambda item: -item[1])
    return [{"feature": n, "mutual_info": round(float(s), 5)} for n, s in ranked]


def unseen_category_rate(frames, prep):
    report = {}
    for split in ("val", "test"):
        frame = frames[split]
        rates = {}
        for c in prep.categorical_cols:
            known = prep.category_maps[c]
            values = frame[c].dropna().astype(str)
            rates[c] = round(float((~values.isin(known)).mean() * 100), 3) if len(values) else 0.0
        report[split] = rates
    return report


def token_layout(prep, tensors, d_token=64):
    x_num, x_cat, _ = tensors["train"]
    n_tokens = x_num.shape[1] + x_cat.shape[1] + 1
    return {
        "n_numeric_tokens": int(x_num.shape[1]),
        "n_categorical_tokens": int(x_cat.shape[1]),
        "cls_token": 1,
        "sequence_length": int(n_tokens),
        "d_token": d_token,
        "input_tensor_shape_per_batch": f"[batch, {n_tokens}, {d_token}]",
        "x_num_shape": list(x_num.shape),
        "x_cat_shape": list(x_cat.shape),
        "x_num_dtype": str(x_num.dtype),
        "x_cat_dtype": str(x_cat.dtype),
        "attention_matrix_size": int(n_tokens * n_tokens),
        "cardinalities": prep.cardinalities,
        "embedding_rows_total": int(sum(prep.cardinalities)),
    }


def numeric_sanity(tensors):
    out = {}
    for split, (x_num, x_cat, y) in tensors.items():
        out[split] = {
            "rows": int(x_num.shape[0]),
            "num_nan": int(np.isnan(x_num.numpy()).sum()),
            "num_inf": int(np.isinf(x_num.numpy()).sum()),
            "num_mean_abs": float(np.abs(x_num.numpy()).mean()) if x_num.numel() else 0.0,
            "num_max_abs": float(np.abs(x_num.numpy()).max()) if x_num.numel() else 0.0,
            "cat_unknown_pct": round(float((x_cat.numpy() == 0).mean() * 100), 3) if x_cat.numel() else 0.0,
        }
    return out


def class_balance(frames, prep):
    if prep.task != "classification":
        return {}
    counts = frames["train"][prep.target].astype(str).value_counts()
    return {k: int(v) for k, v in counts.items()}


def analyze(frames, tensors, prep, d_token=64, corr_threshold=0.8):
    train_df = frames["train"]
    x_num, x_cat, y = tensors["train"]
    report = {
        "token_layout": token_layout(prep, tensors, d_token),
        "numeric_features": numeric_summary(train_df, prep),
        "categorical_features": categorical_summary(train_df, prep),
        "high_correlation_pairs": correlation_pairs(train_df, prep, corr_threshold),
        "mutual_information_ranking": mutual_information(x_num, x_cat, y, prep),
        "unseen_category_rate_pct": unseen_category_rate(frames, prep),
        "tensor_sanity": numeric_sanity(tensors),
        "class_balance_train": class_balance(frames, prep),
    }
    return report


def save_report(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "feature_analysis.json").write_text(json.dumps(report, indent=2))
    features = pd.DataFrame(report["numeric_features"] + report["categorical_features"])
    features.to_csv(directory / "feature_summary.csv", index=False)
    pd.DataFrame(report["mutual_information_ranking"]).to_csv(directory / "mutual_information.csv", index=False)
    return directory
