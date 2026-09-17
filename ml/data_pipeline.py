"""
Dataset Loading Pipeline — Week 2
Hybrid ML Based Intrusion Detection System

Loads CICIDS2017 and CIC-IoT2023, cleans/encodes/normalizes them into a
single unified feature space, and writes `feature_columns.json` in the
`configs/` directory so the backend's `model_registry.py` can align
incoming traffic to the exact feature schema the model was trained on.

IMPORTANT: I don't have your actual `model_registry.py`, so the exact
shape of `feature_columns.json` below is my best-practice guess (an
ordered feature list + label mapping + scaler params). Paste in
`model_registry.py` and I'll adjust `save_feature_schema()` to match
whatever key names/structure it actually expects — that's the one part
of this file you should verify before it's "real."

Usage:
    python data_pipeline.py \
        --cicids-dir data/raw/CICIDS2017 \
        --ciciot-dir data/raw/CICIoT2023 \
        --out-dir data/processed \
        --config-dir configs
"""

import argparse
import glob
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

RANDOM_STATE = 42


# --------------------------------------------------------------------------
# 1. Loading
# --------------------------------------------------------------------------
def _read_all_csv(directory: str) -> pd.DataFrame:
    """Read and concatenate every CSV in a directory (CICIDS2017/CIC-IoT2023
    both ship as many per-day / per-attack CSV files)."""
    paths = sorted(glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {directory}")

    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p, low_memory=False, encoding="utf-8", on_bad_lines="skip")
        except UnicodeDecodeError:
            df = pd.read_csv(p, low_memory=False, encoding="latin1", on_bad_lines="skip")
        df["__source_file"] = os.path.basename(p)
        frames.append(df)
        print(f"  loaded {os.path.basename(p):<45} rows={len(df):>8}")
    return pd.concat(frames, ignore_index=True)


def load_cicids2017(directory: str) -> pd.DataFrame:
    print(f"[CICIDS2017] reading from {directory}")
    df = _read_all_csv(directory)
    df.columns = [c.strip() for c in df.columns]  # CICIDS2017 headers have leading spaces
    if "Label" not in df.columns:
        raise KeyError("Expected a 'Label' column in CICIDS2017 files")
    df["dataset_source"] = "CICIDS2017"
    return df


def load_ciciot2023(directory: str) -> pd.DataFrame:
    print(f"[CIC-IoT2023] reading from {directory}")
    df = _read_all_csv(directory)
    df.columns = [c.strip() for c in df.columns]
    label_col = "label" if "label" in df.columns else "Label"
    if label_col not in df.columns:
        raise KeyError("Expected a 'label'/'Label' column in CIC-IoT2023 files")
    df = df.rename(columns={label_col: "Label"})
    df["dataset_source"] = "CIC-IoT2023"
    return df


# --------------------------------------------------------------------------
# 2. Cleaning
# --------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    df = df.drop(columns=[c for c in df.columns if c.startswith("__")], errors="ignore")
    df = df.drop_duplicates()

    # Numeric coercion + inf/NaN handling (both datasets are notorious for
    # stray "Infinity" strings and NaNs in flow-duration-derived columns)
    non_feature_cols = {"Label", "dataset_source"}
    numeric_cols = [c for c in df.columns if c not in non_feature_cols]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0, how="any")

    # drop columns that are constant (zero variance -> useless for the model)
    const_cols = [c for c in numeric_cols if c in df.columns and df[c].nunique() <= 1]
    df = df.drop(columns=const_cols)

    print(f"[clean] {before} -> {len(df)} rows, dropped {len(const_cols)} constant columns")
    return df


def merge_datasets(df_cicids: pd.DataFrame, df_ciciot: pd.DataFrame) -> pd.DataFrame:
    common = sorted(set(df_cicids.columns) & set(df_ciciot.columns))
    # Label/dataset_source must always be kept even if a caller reorders things
    for required in ("Label", "dataset_source"):
        if required not in common:
            common.append(required)
    print(f"[merge] {len(common)} overlapping columns between the two datasets")
    merged = pd.concat([df_cicids[common], df_ciciot[common]], ignore_index=True)
    return merged


# --------------------------------------------------------------------------
# 3. Encode + normalize
# --------------------------------------------------------------------------
def encode_and_scale(df: pd.DataFrame):
    y_raw = df["Label"].astype(str).str.strip()
    is_benign = y_raw.str.lower().isin({"benign", "normal", "background"})

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    y_binary = (~is_benign).astype(int)  # 0 = benign, 1 = attack (any type)

    feature_cols = [c for c in df.columns if c not in ("Label", "dataset_source")]
    X = df[feature_cols].astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return (
        pd.DataFrame(X_scaled, columns=feature_cols),
        y,
        y_binary,
        feature_cols,
        label_encoder,
        scaler,
    )


# --------------------------------------------------------------------------
# 4. Persist artifacts
# --------------------------------------------------------------------------
def save_feature_schema(feature_cols, label_encoder, scaler, config_dir: str):
    """Writes feature_columns.json for model_registry.py to consume at
    inference time (schema alignment) — update this to match the real
    contract once you paste in model_registry.py."""
    os.makedirs(config_dir, exist_ok=True)

    schema = {
        "feature_columns": feature_cols,
        "num_features": len(feature_cols),
        "label_column": "Label",
        "label_classes": list(label_encoder.classes_),
        "label_mapping": {cls: int(i) for i, cls in enumerate(label_encoder.classes_)},
        "scaler": "StandardScaler",
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    out_path = os.path.join(config_dir, "feature_columns.json")
    with open(out_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"[save] wrote {out_path} ({len(feature_cols)} features, "
          f"{len(label_encoder.classes_)} classes)")


def save_processed(X: pd.DataFrame, y, y_binary, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    X_train, X_test, y_train, y_test, yb_train, yb_test = train_test_split(
        X, y, y_binary, test_size=0.2, random_state=RANDOM_STATE, stratify=y_binary
    )

    for name, data in [
        ("X_train", X_train), ("X_test", X_test),
        ("y_train_multiclass", pd.Series(y_train, name="label")),
        ("y_test_multiclass", pd.Series(y_test, name="label")),
        ("y_train_binary", pd.Series(yb_train, name="label")),
        ("y_test_binary", pd.Series(yb_test, name="label")),
    ]:
        data.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)

    print(f"[save] wrote train/test splits to {out_dir} "
          f"(train={len(X_train)}, test={len(X_test)})")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def run(cicids_dir: str, ciciot_dir: str, out_dir: str, config_dir: str):
    df_cicids = clean(load_cicids2017(cicids_dir))
    df_ciciot = clean(load_ciciot2023(ciciot_dir))
    merged = merge_datasets(df_cicids, df_ciciot)

    X, y, y_binary, feature_cols, label_encoder, scaler = encode_and_scale(merged)

    save_processed(X, y, y_binary, out_dir)
    save_feature_schema(feature_cols, label_encoder, scaler, config_dir)
    joblib.dump(scaler, os.path.join(config_dir, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(config_dir, "label_encoder.joblib"))

    print("\n[SUCCESS] Dataset loading pipeline complete.")
    print(f"  rows: {len(merged)}  |  features: {len(feature_cols)}  |  classes: {len(label_encoder.classes_)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load, clean, encode and normalize CICIDS2017 + CIC-IoT2023")
    parser.add_argument("--cicids-dir", default="data/raw/CICIDS2017")
    parser.add_argument("--ciciot-dir", default="data/raw/CICIoT2023")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--config-dir", default="configs")
    args = parser.parse_args()

    run(args.cicids_dir, args.ciciot_dir, args.out_dir, args.config_dir)
