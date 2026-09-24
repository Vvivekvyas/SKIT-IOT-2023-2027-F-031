"""
Preprocessing utilities: clean -> encode -> select features -> scale.
Mirrors the "Data Preprocessing" and "Feature Selection" steps in the
proposal's Proposed Methodology diagram.
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler

NON_FEATURE_COLUMNS = {"label", "raw_label", "dataset_source"}


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Both source datasets are known to contain Infinity/NaN values (mostly in
    rate-based columns like Flow Bytes/s when a flow duration is 0) and
    exact duplicate rows. Drop rather than impute — for this data, a bad
    row is more dangerous than a smaller dataset.
    """
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encodes any remaining non-numeric feature columns (e.g. protocol names)."""
    df = df.copy()
    encoders: dict[str, LabelEncoder] = {}
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]

    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    return df, encoders


def select_top_k_features(
    X: pd.DataFrame, y: pd.Series, k: int = 30
) -> list[str]:
    """
    ANOVA F-value feature selection — fast, works well as a first pass on
    high-dimensional flow-statistics data like these two datasets (CICIDS2017
    has 78 columns, CIC-IoT2023 has ~46-83 depending on version).
    Matches the "select important traffic features... to reduce noise and
    improve speed" goal from the proposal.
    """
    k = min(k, X.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    selected_mask = selector.get_support()
    return list(X.columns[selected_mask])


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fits the scaler on train only, to avoid leaking test-set statistics."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler