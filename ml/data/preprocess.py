import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def add_iot_attack_category(df):
    """
    Extract CIC-IoT2023 attack category from Source_File.

    Example:
    XSS\\XSS.pcap.csv
        -> XSS

    Benign_Final\\BenignTraffic.pcap.csv
        -> Benign_Final
    """

    df = df.copy()

    if "Source_File" not in df.columns:
        raise ValueError(
            "Source_File column is required."
        )

    df["Attack_Category"] = (
        df["Source_File"]
        .astype(str)
        .apply(lambda x: Path(x).parts[0])
    )

    return df

def identify_columns(df, label_column):
    """
    Identify feature and non-feature columns.
    """

    if label_column not in df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in dataframe."
        )

    excluded_columns = {
        label_column,
        "Source_File",
        "Dataset"
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    numeric_columns = df[feature_columns].select_dtypes(
        include=["number"]
    ).columns.tolist()

    non_numeric_columns = [
        column
        for column in feature_columns
        if column not in numeric_columns
    ]

    return numeric_columns, non_numeric_columns


def scale_numeric_features(df, numeric_columns, scaler=None):
    """
    Scale numeric features. If scaler is None, fits on df.
    If a fitted scaler is provided, reuses it without data leakage.
    """
    df = df.copy()

    # Fill any unexpected inf/nan prior to scaling
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
    df[numeric_columns] = df[numeric_columns].fillna(0.0)

    if scaler is None:
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    else:
        df[numeric_columns] = scaler.transform(df[numeric_columns])

    return df, scaler


def preprocess_features(df, label_column, scaler=None):
    """
    Identify and scale numeric features.
    """
    numeric_columns, non_numeric_columns = identify_columns(
        df,
        label_column
    )

    processed_df, scaler = scale_numeric_features(
        df,
        numeric_columns,
        scaler=scaler,
    )

    return (
        processed_df,
        scaler,
        numeric_columns,
        non_numeric_columns
    )
