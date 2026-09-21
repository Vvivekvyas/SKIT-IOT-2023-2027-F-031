import numpy as np
import pandas as pd


def standardize_column_names(df):
    """
    Standardize column names by:
    - removing leading/trailing spaces
    - replacing spaces with underscores
    """

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_", regex=False)
    )

    return df


def replace_infinite_values(df):
    """
    Replace positive and negative infinity with NaN.
    """

    df = df.copy()

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df


def remove_duplicate_rows(df):
    """
    Remove duplicate rows.
    """

    df = df.copy()

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print("Duplicate rows removed:", before - after)

    return df


def handle_missing_values(df):
    """
    Remove rows containing missing values.

    Missing-value handling will be reviewed further
    during preprocessing.
    """

    df = df.copy()

    before = len(df)

    df = df.dropna()

    after = len(df)

    print("Rows removed because of missing values:", before - after)

    return df


def clean_dataset(df):
    """
    Complete basic dataset cleaning pipeline.
    """

    print("Original shape:", df.shape)

    df = standardize_column_names(df)

    df = replace_infinite_values(df)

    df = remove_duplicate_rows(df)

    df = handle_missing_values(df)

    print("Cleaned shape:", df.shape)

    return df