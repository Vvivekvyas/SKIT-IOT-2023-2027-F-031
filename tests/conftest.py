"""Pytest fixtures providing synthetic flow records for testing."""

import pytest
import numpy as np
import pandas as pd

from src.data.schemas import (
    CICIDS2017_MANDATORY_NUMERICAL_FEATURES,
    CIC_IOT2023_MANDATORY_NUMERICAL_FEATURES,
)


@pytest.fixture
def sample_cicids2017_valid_df() -> pd.DataFrame:
    """Generates a valid, clean sample DataFrame matching CICIDS2017 schema."""
    np.random.seed(42)
    n_samples = 100

    data = {}
    for col in CICIDS2017_MANDATORY_NUMERICAL_FEATURES:
        data[col] = np.random.uniform(10.0, 1000.0, size=n_samples)

    # Specific realistic adjustments
    data["Flow Duration"] = np.random.uniform(100.0, 50000.0, size=n_samples)
    data["Total Fwd Packets"] = np.random.randint(1, 100, size=n_samples)
    data["Total Backward Packets"] = np.random.randint(1, 100, size=n_samples)

    # Add valid labels
    labels = ["BENIGN"] * 60 + ["DDoS"] * 20 + ["PortScan"] * 15 + ["Bot"] * 5
    data["Label"] = labels

    return pd.DataFrame(data)


@pytest.fixture
def sample_ciciot2023_valid_df() -> pd.DataFrame:
    """Generates a valid, clean sample DataFrame matching CIC-IoT2023 schema."""
    np.random.seed(42)
    n_samples = 100

    data = {}
    for col in CIC_IOT2023_MANDATORY_NUMERICAL_FEATURES:
        data[col] = np.random.uniform(1.0, 500.0, size=n_samples)

    data["flow_duration"] = np.random.uniform(0.01, 10.0, size=n_samples)
    data["Header_Length"] = np.random.randint(20, 60, size=n_samples)

    # Add valid IoT labels
    labels = ["BenignTraffic"] * 50 + ["DDoS-SYN_Flood"] * 25 + ["Mirai-greeth_flood"] * 25
    data["label"] = labels

    return pd.DataFrame(data)


@pytest.fixture
def sample_cicids2017_dirty_df(sample_cicids2017_valid_df: pd.DataFrame) -> pd.DataFrame:
    """Generates a DataFrame with anomalies: infs, NaNs, duplicates, leakage cols, negative values."""
    dirty_df = sample_cicids2017_valid_df.copy()

    # 1. Add leakage columns
    dirty_df["Source IP"] = "192.168.1.10"
    dirty_df["Destination IP"] = "172.16.0.1"
    dirty_df["Source Port"] = 443
    dirty_df["Timestamp"] = "2017-07-07 10:00:00"

    # 2. Inject +inf and -inf
    dirty_df.loc[2, "Flow Bytes/s"] = np.inf
    dirty_df.loc[5, "Flow Packets/s"] = -np.inf

    # 3. Inject NaN values
    dirty_df.loc[10:12, "Flow IAT Mean"] = np.nan

    # 4. Inject duplicate rows
    duplicate_rows = dirty_df.iloc[0:5].copy()
    dirty_df = pd.concat([dirty_df, duplicate_rows], ignore_index=True)

    # 5. Inject a zero-variance column
    dirty_df["Fwd PSH Flags"] = 0.0

    # 6. Inject negative duration violation
    dirty_df.loc[15, "Flow Duration"] = -500.0

    return dirty_df
