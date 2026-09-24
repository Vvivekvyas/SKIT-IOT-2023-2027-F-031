"""Tests for DataCleaner component."""

import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import DataCleaner


class TestDataCleaner:
    """Verifies data sanitization, leakage elimination, and imputation."""

    def test_leakage_columns_removal(self, sample_cicids2017_dirty_df: pd.DataFrame):
        """Verify that IP addresses, ports, and timestamps are stripped."""
        cleaner = DataCleaner(drop_leakage=True)
        cleaned = cleaner.fit_transform(sample_cicids2017_dirty_df, label_col="Label")

        assert "Source IP" not in cleaned.columns
        assert "Destination IP" not in cleaned.columns
        assert "Source Port" not in cleaned.columns
        assert "Timestamp" not in cleaned.columns

    def test_duplicate_removal(self, sample_cicids2017_dirty_df: pd.DataFrame):
        """Verify that duplicate rows are eliminated."""
        cleaner = DataCleaner(drop_duplicates=True)
        # sample_cicids2017_dirty_df has 105 rows (100 original + 5 duplicates)
        cleaned = cleaner.fit_transform(sample_cicids2017_dirty_df, label_col="Label")
        assert len(cleaned) == 100

    def test_infinite_and_nan_imputation(self, sample_cicids2017_dirty_df: pd.DataFrame):
        """Verify that inf and NaN are replaced using learned median statistics."""
        cleaner = DataCleaner(imputation_strategy="median")
        cleaned = cleaner.fit_transform(sample_cicids2017_dirty_df, label_col="Label")

        assert not np.isinf(cleaned["Flow Bytes/s"]).any()
        assert not np.isinf(cleaned["Flow Packets/s"]).any()
        assert not cleaned["Flow IAT Mean"].isna().any()

    def test_zero_variance_dropping(self, sample_cicids2017_dirty_df: pd.DataFrame):
        """Verify that constant columns are dropped."""
        cleaner = DataCleaner(drop_zero_variance=True)
        cleaned = cleaner.fit_transform(sample_cicids2017_dirty_df, label_col="Label")

        assert "Fwd PSH Flags" not in cleaned.columns

    def test_ciciot2023_label_consolidation(self, sample_ciciot2023_valid_df: pd.DataFrame):
        """Verify mapping of 33 sub-classes to 8 high-level attack families."""
        cleaner = DataCleaner(consolidate_iot_labels=True)
        cleaned = cleaner.fit_transform(sample_ciciot2023_valid_df, label_col="label")

        unique_labels = set(cleaned["label"].unique())
        # 'DDoS-SYN_Flood' should be mapped to 'DDoS'
        # 'Mirai-greeth_flood' should be mapped to 'Mirai'
        assert "DDoS" in unique_labels
        assert "Mirai" in unique_labels
        assert "DDoS-SYN_Flood" not in unique_labels
        assert "Mirai-greeth_flood" not in unique_labels
