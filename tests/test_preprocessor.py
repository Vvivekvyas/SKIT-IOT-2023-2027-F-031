"""Tests for DataPreprocessor pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Verifies end-to-end preprocessing, stratified splitting, and leakage prevention."""

    def test_stratified_split_ratios(self, sample_cicids2017_valid_df: pd.DataFrame):
        """Verify train/val/test splits match requested ratios and preserve class distribution."""
        preprocessor = DataPreprocessor(dataset_name="CICIDS2017")
        train_df, val_df, test_df = preprocessor.split_data(
            sample_cicids2017_valid_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
        )

        assert len(train_df) == 68  # integer floor per class
        assert len(val_df) == 13
        assert len(test_df) == 19
        assert len(train_df) + len(val_df) + len(test_df) == len(sample_cicids2017_valid_df)

        # Ensure all classes exist in training set
        assert set(train_df["Label"].unique()) == set(sample_cicids2017_valid_df["Label"].unique())

    def test_minmax_scaling_bounds(self, sample_cicids2017_valid_df: pd.DataFrame):
        """Verify that MinMax scaled features strictly fall within [0.0, 1.0]."""
        preprocessor = DataPreprocessor(dataset_name="CICIDS2017", scaler_type="minmax")
        train_df, val_df, test_df = preprocessor.split_data(sample_cicids2017_valid_df)

        X_train, y_train = preprocessor.fit_transform(train_df)
        X_test, y_test = preprocessor.transform(test_df)

        assert X_train.min() >= 0.0
        assert X_train.max() <= 1.0
        assert X_test.min() >= 0.0
        assert X_test.max() <= 1.0

    def test_label_encoding_consistency(self, sample_cicids2017_valid_df: pd.DataFrame):
        """Verify integer label encoding and bidirectional decoding map."""
        preprocessor = DataPreprocessor(dataset_name="CICIDS2017")
        train_df, val_df, test_df = preprocessor.split_data(sample_cicids2017_valid_df)

        X_train, y_train = preprocessor.fit_transform(train_df)
        X_test, y_test = preprocessor.transform(test_df)

        num_classes = len(preprocessor.class_to_idx_)
        assert y_train.min() >= 0
        assert y_train.max() < num_classes
        assert y_test.min() >= 0
        assert y_test.max() < num_classes

        # Verify mapping invertibility
        for cls_name, idx in preprocessor.class_to_idx_.items():
            assert preprocessor.idx_to_class_[idx] == cls_name

    def test_unseen_transform_without_fit_raises(self, sample_cicids2017_valid_df: pd.DataFrame):
        """Verify error is raised if transform is invoked before fit."""
        preprocessor = DataPreprocessor(dataset_name="CICIDS2017")
        with pytest.raises(RuntimeError):
            preprocessor.transform(sample_cicids2017_valid_df)
