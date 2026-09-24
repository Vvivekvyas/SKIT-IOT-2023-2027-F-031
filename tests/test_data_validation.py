"""Automated data validation test suite corresponding to TC-001 through TC-007."""

import numpy as np
import pandas as pd
import pytest

from src.data.validator import DatasetValidator, ValidationReport


class TestDataValidation:
    """Test suite executing comprehensive data quality, integrity, and schema checks."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.validator = DatasetValidator(max_nan_ratio=0.05)

    def test_tc001_valid_cicids2017_validation(self, sample_cicids2017_valid_df: pd.DataFrame):
        """TC-001: Validate compliant CICIDS2017 flow dataset passes validation."""
        report = self.validator.validate(sample_cicids2017_valid_df, "CICIDS2017")
        assert report.is_valid is True
        assert len(report.errors) == 0
        assert report.total_records == 100
        assert report.duplicate_record_count == 0
        assert sum(report.infinite_value_counts.values()) == 0

    def test_tc002_valid_ciciot2023_validation(self, sample_ciciot2023_valid_df: pd.DataFrame):
        """TC-001b: Validate compliant CIC-IoT2023 dataset passes validation."""
        report = self.validator.validate(sample_ciciot2023_valid_df, "CIC-IoT2023")
        assert report.is_valid is True
        assert len(report.errors) == 0
        assert report.total_records == 100

    def test_tc003_empty_dataset_rejection(self):
        """TC-003: Validate that empty (0-record) dataset is rejected."""
        empty_df = pd.DataFrame()
        report = self.validator.validate(empty_df, "CICIDS2017")
        assert report.is_valid is False
        assert any("empty" in err.lower() for err in report.errors)

    def test_tc004_missing_mandatory_columns(self, sample_cicids2017_valid_df: pd.DataFrame):
        """TC-004: Validate detection of missing mandatory flow features."""
        corrupted = sample_cicids2017_valid_df.drop(columns=["Flow Duration", "Total Fwd Packets"])
        report = self.validator.validate(corrupted, "CICIDS2017")
        assert report.is_valid is False
        assert "Flow Duration" in report.missing_columns
        assert "Total Fwd Packets" in report.missing_columns
        assert any("Missing" in err for err in report.errors)

    def test_tc005_missing_label_column(self, sample_cicids2017_valid_df: pd.DataFrame):
        """Validate detection of missing target label column."""
        no_label = sample_cicids2017_valid_df.drop(columns=["Label"])
        report = self.validator.validate(no_label, "CICIDS2017")
        assert report.is_valid is False
        assert any("label" in err.lower() for err in report.errors)

    def test_tc006_infinite_values_detection(self, sample_cicids2017_dirty_df: pd.DataFrame):
        """TC-006a: Verify detection of +inf and -inf values."""
        report = self.validator.validate(sample_cicids2017_dirty_df, "CICIDS2017")
        assert "Flow Bytes/s" in report.infinite_value_counts
        assert "Flow Packets/s" in report.infinite_value_counts
        assert report.infinite_value_counts["Flow Bytes/s"] == 1
        assert report.infinite_value_counts["Flow Packets/s"] == 1

    def test_tc007_excessive_nan_threshold(self, sample_cicids2017_valid_df: pd.DataFrame):
        """TC-006b: Verify error when missing value ratio exceeds threshold."""
        df = sample_cicids2017_valid_df.copy()
        # Set 20% of values in Flow Duration to NaN (threshold is 5%)
        df.loc[:20, "Flow Duration"] = np.nan
        report = self.validator.validate(df, "CICIDS2017")
        assert report.is_valid is False
        assert any("exceeds maximum allowable missing ratio" in err for err in report.errors)

    def test_tc008_duplicate_record_detection(self, sample_cicids2017_dirty_df: pd.DataFrame):
        """Verify detection and counting of duplicated network flow records."""
        report = self.validator.validate(sample_cicids2017_dirty_df, "CICIDS2017")
        assert report.duplicate_record_count == 5
        assert any("duplicate" in warn.lower() for warn in report.warnings)

    def test_tc009_value_constraint_violations(self, sample_cicids2017_dirty_df: pd.DataFrame):
        """TC-005b: Verify constraint failure when non-negative features contain negative values."""
        report = self.validator.validate(sample_cicids2017_dirty_df, "CICIDS2017")
        assert "Flow Duration" in report.constraint_violations
        assert report.constraint_violations["Flow Duration"] == 1
        assert report.is_valid is False
        assert any("negative value" in err.lower() for err in report.errors)

    def test_tc010_leakage_identifier_columns(self, sample_cicids2017_dirty_df: pd.DataFrame):
        """Verify that non-generalizable network metadata (IPs, Ports, Timestamp) are flagged."""
        report = self.validator.validate(sample_cicids2017_dirty_df, "CICIDS2017")
        assert "Source IP" in report.detected_leakage_columns
        assert "Destination IP" in report.detected_leakage_columns
        assert "Source Port" in report.detected_leakage_columns
        assert "Timestamp" in report.detected_leakage_columns
        assert any("leakage" in warn.lower() for warn in report.warnings)

    def test_tc011_zero_variance_detection(self, sample_cicids2017_dirty_df: pd.DataFrame):
        """Verify detection of constant zero-variance features."""
        report = self.validator.validate(sample_cicids2017_dirty_df, "CICIDS2017")
        assert "Fwd PSH Flags" in report.zero_variance_columns
        assert any("zero-variance" in warn.lower() for warn in report.warnings)

    def test_tc012_unknown_label_warning(self, sample_cicids2017_valid_df: pd.DataFrame):
        """Verify warning when an uncataloged attack class is encountered."""
        df = sample_cicids2017_valid_df.copy()
        df.loc[0, "Label"] = "Ransomware_WannaCry_Unknown"
        report = self.validator.validate(df, "CICIDS2017")
        assert "Ransomware_WannaCry_Unknown" in report.invalid_labels
        assert any("uncataloged" in warn.lower() for warn in report.warnings)
