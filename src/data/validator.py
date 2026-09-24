"""Comprehensive dataset validation engine for CICIDS2017 and CIC-IoT2023."""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional
import numpy as np
import pandas as pd

from src.data.schemas import DATASET_CONFIGS, LEAKAGE_COLUMNS


@dataclass
class ValidationReport:
    """Stores the structured results of a dataset validation run."""
    dataset_name: str
    total_records: int = 0
    total_features: int = 0
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_columns: List[str] = field(default_factory=list)
    detected_leakage_columns: List[str] = field(default_factory=list)
    infinite_value_counts: Dict[str, int] = field(default_factory=dict)
    nan_value_counts: Dict[str, int] = field(default_factory=dict)
    duplicate_record_count: int = 0
    zero_variance_columns: List[str] = field(default_factory=list)
    invalid_labels: List[str] = field(default_factory=list)
    constraint_violations: Dict[str, int] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation report to a serializable dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "total_records": self.total_records,
            "total_features": self.total_features,
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "missing_columns": self.missing_columns,
            "detected_leakage_columns": self.detected_leakage_columns,
            "total_infinite_values": sum(self.infinite_value_counts.values()),
            "total_nan_values": sum(self.nan_value_counts.values()),
            "duplicate_record_count": self.duplicate_record_count,
            "zero_variance_columns": self.zero_variance_columns,
            "invalid_labels": self.invalid_labels,
            "constraint_violations": self.constraint_violations,
            "summary": self.summary,
        }


class DatasetValidator:
    """Validates raw and ingested network flow datasets against strict quality standards."""

    def __init__(self, max_nan_ratio: float = 0.05):
        """
        Args:
            max_nan_ratio: Maximum allowable proportion of missing values before throwing an error.
        """
        self.max_nan_ratio = max_nan_ratio

    def validate(self, df: pd.DataFrame, dataset_name: str) -> ValidationReport:
        """
        Executes end-to-end validation on a dataset.

        Args:
            df: The pandas DataFrame to validate.
            dataset_name: Name of the dataset ("CICIDS2017" or "CIC-IoT2023").

        Returns:
            ValidationReport object detailing all checks and anomalies.
        """
        report = ValidationReport(dataset_name=dataset_name)

        if df is None or not isinstance(df, pd.DataFrame):
            report.is_valid = False
            report.errors.append("Invalid input: DataFrame is None or not a pandas DataFrame.")
            return report

        report.total_records = len(df)
        report.total_features = len(df.columns)

        if df.empty:
            report.is_valid = False
            report.errors.append("Dataset is empty (0 records).")
            return report

        if dataset_name not in DATASET_CONFIGS:
            report.is_valid = False
            report.errors.append(f"Unknown dataset '{dataset_name}'. Supported: {list(DATASET_CONFIGS.keys())}")
            return report

        config = DATASET_CONFIGS[dataset_name]

        # 1. Schema Validation (Columns & Mandatory Features)
        self._check_schema(df, config, report)

        # 2. Leakage Column Identification
        self._check_leakage(df, report)

        # 3. Missing and Infinite Values
        self._check_nan_and_inf(df, report)

        # 4. Duplicate Records Check
        self._check_duplicates(df, report)

        # 5. Value Constraint Checks (e.g., Non-negative durations, rates)
        self._check_constraints(df, config, report)

        # 6. Target Label Validation
        self._check_labels(df, config, report)

        # 7. Zero-Variance Columns
        self._check_zero_variance(df, report)

        # Final validity determination
        report.is_valid = len(report.errors) == 0

        report.summary = {
            "valid_records_ratio": 1.0 - (report.duplicate_record_count / report.total_records) if report.total_records else 0,
            "clean_features_count": report.total_features - len(report.zero_variance_columns) - len(report.detected_leakage_columns),
        }

        return report

    def _check_schema(self, df: pd.DataFrame, config: Dict[str, Any], report: ValidationReport) -> None:
        """Verify presence of mandatory features and target label column."""
        # Strip potential leading/trailing whitespace from column names
        df_cols = set(c.strip() for c in df.columns)
        mandatory = config["mandatory_features"]
        label_col = config["label_column"]

        missing_mand = [col for col in mandatory if col.strip() not in df_cols]
        if missing_mand:
            report.missing_columns.extend(missing_mand)
            report.errors.append(f"Missing {len(missing_mand)} mandatory feature(s): {missing_mand[:5]}...")

        if label_col.strip() not in df_cols:
            report.errors.append(f"Missing required label column: '{label_col}'.")

    def _check_leakage(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Check for presence of non-generalizable network identifiers."""
        df_cols = set(c.strip() for c in df.columns)
        detected = [c for c in df_cols if c in LEAKAGE_COLUMNS]
        if detected:
            report.detected_leakage_columns.extend(detected)
            report.warnings.append(
                f"Detected {len(detected)} identifier/leakage column(s) that must be dropped: {detected}"
            )

    def _check_nan_and_inf(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Detect NaN, null, +inf, and -inf values across numerical columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            series = df[col]
            # Infinite values check
            inf_count = int(np.isinf(series).sum())
            if inf_count > 0:
                report.infinite_value_counts[col] = inf_count

            # NaN values check
            nan_count = int(series.isna().sum())
            if nan_count > 0:
                report.nan_value_counts[col] = nan_count
                nan_ratio = nan_count / len(df)
                if nan_ratio > self.max_nan_ratio:
                    report.errors.append(
                        f"Column '{col}' exceeds maximum allowable missing ratio ({nan_ratio:.2%} > {self.max_nan_ratio:.2%})."
                    )

        if report.infinite_value_counts:
            report.warnings.append(
                f"Found infinite values in {len(report.infinite_value_counts)} column(s). Total: {sum(report.infinite_value_counts.values())}."
            )

    def _check_duplicates(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Count identical duplicate records."""
        num_duplicates = int(df.duplicated().sum())
        report.duplicate_record_count = num_duplicates
        if num_duplicates > 0:
            dup_ratio = num_duplicates / len(df)
            report.warnings.append(
                f"Found {num_duplicates} duplicate records ({dup_ratio:.2%}) in dataset."
            )

    def _check_constraints(self, df: pd.DataFrame, config: Dict[str, Any], report: ValidationReport) -> None:
        """Check domain-specific boundaries (e.g. durations and packet counts must be >= 0)."""
        non_negative_cols = config.get("non_negative_columns", [])
        clean_cols = {c.strip(): c for c in df.columns}

        for col in non_negative_cols:
            if col in clean_cols:
                actual_col = clean_cols[col]
                # Filter out inf and nan for numeric constraint validation
                numeric_vals = pd.to_numeric(df[actual_col], errors="coerce").dropna()
                neg_count = int((numeric_vals < 0).sum())
                if neg_count > 0:
                    report.constraint_violations[col] = neg_count
                    report.errors.append(
                        f"Constraint violation: Column '{col}' contains {neg_count} negative value(s)."
                    )

    def _check_labels(self, df: pd.DataFrame, config: Dict[str, Any], report: ValidationReport) -> None:
        """Validate that all target labels belong to the expected attack taxonomy."""
        label_col = config["label_column"]
        clean_cols = {c.strip(): c for c in df.columns}

        if label_col in clean_cols:
            actual_col = clean_cols[label_col]
            unique_labels = set(df[actual_col].dropna().astype(str).unique())
            valid_labels = config["valid_labels"]

            invalid = [l for l in unique_labels if l not in valid_labels]
            if invalid:
                report.invalid_labels.extend(invalid)
                report.warnings.append(
                    f"Found {len(invalid)} uncataloged label(s) in target column: {invalid[:5]}"
                )

    def _check_zero_variance(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Identify constant/zero-variance features that provide no discriminatory signal."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        zero_var = []
        for col in numeric_cols:
            # Check unique values count ignoring NaN
            if df[col].nunique(dropna=True) <= 1:
                zero_var.append(col)

        if zero_var:
            report.zero_variance_columns.extend(zero_var)
            report.warnings.append(
                f"Found {len(zero_var)} zero-variance / constant feature(s): {zero_var[:5]}..."
            )
