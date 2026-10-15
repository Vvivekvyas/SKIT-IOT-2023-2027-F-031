"""End-to-end Data Pipeline Orchestrator for Network Flow Datasets.

Coordinates validation, sanitization, leakage elimination, scaling, stratified partitioning,
streaming batch generation, and artifact serialization.
"""

from dataclasses import dataclass, field
import json
import os
from typing import Dict, Generator, List, Optional, Tuple, Union, Any

import joblib
import numpy as np
import pandas as pd

from src.data.cleaner import DataCleaner
from src.data.preprocessor import DataPreprocessor
from src.data.schemas import DATASET_CONFIGS, LEAKAGE_COLUMNS
from src.data.validator import DataValidator, ValidationReport


@dataclass
class ProcessedDataset:
    """Container for processed training, validation, and test datasets with target encodings."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    y_train_binary: np.ndarray
    y_val_binary: np.ndarray
    y_test_binary: np.ndarray
    feature_names: List[str] = field(default_factory=list)
    class_mapping: Dict[str, int] = field(default_factory=dict)
    inverse_class_mapping: Dict[int, str] = field(default_factory=dict)

    @property
    def train_df(self) -> pd.DataFrame:
        """Returns X_train as a pandas DataFrame with original feature names."""
        return pd.DataFrame(self.X_train, columns=self.feature_names)

    @property
    def val_df(self) -> pd.DataFrame:
        """Returns X_val as a pandas DataFrame with original feature names."""
        return pd.DataFrame(self.X_val, columns=self.feature_names)

    @property
    def test_df(self) -> pd.DataFrame:
        """Returns X_test as a pandas DataFrame with original feature names."""
        return pd.DataFrame(self.X_test, columns=self.feature_names)


class DataPipeline:
    """
    Production-grade Data Pipeline orchestrating the end-to-end lifecycle
    of cybersecurity network flow data for intrusion detection models.
    """

    def __init__(
        self,
        dataset_name: str = "CICIDS2017",
        scaler_type: str = "minmax",
        imputation_strategy: str = "median",
        drop_leakage: bool = True,
        strict_validation: bool = True,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42,
    ):
        """
        Args:
            dataset_name: 'CICIDS2017' or 'CIC-IoT2023'.
            scaler_type: 'minmax', 'standard', or 'robust'.
            imputation_strategy: 'median' or 'mean'.
            drop_leakage: Whether to strip identity/socket metadata columns.
            strict_validation: Whether to run DataValidator before processing.
            train_ratio: Proportion of training data.
            val_ratio: Proportion of validation data.
            test_ratio: Proportion of test data.
            random_state: Random seed for deterministic reproducibility.
        """
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset: '{dataset_name}'. Available: {list(DATASET_CONFIGS.keys())}")

        self.dataset_name = dataset_name
        self.scaler_type = scaler_type.lower()
        self.imputation_strategy = imputation_strategy
        self.drop_leakage = drop_leakage
        self.strict_validation = strict_validation
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

        self.config = DATASET_CONFIGS[dataset_name]
        self.label_col = self.config["label_column"]

        self.validator = DataValidator()
        self.preprocessor = DataPreprocessor(
            dataset_name=dataset_name,
            scaler_type=scaler_type,
            imputation_strategy=imputation_strategy,
            drop_leakage=drop_leakage,
            random_state=random_state,
        )

        self.is_fitted: bool = False
        self.last_validation_report: Optional[ValidationReport] = None

    def validate_dataset(self, df: pd.DataFrame) -> ValidationReport:
        """Runs the validation engine on a given dataset DataFrame."""
        report = self.validator.validate(df, self.dataset_name)
        self.last_validation_report = report
        return report

    def _create_binary_targets(self, y_raw: pd.Series) -> np.ndarray:
        """Encodes string or numeric labels into binary targets (0=Benign, 1=Attack)."""
        labels_str = y_raw.astype(str).str.strip().str.lower()
        benign_tokens = {"benign", "normal", "benigntraffic", "background", "0"}
        is_attack = ~labels_str.isin(benign_tokens)
        return is_attack.astype(int).to_numpy()

    def fit(self, df: pd.DataFrame) -> "DataPipeline":
        """
        Fits the data pipeline (imputers, scalers, encoders) strictly on the provided DataFrame.
        Enforces zero data leakage.
        """
        if self.strict_validation:
            report = self.validate_dataset(df)
            if not report.is_valid:
                errors = "; ".join(report.errors)
                raise ValueError(f"Dataset validation failed: {errors}")

        self.preprocessor.fit(df)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transforms a DataFrame using fitted pipeline parameters.

        Returns:
            Tuple of (X_scaled_array, y_multiclass_array, y_binary_array).
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform can be called.")

        X, y = self.preprocessor.transform(df)

        clean_df = self.preprocessor.cleaner.transform(df, label_col=self.label_col)
        clean_cols = {c.strip(): c for c in clean_df.columns}
        actual_label = clean_cols.get(self.label_col, self.label_col)

        if actual_label in clean_df.columns:
            y_binary = self._create_binary_targets(clean_df[actual_label])
        else:
            y_binary = np.zeros(len(X), dtype=int)

        return X, y, y_binary

    def fit_transform(self, df: pd.DataFrame) -> ProcessedDataset:
        """
        Executes end-to-end pipeline:
        1. Validates dataset integrity (if strict_validation=True).
        2. Performs stratified splitting into train, val, and test subsets.
        3. Fits cleaner, scaler, and encoder on training partition ONLY.
        4. Transforms train, val, and test partitions with zero leakage.
        5. Computes binary target representations.
        6. Packages results into a ProcessedDataset container.
        """
        if self.strict_validation:
            report = self.validate_dataset(df)
            if not report.is_valid:
                errors = "; ".join(report.errors)
                raise ValueError(f"Dataset validation failed: {errors}")

        train_df, val_df, test_df = self.preprocessor.split_data(
            df,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
        )

        # Fit strictly on train fold
        self.preprocessor.fit(train_df)
        self.is_fitted = True

        X_train, y_train, y_train_binary = self.transform(train_df)
        X_val, y_val, y_val_binary = self.transform(val_df)
        X_test, y_test, y_test_binary = self.transform(test_df)

        return ProcessedDataset(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            y_train_binary=y_train_binary,
            y_val_binary=y_val_binary,
            y_test_binary=y_test_binary,
            feature_names=self.preprocessor.feature_names_,
            class_mapping=self.preprocessor.class_to_idx_,
            inverse_class_mapping=self.preprocessor.idx_to_class_,
        )

    def process_stream(
        self,
        df: pd.DataFrame,
        chunk_size: int = 32,
    ) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Streams flow records in micro-batches through the pipeline.
        Enables low-latency streaming evaluation and batch inference testing.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before process_stream can be called.")

        n_samples = len(df)
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk = df.iloc[start_idx:end_idx]
            X_chunk, y_chunk, y_bin_chunk = self.transform(chunk)
            yield X_chunk, y_chunk, y_bin_chunk

    def export_artifacts(self, output_dir: str) -> Dict[str, str]:
        """
        Serializes pipeline parameters, scaling bounds, label mappings, and feature schema contract.

        Returns:
            Dictionary of saved artifact filepaths.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot export unfitted pipeline artifacts.")

        os.makedirs(output_dir, exist_ok=True)

        feature_schema = {
            "dataset_name": self.dataset_name,
            "feature_columns": self.preprocessor.feature_names_,
            "num_features": len(self.preprocessor.feature_names_),
            "label_column": self.label_col,
            "scaler_type": self.scaler_type,
            "scale_params": self.preprocessor.scale_params_,
            "class_to_idx": self.preprocessor.class_to_idx_,
            "idx_to_class": {str(k): v for k, v in self.preprocessor.idx_to_class_.items()},
            "num_classes": len(self.preprocessor.class_to_idx_),
            "imputation_strategy": self.imputation_strategy,
            "drop_leakage": self.drop_leakage,
            "random_state": self.random_state,
        }

        schema_path = os.path.join(output_dir, "feature_columns.json")
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(feature_schema, f, indent=2)

        preprocessor_path = os.path.join(output_dir, "preprocessor.joblib")
        joblib.dump(self.preprocessor, preprocessor_path)

        cleaner_path = os.path.join(output_dir, "cleaner.joblib")
        joblib.dump(self.preprocessor.cleaner, cleaner_path)

        return {
            "schema_json": schema_path,
            "preprocessor_joblib": preprocessor_path,
            "cleaner_joblib": cleaner_path,
        }

    @classmethod
    def load_artifacts(cls, artifact_dir: str) -> "DataPipeline":
        """
        Instantiates a DataPipeline instance from serialized disk artifacts.
        """
        schema_path = os.path.join(artifact_dir, "feature_columns.json")
        preprocessor_path = os.path.join(artifact_dir, "preprocessor.joblib")

        if not os.path.exists(schema_path) or not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Missing required artifacts in directory: {artifact_dir}")

        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        preprocessor = joblib.load(preprocessor_path)

        pipeline = cls(
            dataset_name=schema["dataset_name"],
            scaler_type=schema.get("scaler_type", "minmax"),
            imputation_strategy=schema.get("imputation_strategy", "median"),
            drop_leakage=schema.get("drop_leakage", True),
            strict_validation=False,
            random_state=schema.get("random_state", 42),
        )

        pipeline.preprocessor = preprocessor
        pipeline.is_fitted = True

        return pipeline
