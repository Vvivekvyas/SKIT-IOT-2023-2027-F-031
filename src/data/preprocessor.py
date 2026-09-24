"""End-to-end data preprocessing pipeline with leakage prevention, scaling, and encoding."""

from typing import Tuple, Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd

from src.data.cleaner import DataCleaner
from src.data.schemas import DATASET_CONFIGS


class DataPreprocessor:
    """
    Coordinates data cleaning, leakage removal, scaling, label encoding, and stratified splitting.
    Strictly enforces zero data leakage by fitting parameters on training data only.
    """

    def __init__(
        self,
        dataset_name: str,
        scaler_type: str = "minmax",
        imputation_strategy: str = "median",
        drop_leakage: bool = True,
        random_state: int = 42,
    ):
        """
        Args:
            dataset_name: 'CICIDS2017' or 'CIC-IoT2023'.
            scaler_type: 'minmax', 'robust', or 'standard'.
            imputation_strategy: 'median' or 'mean'.
            drop_leakage: Whether to strip identity columns.
            random_state: Random seed for reproducible splits.
        """
        self.dataset_name = dataset_name
        self.scaler_type = scaler_type.lower()
        self.imputation_strategy = imputation_strategy
        self.drop_leakage = drop_leakage
        self.random_state = random_state

        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_CONFIGS.keys())}")

        self.config = DATASET_CONFIGS[dataset_name]
        self.label_col = self.config["label_column"]

        self.cleaner = DataCleaner(
            drop_leakage=drop_leakage,
            drop_duplicates=True,
            imputation_strategy=imputation_strategy,
            drop_zero_variance=True,
            consolidate_iot_labels=(dataset_name == "CIC-IoT2023"),
        )

        # Fitted parameters
        self.feature_names_: List[str] = []
        self.scale_params_: Dict[str, Dict[str, float]] = {}
        self.class_to_idx_: Dict[str, int] = {}
        self.idx_to_class_: Dict[int, str] = {}
        self.is_fitted: bool = False

    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Performs stratified train/val/test split across target label classes.

        Args:
            df: Cleaned or raw DataFrame.
            train_ratio: Proportion of training data (e.g., 0.70).
            val_ratio: Proportion of validation data (e.g., 0.15).
            test_ratio: Proportion of test data (e.g., 0.15).

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got: {total}")

        clean_cols = {c.strip(): c for c in df.columns}
        actual_label = clean_cols.get(self.label_col, self.label_col)

        np.random.seed(self.random_state)

        train_dfs = []
        val_dfs = []
        test_dfs = []

        # Stratified sampling per class
        for label_val, group in df.groupby(actual_label):
            shuffled = group.sample(frac=1.0, random_state=self.random_state)
            n = len(shuffled)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_dfs.append(shuffled.iloc[:n_train])
            val_dfs.append(shuffled.iloc[n_train:n_train + n_val])
            test_dfs.append(shuffled.iloc[n_train + n_val:])

        train_df = pd.concat(train_dfs).sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)
        val_df = pd.concat(val_dfs).sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)
        test_df = pd.concat(test_dfs).sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)

        return train_df, val_df, test_df

    def fit(self, train_df: pd.DataFrame) -> "DataPreprocessor":
        """
        Learns scaling statistics and label vocabulary strictly from training data.

        Args:
            train_df: Raw training DataFrame.

        Returns:
            self
        """
        # Fit cleaner first
        clean_train = self.cleaner.fit_transform(train_df, label_col=self.label_col)

        clean_cols = {c.strip(): c for c in clean_train.columns}
        actual_label = clean_cols[self.label_col]

        # Extract features and targets
        feature_cols = [c for c in clean_train.columns if c != actual_label]
        self.feature_names_ = feature_cols

        # Compute scaling parameters strictly on training features
        for col in feature_cols:
            series = clean_train[col].astype(float)
            if self.scaler_type == "minmax":
                c_min = float(series.min())
                c_max = float(series.max())
                diff = c_max - c_min if c_max != c_min else 1.0
                self.scale_params_[col] = {"min": c_min, "diff": diff}
            elif self.scaler_type == "standard":
                mean = float(series.mean())
                std = float(series.std()) if series.std() != 0 else 1.0
                self.scale_params_[col] = {"mean": mean, "std": std}
            elif self.scaler_type == "robust":
                median = float(series.median())
                q25 = float(series.quantile(0.25))
                q75 = float(series.quantile(0.75))
                iqr = q75 - q25 if q75 != q25 else 1.0
                self.scale_params_[col] = {"median": median, "iqr": iqr}
            else:
                raise ValueError(f"Unsupported scaler '{self.scaler_type}'. Choose minmax, standard, or robust.")

        # Build label encoding mapping
        unique_labels = sorted(clean_train[actual_label].astype(str).unique())
        self.class_to_idx_ = {cls: idx for idx, cls in enumerate(unique_labels)}
        self.idx_to_class_ = {idx: cls for cls, idx in self.class_to_idx_.items()}

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms a dataset using training-fitted parameters.

        Args:
            df: Input DataFrame (validation or test split).

        Returns:
            Tuple of (X_features, y_labels) as numpy arrays.
        """
        if not self.is_fitted:
            raise RuntimeError("DataPreprocessor must be fitted on training data before calling transform.")

        clean_df = self.cleaner.transform(df, label_col=self.label_col)

        clean_cols = {c.strip(): c for c in clean_df.columns}
        actual_label = clean_cols.get(self.label_col, self.label_col)

        # Scale features
        scaled_features = np.zeros((len(clean_df), len(self.feature_names_)), dtype=np.float32)

        for i, col in enumerate(self.feature_names_):
            if col in clean_df.columns:
                series = clean_df[col].astype(float).values
                params = self.scale_params_[col]
                if self.scaler_type == "minmax":
                    scaled = (series - params["min"]) / params["diff"]
                    scaled = np.clip(scaled, 0.0, 1.0)
                elif self.scaler_type == "standard":
                    scaled = (series - params["mean"]) / params["std"]
                elif self.scaler_type == "robust":
                    scaled = (series - params["median"]) / params["iqr"]
                scaled_features[:, i] = scaled
            else:
                # If feature missing in test sample, default to 0
                scaled_features[:, i] = 0.0

        # Encode labels
        if actual_label in clean_df.columns:
            labels_str = clean_df[actual_label].astype(str).values
            encoded_labels = np.array(
                [self.class_to_idx_.get(lbl, -1) for lbl in labels_str],
                dtype=np.int64
            )
        else:
            encoded_labels = np.full(len(clean_df), -1, dtype=np.int64)

        return scaled_features, encoded_labels

    def fit_transform(self, train_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fits on train_df and returns scaled features and encoded labels."""
        return self.fit(train_df).transform(train_df)
