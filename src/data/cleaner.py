"""Data cleaning and sanitization pipeline for network flow datasets."""

from typing import List, Optional, Set, Dict, Any
import numpy as np
import pandas as pd

from src.data.schemas import LEAKAGE_COLUMNS, CIC_IOT2023_ATTACK_MAPPING


class DataCleaner:
    """Performs rigorous cleaning, sanitization, and leakage prevention on network flow datasets."""

    def __init__(
        self,
        drop_leakage: bool = True,
        drop_duplicates: bool = True,
        imputation_strategy: str = "median",
        drop_zero_variance: bool = True,
        consolidate_iot_labels: bool = True,
    ):
        """
        Args:
            drop_leakage: Whether to remove IP addresses, ports, timestamps, etc.
            drop_duplicates: Whether to remove duplicate flow entries.
            imputation_strategy: Strategy for handling missing/inf values ('median', 'mean', 'drop').
            drop_zero_variance: Whether to drop constant columns.
            consolidate_iot_labels: Whether to map 33 CIC-IoT2023 sub-classes to 8 parent classes.
        """
        self.drop_leakage = drop_leakage
        self.drop_duplicates = drop_duplicates
        self.imputation_strategy = imputation_strategy
        self.drop_zero_variance = drop_zero_variance
        self.consolidate_iot_labels = consolidate_iot_labels

        # State stored during fit to prevent data leakage across train/test splits
        self.imputation_values_: Dict[str, float] = {}
        self.zero_variance_cols_: List[str] = []
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame, label_col: Optional[str] = None) -> "DataCleaner":
        """
        Computes imputation statistics and identifies zero-variance features on training data ONLY.

        Args:
            df: Training DataFrame.
            label_col: Name of label column to exclude from numerical computation.

        Returns:
            self
        """
        clean_df = df.copy()
        clean_df.columns = [c.strip() for c in clean_df.columns]

        # 1. Temporarily replace inf with nan to calculate realistic stats
        numeric_cols = clean_df.select_dtypes(include=[np.number]).columns
        if label_col and label_col in numeric_cols:
            numeric_cols = numeric_cols.drop(label_col)

        for col in numeric_cols:
            clean_df[col] = clean_df[col].replace([np.inf, -np.inf], np.nan)
            if self.imputation_strategy == "median":
                val = float(clean_df[col].median(skipna=True))
            elif self.imputation_strategy == "mean":
                val = float(clean_df[col].mean(skipna=True))
            else:
                val = 0.0
            self.imputation_values_[col] = val

            # Check zero variance on training partition
            if self.drop_zero_variance and clean_df[col].nunique(dropna=True) <= 1:
                self.zero_variance_cols_.append(col)

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, label_col: Optional[str] = None) -> pd.DataFrame:
        """
        Applies cleaning, leakage prevention, and learned imputations to a DataFrame.

        Args:
            df: Input DataFrame.
            label_col: Name of label column.

        Returns:
            Cleaned and sanitized DataFrame.
        """
        cleaned = df.copy()
        # Clean column whitespace
        cleaned.columns = [c.strip() for c in cleaned.columns]

        # 1. Drop non-generalizable leakage columns
        if self.drop_leakage:
            cols_to_drop = [c for c in cleaned.columns if c in LEAKAGE_COLUMNS]
            if cols_to_drop:
                cleaned.drop(columns=cols_to_drop, inplace=True, errors="ignore")

        # 2. Drop duplicates
        if self.drop_duplicates:
            cleaned.drop_duplicates(inplace=True)

        # 3. Replace infinite values with NaN
        cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 4. Impute missing values
        if self.imputation_strategy in ["median", "mean"]:
            if self.is_fitted:
                # Use training-derived values
                cleaned.fillna(value=self.imputation_values_, inplace=True)
            else:
                # Fallback: compute on current batch
                for col in cleaned.select_dtypes(include=[np.number]).columns:
                    fill_val = cleaned[col].median() if self.imputation_strategy == "median" else cleaned[col].mean()
                    cleaned[col].fillna(fill_val if pd.notna(fill_val) else 0.0, inplace=True)
        elif self.imputation_strategy == "drop":
            cleaned.dropna(inplace=True)

        # 5. Drop zero-variance columns identified during fit
        if self.drop_zero_variance and self.zero_variance_cols_:
            cleaned.drop(columns=self.zero_variance_cols_, inplace=True, errors="ignore")

        # 6. Consolidate labels if applicable
        if self.consolidate_iot_labels and label_col and label_col in cleaned.columns:
            cleaned[label_col] = cleaned[label_col].astype(str).str.strip()
            cleaned[label_col] = cleaned[label_col].map(
                lambda x: CIC_IOT2023_ATTACK_MAPPING.get(x, x)
            )

        return cleaned

    def fit_transform(self, df: pd.DataFrame, label_col: Optional[str] = None) -> pd.DataFrame:
        """Convenience method to fit statistics and transform training data."""
        return self.fit(df, label_col=label_col).transform(df, label_col=label_col)
