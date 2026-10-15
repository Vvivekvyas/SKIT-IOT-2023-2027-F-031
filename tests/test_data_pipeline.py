"""Comprehensive integration and unit tests for DataPipeline (Week 6 - Ronak).

Verifies end-to-end data pipeline lifecycle:
- Schema validation
- Data sanitization & leakage elimination
- Feature scaling bounds & stratified partitioning
- Binary & multiclass label encoding
- Micro-batch streaming & memory optimization
- Artifact serialization & backend schema contracts
"""

import json
import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.data.pipeline import DataPipeline, ProcessedDataset
from src.data.schemas import LEAKAGE_COLUMNS


class TestDataPipeline:
    """Verifies end-to-end pipeline execution, streaming, serialization, and leakage isolation."""

    def test_pipeline_initialization(self):
        """TC-023: Verify pipeline initializes properly with valid configs and rejects invalid datasets."""
        pipeline = DataPipeline(dataset_name="CICIDS2017", scaler_type="minmax")
        assert pipeline.dataset_name == "CICIDS2017"
        assert pipeline.scaler_type == "minmax"
        assert pipeline.is_fitted is False

        with pytest.raises(ValueError, match="Unsupported dataset"):
            DataPipeline(dataset_name="UnknownDataset")

    def test_pipeline_fit_transform_cicids2017(self, sample_cicids2017_valid_df: pd.DataFrame):
        """TC-024: Verify complete end-to-end fit_transform flow on CICIDS2017."""
        pipeline = DataPipeline(dataset_name="CICIDS2017", scaler_type="minmax")
        dataset: ProcessedDataset = pipeline.fit_transform(sample_cicids2017_valid_df)

        assert pipeline.is_fitted is True
        assert isinstance(dataset, ProcessedDataset)

        # Split assertions (69 train, 14 val, 17 test)
        assert len(dataset.X_train) == 69
        assert len(dataset.X_val) == 14
        assert len(dataset.X_test) == 17
        assert len(dataset.y_train) == 69
        assert len(dataset.y_val) == 14
        assert len(dataset.y_test) == 17

        # Binary targets
        assert len(dataset.y_train_binary) == 69
        assert set(np.unique(dataset.y_train_binary)).issubset({0, 1})

        # Feature scaling bounds strictly in [0.0, 1.0]
        assert dataset.X_train.min() >= 0.0
        assert dataset.X_train.max() <= 1.0
        assert dataset.X_val.min() >= 0.0
        assert dataset.X_val.max() <= 1.0
        assert dataset.X_test.min() >= 0.0
        assert dataset.X_test.max() <= 1.0

        # Zero NaNs / infs
        assert not np.isnan(dataset.X_train).any()
        assert not np.isinf(dataset.X_train).any()

        # Class mapping invertible
        for cls_name, idx in dataset.class_mapping.items():
            assert dataset.inverse_class_mapping[idx] == cls_name

        # Property accessors check
        assert isinstance(dataset.train_df, pd.DataFrame)
        assert len(dataset.train_df) == 69
        assert list(dataset.train_df.columns) == dataset.feature_names

    def test_pipeline_handles_dirty_data_with_sanitization(
        self, sample_cicids2017_dirty_df: pd.DataFrame
    ):
        """TC-025: Verify pipeline sanitizes dirty data (infs, NaNs, duplicates, leakage cols)."""
        pipeline = DataPipeline(
            dataset_name="CICIDS2017",
            scaler_type="minmax",
            drop_leakage=True,
            strict_validation=False,
        )
        dataset = pipeline.fit_transform(sample_cicids2017_dirty_df)

        # Leakage columns must be pruned from features
        for col in LEAKAGE_COLUMNS:
            assert col not in dataset.feature_names

        # Check bounded values and zero NaNs/infs
        assert not np.isnan(dataset.X_train).any()
        assert not np.isinf(dataset.X_train).any()
        assert dataset.X_train.min() >= 0.0
        assert dataset.X_train.max() <= 1.0

    def test_pipeline_strict_validation_rejection(self, sample_cicids2017_valid_df: pd.DataFrame):
        """TC-026: Verify strict_validation halts pipeline on missing mandatory features."""
        corrupt_df = sample_cicids2017_valid_df.drop(columns=["Flow Duration"])
        pipeline = DataPipeline(dataset_name="CICIDS2017", strict_validation=True)

        with pytest.raises(ValueError, match="Dataset validation failed"):
            pipeline.fit(corrupt_df)

    def test_pipeline_train_test_leakage_isolation(self, sample_cicids2017_valid_df: pd.DataFrame):
        """TC-027: Verify pipeline parameters are isolated to training data only."""
        pipeline = DataPipeline(dataset_name="CICIDS2017")
        dataset = pipeline.fit_transform(sample_cicids2017_valid_df)

        # Capture scale params fitted on train fold
        initial_params = dict(pipeline.preprocessor.scale_params_)

        # Transform test set multiple times
        train_df, val_df, test_df = pipeline.preprocessor.split_data(sample_cicids2017_valid_df)
        pipeline.transform(test_df)
        pipeline.transform(test_df)

        # Scale parameters must remain completely unchanged
        assert pipeline.preprocessor.scale_params_ == initial_params

    def test_pipeline_batch_streaming_generator(self, sample_cicids2017_valid_df: pd.DataFrame):
        """TC-028: Verify micro-batch streaming yields consistent chunked transformations."""
        pipeline = DataPipeline(dataset_name="CICIDS2017")
        pipeline.fit(sample_cicids2017_valid_df)

        # Batch transform full dataframe
        full_X, full_y, full_y_bin = pipeline.transform(sample_cicids2017_valid_df)

        # Stream in micro-batches of size 20
        chunk_size = 20
        stream_X_list = []
        stream_y_list = []
        stream_y_bin_list = []

        for X_chunk, y_chunk, y_bin_chunk in pipeline.process_stream(
            sample_cicids2017_valid_df, chunk_size=chunk_size
        ):
            assert len(X_chunk) <= chunk_size
            stream_X_list.append(X_chunk)
            stream_y_list.append(y_chunk)
            stream_y_bin_list.append(y_bin_chunk)

        stream_X = np.vstack(stream_X_list)
        stream_y = np.concatenate(stream_y_list)
        stream_y_bin = np.concatenate(stream_y_bin_list)

        np.testing.assert_allclose(full_X, stream_X, rtol=1e-5)
        np.testing.assert_array_equal(full_y, stream_y)
        np.testing.assert_array_equal(full_y_bin, stream_y_bin)

    def test_pipeline_artifact_serialization_roundtrip(
        self, sample_cicids2017_valid_df: pd.DataFrame
    ):
        """TC-029: Verify pipeline serialization to disk and reload produces identical results."""
        pipeline = DataPipeline(dataset_name="CICIDS2017")
        pipeline.fit(sample_cicids2017_valid_df)
        X_orig, y_orig, y_bin_orig = pipeline.transform(sample_cicids2017_valid_df)

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_paths = pipeline.export_artifacts(tmp_dir)
            assert os.path.exists(artifact_paths["schema_json"])
            assert os.path.exists(artifact_paths["preprocessor_joblib"])
            assert os.path.exists(artifact_paths["cleaner_joblib"])

            # Reload pipeline from disk artifacts
            reloaded_pipeline = DataPipeline.load_artifacts(tmp_dir)
            assert reloaded_pipeline.is_fitted is True

            X_reloaded, y_reloaded, y_bin_reloaded = reloaded_pipeline.transform(
                sample_cicids2017_valid_df
            )

            np.testing.assert_allclose(X_orig, X_reloaded, rtol=1e-5)
            np.testing.assert_array_equal(y_orig, y_reloaded)
            np.testing.assert_array_equal(y_bin_orig, y_bin_reloaded)

    def test_pipeline_ciciot2023_support(self, sample_ciciot2023_valid_df: pd.DataFrame):
        """TC-030: Verify pipeline operates cleanly on CIC-IoT2023 dataset with label consolidation."""
        pipeline = DataPipeline(dataset_name="CIC-IoT2023", scaler_type="minmax")
        dataset = pipeline.fit_transform(sample_ciciot2023_valid_df)

        assert pipeline.is_fitted is True
        assert len(dataset.X_train) == 69
        assert len(dataset.X_val) == 13
        assert len(dataset.X_test) == 18

        # Check consolidated labels (DDoS-SYN_Flood -> DDoS, Mirai-greeth_flood -> Mirai)
        assert "DDoS" in dataset.class_mapping
        assert "Mirai" in dataset.class_mapping
        assert "DDoS-SYN_Flood" not in dataset.class_mapping
        assert "Mirai-greeth_flood" not in dataset.class_mapping

        # Binary mapping: BenignTraffic -> 0, attacks -> 1
        assert dataset.y_train_binary.sum() > 0

    def test_pipeline_unfitted_transform_raises(self, sample_cicids2017_valid_df: pd.DataFrame):
        """TC-031: Verify calling transform or process_stream before fit raises RuntimeError."""
        pipeline = DataPipeline(dataset_name="CICIDS2017")
        with pytest.raises(RuntimeError, match="must be fitted"):
            pipeline.transform(sample_cicids2017_valid_df)

        with pytest.raises(RuntimeError, match="must be fitted"):
            list(pipeline.process_stream(sample_cicids2017_valid_df))

    def test_pipeline_schema_metadata_json_contract(
        self, sample_cicids2017_valid_df: pd.DataFrame
    ):
        """TC-032: Verify feature_columns.json schema contract structure for backend serving."""
        pipeline = DataPipeline(dataset_name="CICIDS2017")
        pipeline.fit(sample_cicids2017_valid_df)

        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = pipeline.export_artifacts(tmp_dir)
            with open(paths["schema_json"], "r", encoding="utf-8") as f:
                schema = json.load(f)

            required_keys = [
                "dataset_name",
                "feature_columns",
                "num_features",
                "label_column",
                "scaler_type",
                "scale_params",
                "class_to_idx",
                "idx_to_class",
                "num_classes",
            ]
            for key in required_keys:
                assert key in schema

            assert schema["num_features"] == len(schema["feature_columns"])
            assert schema["num_classes"] == len(schema["class_to_idx"])

    @pytest.mark.parametrize("scaler_type", ["standard", "robust"])
    def test_pipeline_alternative_scalers(
        self, sample_cicids2017_valid_df: pd.DataFrame, scaler_type: str
    ):
        """TC-033: Verify pipeline execution with standard and robust scalers."""
        pipeline = DataPipeline(dataset_name="CICIDS2017", scaler_type=scaler_type)
        dataset = pipeline.fit_transform(sample_cicids2017_valid_df)
        assert pipeline.is_fitted is True
        assert not np.isnan(dataset.X_train).any()
        assert not np.isinf(dataset.X_train).any()
