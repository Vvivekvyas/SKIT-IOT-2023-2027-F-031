"""Integration and end-to-end service tests for DataPipeline & Dataset Ingestion (Week 8 - Ronak).

Verifies the integration between the ML data pipeline (`src/data`) and the backend
dataset ingestion service (`backend/app/services/data_pipeline.py`):
- Auto-detection of dataset types (CICIDS2017 vs. CIC-IoT2023 vs. unknown schema)
- Schema validation via backend service interface (`run_validation`)
- Sampling mechanics for high-volume dataset uploads
- Graceful degradation under missing pipeline components (`PipelineUnavailable`)
- Seamless transition from ingested CSV to ML `DataPipeline` fit/transform
- Micro-batch streaming throughput and memory stability
"""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.core.config import settings
from app.services.data_pipeline import (
    MIN_COLUMN_MATCH,
    PipelineUnavailable,
    _load,
    _repo_root,
    detect_dataset_type,
    run_validation,
)
from src.data.pipeline import DataPipeline, ProcessedDataset
from src.data.schemas import (
    CICIDS2017_MANDATORY_NUMERICAL_FEATURES,
    CIC_IOT2023_MANDATORY_NUMERICAL_FEATURES,
    DATASET_CONFIGS,
)


class TestDataPipelineIntegration:
    """Week 8 Integration test suite for ML Data Pipeline & Backend Ingestion Service."""

    def test_tc_041_detect_dataset_type_cicids2017(self):
        """TC-041: Detect CICIDS2017 from CSV header columns exceeding MIN_COLUMN_MATCH (50%)."""
        cols = list(CICIDS2017_MANDATORY_NUMERICAL_FEATURES[:50]) + ["Label", "Source IP", "Destination IP"]
        detected = detect_dataset_type(cols, DATASET_CONFIGS)
        assert detected == "CICIDS2017"

    def test_tc_042_detect_dataset_type_ciciot2023(self):
        """TC-042: Detect CIC-IoT2023 from IoT feature columns exceeding MIN_COLUMN_MATCH (50%)."""
        cols = list(CIC_IOT2023_MANDATORY_NUMERICAL_FEATURES[:35]) + ["label", "Header_Length"]
        detected = detect_dataset_type(cols, DATASET_CONFIGS)
        assert detected == "CIC-IoT2023"

    def test_tc_043_detect_dataset_type_unknown_columns(self):
        """TC-043: Return None when columns do not match any known dataset schema."""
        unknown_cols = ["employee_id", "salary", "department", "hire_date", "performance_score"]
        detected = detect_dataset_type(unknown_cols, DATASET_CONFIGS)
        assert detected is None

        # Partial columns under threshold (< 50%)
        insufficient_cols = list(CICIDS2017_MANDATORY_NUMERICAL_FEATURES[:5]) + ["random_a", "random_b"]
        detected_insufficient = detect_dataset_type(insufficient_cols, DATASET_CONFIGS)
        assert detected_insufficient is None

    def test_tc_044_run_validation_valid_cicids2017(self, sample_cicids2017_valid_df: pd.DataFrame):
        """TC-044: Backend service successfully validates valid CICIDS2017 CSV."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "cicids_valid.csv")
            sample_cicids2017_valid_df.to_csv(csv_path, index=False)

            result = run_validation(csv_path, dataset_type=None, total_rows=len(sample_cicids2017_valid_df))

            assert result["dataset_type"] == "CICIDS2017"
            assert result["valid"] is True
            assert result["report"] is not None
            assert result["report"]["is_valid"] is True
            assert result["report"]["rows_checked"] == 100
            assert result["report"]["sampled"] is False
            assert len(result["report"]["errors"]) == 0

    def test_tc_045_run_validation_valid_ciciot2023(self, sample_ciciot2023_valid_df: pd.DataFrame):
        """TC-045: Backend service successfully validates valid CIC-IoT2023 CSV."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "ciciot_valid.csv")
            sample_ciciot2023_valid_df.to_csv(csv_path, index=False)

            result = run_validation(csv_path, dataset_type="CIC-IoT2023", total_rows=len(sample_ciciot2023_valid_df))

            assert result["dataset_type"] == "CIC-IoT2023"
            assert result["valid"] is True
            assert result["report"]["is_valid"] is True
            assert result["report"]["rows_checked"] == 100
            assert result["report"]["sampled"] is False

    def test_tc_046_run_validation_unrecognized_dataset(self):
        """TC-046: Reject CSV with unrecognised columns and provide clear error message."""
        dummy_df = pd.DataFrame({"col_x": [1, 2, 3], "col_y": ["a", "b", "c"]})
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "unknown.csv")
            dummy_df.to_csv(csv_path, index=False)

            result = run_validation(csv_path, dataset_type=None, total_rows=3)

            assert result["dataset_type"] is None
            assert result["valid"] is False
            assert result["report"] is None
            assert "Columns do not match a known dataset" in result["message"]

    def test_tc_047_run_validation_corrupted_dataset(self, sample_cicids2017_valid_df: pd.DataFrame):
        """TC-047: Flag validation failure on CSV missing mandatory features."""
        corrupt_df = sample_cicids2017_valid_df.drop(columns=["Flow Duration", "Total Fwd Packets"])
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "corrupt.csv")
            corrupt_df.to_csv(csv_path, index=False)

            result = run_validation(csv_path, dataset_type="CICIDS2017", total_rows=len(corrupt_df))

            assert result["dataset_type"] == "CICIDS2017"
            assert result["valid"] is False
            assert result["report"]["is_valid"] is False
            assert len(result["report"]["errors"]) > 0
            assert "missing" in result["message"].lower() and "mandatory feature" in result["message"].lower()

    def test_tc_048_run_validation_sampling_enforcement(self, sample_cicids2017_valid_df: pd.DataFrame):
        """TC-048: Enforce sample limit on large datasets without excessive memory usage."""
        # Create a larger DataFrame of 250 rows
        large_df = pd.concat([sample_cicids2017_valid_df] * 5, ignore_index=True)
        total_rows = len(large_df)  # 500 rows

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "large_sample.csv")
            large_df.to_csv(csv_path, index=False)

            # Patch VALIDATION_SAMPLE_ROWS to 150 rows for testing sampling limit
            with patch.object(settings, "VALIDATION_SAMPLE_ROWS", 150):
                result = run_validation(csv_path, dataset_type="CICIDS2017", total_rows=total_rows)

                assert result["valid"] is True
                assert result["report"]["sampled"] is True
                assert result["report"]["rows_checked"] == 150
                assert "Validated the first 150 of 500 rows" in result["message"]

    def test_tc_049_pipeline_unavailable_error_handling(self):
        """TC-049: Raise PipelineUnavailable gracefully if pipeline root does not exist."""
        with patch.object(settings, "PIPELINE_ROOT", "/nonexistent/path/to/pipeline"):
            with patch("app.services.data_pipeline._cache", None):
                with pytest.raises(PipelineUnavailable, match="data pipeline code not found"):
                    _load()

    def test_tc_050_end_to_end_validated_dataset_to_model_pipeline(
        self, sample_cicids2017_valid_df: pd.DataFrame
    ):
        """TC-050: Full ingestion loop from validated CSV directly to DataPipeline transform & export."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "ingested_cicids2017.csv")
            sample_cicids2017_valid_df.to_csv(csv_path, index=False)

            # 1. Run backend ingestion validation
            val_result = run_validation(csv_path, dataset_type=None, total_rows=len(sample_cicids2017_valid_df))
            assert val_result["valid"] is True
            dataset_name = val_result["dataset_type"]

            # 2. Feed validated data directly to ML DataPipeline
            pipeline = DataPipeline(dataset_name=dataset_name, scaler_type="minmax")
            dataset: ProcessedDataset = pipeline.fit_transform(sample_cicids2017_valid_df)

            assert pipeline.is_fitted is True
            assert dataset.X_train.shape[0] == 69
            assert dataset.X_train.shape[1] == len(dataset.feature_names)
            assert dataset.X_train.min() >= 0.0 and dataset.X_train.max() <= 1.0

            # 3. Export artifacts for backend model serving
            artifacts = pipeline.export_artifacts(tmp_dir)
            with open(artifacts["schema_json"], "r", encoding="utf-8") as f:
                schema_contract = json.load(f)

            assert schema_contract["dataset_name"] == "CICIDS2017"
            assert "feature_columns" in schema_contract
            assert schema_contract["num_features"] == len(dataset.feature_names)
            assert "class_to_idx" in schema_contract
            assert os.path.exists(artifacts["preprocessor_joblib"])
            assert os.path.exists(artifacts["cleaner_joblib"])

    def test_tc_051_pipeline_microbatch_streaming_throughput(
        self, sample_cicids2017_valid_df: pd.DataFrame
    ):
        """TC-051: Micro-batch streaming on validated data exceeds throughput threshold (> 1,000 flows/sec)."""
        # Expand dataset to 2,000 synthetic flow samples
        expanded_df = pd.concat([sample_cicids2017_valid_df] * 20, ignore_index=True)
        pipeline = DataPipeline(dataset_name="CICIDS2017")
        pipeline.fit(expanded_df)

        chunk_size = 100
        start_time = time.perf_counter()
        processed_count = 0

        for X_chunk, y_chunk, y_bin_chunk in pipeline.process_stream(expanded_df, chunk_size=chunk_size):
            processed_count += len(X_chunk)
            assert not np.isnan(X_chunk).any()
            assert not np.isinf(X_chunk).any()

        elapsed = time.perf_counter() - start_time
        throughput = processed_count / elapsed

        assert processed_count == 2000
        assert elapsed < 5.0  # Should process 2,000 flows in well under 5 seconds
        assert throughput > 1000.0  # Must exceed 1,000 flows per second
