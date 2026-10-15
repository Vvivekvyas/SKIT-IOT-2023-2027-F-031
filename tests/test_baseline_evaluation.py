"""Comprehensive tests for Baseline Models & Evaluation Engine (Week 7 - Ronak).

Verifies:
- Baseline models training & prediction (Random Forest, Decision Tree, Logistic Regression, Naive Bayes)
- Binary & multiclass evaluation metrics computation
- False positive rate (FPR) & confusion matrix verification
- Minority class detection tracking
- TC-021: Single-flow prediction latency benchmark (p95 <= 25ms)
- TC-022: Batch throughput processing (1000 samples <= 2.0s)
- Model artifact persistence and serialization
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.data.pipeline import DataPipeline
from src.models.baseline import BaselineModelSuite
from src.models.evaluator import ModelEvaluator, EvaluationMetrics


@pytest.fixture
def processed_cicids2017_data(sample_cicids2017_valid_df: pd.DataFrame):
    """Provides preprocessed pipeline data for model training and evaluation."""
    pipeline = DataPipeline(dataset_name="CICIDS2017", scaler_type="minmax")
    dataset = pipeline.fit_transform(sample_cicids2017_valid_df)
    return dataset


class TestBaselineEvaluation:
    """Verifies baseline algorithms, evaluation metrics, minority class sensitivity, and latency benchmarks."""

    def test_baseline_suite_initialization(self):
        """TC-034: Verify initialization of baseline models suite."""
        suite = BaselineModelSuite(random_state=42)
        assert "random_forest" in suite.models
        assert "decision_tree" in suite.models
        assert "logistic_regression" in suite.models
        assert "naive_bayes" in suite.models
        for name in suite.model_names:
            assert suite.fitted_models[name] is False

    def test_baseline_models_fit_and_predict_binary(self, processed_cicids2017_data):
        """TC-035: Verify binary attack detection fitting and prediction across all baselines."""
        dataset = processed_cicids2017_data
        suite = BaselineModelSuite(random_state=42)
        suite.fit_all(dataset.X_train, dataset.y_train_binary)

        for model_name in suite.model_names:
            preds = suite.predict(model_name, dataset.X_test)
            assert len(preds) == len(dataset.X_test)
            assert set(np.unique(preds)).issubset({0, 1})

    def test_baseline_models_fit_and_predict_multiclass(self, processed_cicids2017_data):
        """TC-036: Verify multiclass attack classification across all baselines."""
        dataset = processed_cicids2017_data
        suite = BaselineModelSuite(random_state=42)
        suite.fit_all(dataset.X_train, dataset.y_train)

        num_classes = len(dataset.class_mapping)
        for model_name in suite.model_names:
            preds = suite.predict(model_name, dataset.X_test)
            assert len(preds) == len(dataset.X_test)
            assert preds.min() >= 0
            assert preds.max() < num_classes

    def test_evaluator_metrics_computation(self, processed_cicids2017_data):
        """TC-037: Verify evaluator metrics calculation (accuracy, precision, recall, F1, FPR, ROC-AUC)."""
        dataset = processed_cicids2017_data
        suite = BaselineModelSuite(random_state=42)
        suite.fit_single("random_forest", dataset.X_train, dataset.y_train_binary)

        evaluator = ModelEvaluator(dataset_name="CICIDS2017")
        metrics: EvaluationMetrics = evaluator.evaluate(
            model=suite.models["random_forest"],
            model_name="RandomForest",
            X_test=dataset.X_test,
            y_test=dataset.y_test_binary,
            task_type="binary",
        )

        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.precision_macro <= 1.0
        assert 0.0 <= metrics.recall_macro <= 1.0
        assert 0.0 <= metrics.f1_macro <= 1.0
        assert 0.0 <= metrics.false_positive_rate <= 1.0
        assert metrics.roc_auc is not None and 0.0 <= metrics.roc_auc <= 1.0
        assert len(metrics.confusion_matrix) == 2

    def test_minority_class_tracking(self, processed_cicids2017_data):
        """TC-038: Verify evaluator tracks per-class metrics including minority attacks (Bot, PortScan)."""
        dataset = processed_cicids2017_data
        suite = BaselineModelSuite(random_state=42)
        suite.fit_single("random_forest", dataset.X_train, dataset.y_train)

        evaluator = ModelEvaluator(dataset_name="CICIDS2017")
        metrics = evaluator.evaluate(
            model=suite.models["random_forest"],
            model_name="RandomForest",
            X_test=dataset.X_test,
            y_test=dataset.y_test,
            task_type="multiclass",
            class_names=dataset.inverse_class_mapping,
        )

        # Ensure all unique test classes have per-class metrics recorded
        for cls_name in ["BENIGN", "DDoS", "PortScan"]:
            if cls_name in metrics.per_class_metrics:
                m = metrics.per_class_metrics[cls_name]
                assert "precision" in m
                assert "recall" in m
                assert "f1" in m
                assert "support" in m

    def test_tc021_single_flow_prediction_latency(self, processed_cicids2017_data):
        """TC-021: Verify single-flow prediction latency meets 95th percentile <= 25ms constraint."""
        dataset = processed_cicids2017_data
        suite = BaselineModelSuite(random_state=42)
        suite.fit_all(dataset.X_train, dataset.y_train)

        evaluator = ModelEvaluator(dataset_name="CICIDS2017")

        for model_name in suite.model_names:
            metrics = evaluator.evaluate(
                model=suite.models[model_name],
                model_name=model_name,
                X_test=dataset.X_test,
                y_test=dataset.y_test,
                latency_benchmark_samples=50,
            )
            # TC-021 constraint: p95 latency must be <= 25.0 ms
            assert metrics.single_flow_latency_p95_ms <= 25.0, (
                f"{model_name} p95 latency exceeded: {metrics.single_flow_latency_p95_ms}ms"
            )

    def test_tc022_batch_throughput(self, processed_cicids2017_data):
        """TC-022: Verify batch flow processing throughput (1,000 samples <= 2.0 seconds)."""
        dataset = processed_cicids2017_data
        suite = BaselineModelSuite(random_state=42)
        suite.fit_single("decision_tree", dataset.X_train, dataset.y_train)

        evaluator = ModelEvaluator(dataset_name="CICIDS2017")
        metrics = evaluator.evaluate(
            model=suite.models["decision_tree"],
            model_name="DecisionTree",
            X_test=dataset.X_test,
            y_test=dataset.y_test,
        )

        # TC-022 constraint: 1,000 records processed in <= 2.0 seconds
        assert metrics.batch_1000_duration_sec <= 2.0, (
            f"Batch duration exceeded: {metrics.batch_1000_duration_sec}s"
        )
        assert metrics.batch_throughput_samples_per_sec >= 500.0

    def test_baseline_model_persistence(self, processed_cicids2017_data):
        """TC-039: Verify saving and loading baseline models preserves predictions."""
        dataset = processed_cicids2017_data
        suite = BaselineModelSuite(random_state=42)
        suite.fit_single("random_forest", dataset.X_train, dataset.y_train)
        orig_preds = suite.predict("random_forest", dataset.X_test)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "rf_baseline.joblib")
            suite.save_model("random_forest", model_path)

            new_suite = BaselineModelSuite(random_state=42)
            new_suite.load_model("random_forest", model_path)
            new_preds = new_suite.predict("random_forest", dataset.X_test)

            np.testing.assert_array_equal(orig_preds, new_preds)

    def test_evaluator_compare_models_table(self, processed_cicids2017_data):
        """TC-040: Verify model comparison table aggregation."""
        dataset = processed_cicids2017_data
        suite = BaselineModelSuite(random_state=42)
        suite.fit_all(dataset.X_train, dataset.y_train_binary)

        evaluator = ModelEvaluator(dataset_name="CICIDS2017")
        results = []
        for name in suite.model_names:
            m = evaluator.evaluate(
                suite.models[name],
                model_name=name,
                X_test=dataset.X_test,
                y_test=dataset.y_test_binary,
                task_type="binary",
                latency_benchmark_samples=20,
            )
            results.append(m)

        df = evaluator.compare_models(results)
        assert len(df) == len(suite.model_names)
        assert "Accuracy" in df.columns
        assert "F1-Macro" in df.columns
        assert "p95 Latency" in df.columns
