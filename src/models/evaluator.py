"""Evaluation and Benchmarking Engine for Intrusion Detection Models (Week 7 - Ronak).

Calculates multi-dimensional performance metrics:
- Accuracy, Balanced Accuracy
- Macro and Weighted Precision, Recall, F1-Score
- False Positive Rate (FPR), False Negative Rate (FNR)
- ROC-AUC (Binary & Multiclass OvR)
- Confusion Matrix & Per-Class Minority Breakdown
- Inference Latency (p50, p95, p99 ms) & Batch Throughput (samples/sec)
"""

from dataclasses import dataclass, field
import json
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


@dataclass
class EvaluationMetrics:
    """Encapsulates evaluation metrics for a model on a test set."""
    model_name: str
    dataset_name: str
    task_type: str  # 'binary' or 'multiclass'
    total_samples: int
    accuracy: float
    balanced_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    false_positive_rate: float
    false_negative_rate: float
    roc_auc: Optional[float]
    confusion_matrix: List[List[int]]
    per_class_metrics: Dict[str, Dict[str, float]]
    single_flow_latency_p50_ms: float
    single_flow_latency_p95_ms: float
    single_flow_latency_p99_ms: float
    batch_throughput_samples_per_sec: float
    batch_1000_duration_sec: float

    def to_dict(self) -> Dict[str, Any]:
        """Converts metrics to a serializable dictionary."""
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "task_type": self.task_type,
            "total_samples": self.total_samples,
            "accuracy": round(self.accuracy, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "precision_macro": round(self.precision_macro, 4),
            "recall_macro": round(self.recall_macro, 4),
            "f1_macro": round(self.f1_macro, 4),
            "precision_weighted": round(self.precision_weighted, 4),
            "recall_weighted": round(self.recall_weighted, 4),
            "f1_weighted": round(self.f1_weighted, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "false_negative_rate": round(self.false_negative_rate, 4),
            "roc_auc": round(self.roc_auc, 4) if self.roc_auc is not None else None,
            "confusion_matrix": self.confusion_matrix,
            "per_class_metrics": self.per_class_metrics,
            "latency": {
                "p50_ms": round(self.single_flow_latency_p50_ms, 3),
                "p95_ms": round(self.single_flow_latency_p95_ms, 3),
                "p99_ms": round(self.single_flow_latency_p99_ms, 3),
                "throughput_samples_sec": round(self.batch_throughput_samples_per_sec, 1),
                "batch_1000_duration_sec": round(self.batch_1000_duration_sec, 4),
            },
        }


class ModelEvaluator:
    """Evaluates classification models and conducts rigorous benchmarking audits."""

    def __init__(self, dataset_name: str = "CICIDS2017"):
        self.dataset_name = dataset_name

    def evaluate(
        self,
        model: Any,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        task_type: str = "multiclass",
        class_names: Optional[Dict[int, str]] = None,
        latency_benchmark_samples: int = 100,
    ) -> EvaluationMetrics:
        """
        Executes complete evaluation suite across predictions, probabilities, and latencies.
        """
        total_samples = len(X_test)
        y_pred = model.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        b_acc = float(balanced_accuracy_score(y_test, y_pred))

        prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1_m = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

        prec_wt = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec_wt = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1_wt = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # FPR & FNR calculations
        if task_type == "binary" or cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        else:
            # Multiclass macro FPR: treat class 0 (Benign) as negative
            fp = float(np.sum(cm[:, 0]) - cm[0, 0])
            tn = float(np.sum(cm) - np.sum(cm[0, :]) - np.sum(cm[:, 0]) + cm[0, 0])
            fn = float(np.sum(cm[0, :]) - cm[0, 0])
            tp = float(cm[0, 0])
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        # ROC-AUC calculation
        roc_auc_val: Optional[float] = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)
                if task_type == "binary":
                    # Binary ROC-AUC uses probability of positive class
                    prob_col = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob[:, 0]
                    roc_auc_val = float(roc_auc_score(y_test, prob_col))
                else:
                    # Multiclass OvR ROC-AUC
                    roc_auc_val = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"))
            except Exception:
                roc_auc_val = None

        # Per-class metrics
        unique_classes = np.unique(np.concatenate([y_test, y_pred]))
        per_class: Dict[str, Dict[str, float]] = {}
        for c in unique_classes:
            name = class_names.get(int(c), f"Class_{c}") if class_names else f"Class_{c}"
            y_t_c = (y_test == c).astype(int)
            y_p_c = (y_pred == c).astype(int)
            p = float(precision_score(y_t_c, y_p_c, zero_division=0))
            r = float(recall_score(y_t_c, y_p_c, zero_division=0))
            f = float(f1_score(y_t_c, y_p_c, zero_division=0))
            count = int(np.sum(y_t_c))
            per_class[name] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "support": count}

        # Latency & throughput benchmarking
        latencies_ms, throughput, duration_1000 = self.benchmark_latency(model, X_test, samples=latency_benchmark_samples)

        return EvaluationMetrics(
            model_name=model_name,
            dataset_name=self.dataset_name,
            task_type=task_type,
            total_samples=total_samples,
            accuracy=acc,
            balanced_accuracy=b_acc,
            precision_macro=prec_macro,
            recall_macro=rec_macro,
            f1_macro=f1_m,
            precision_weighted=prec_wt,
            recall_weighted=rec_wt,
            f1_weighted=f1_wt,
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            roc_auc=roc_auc_val,
            confusion_matrix=cm.tolist(),
            per_class_metrics=per_class,
            single_flow_latency_p50_ms=float(np.percentile(latencies_ms, 50)),
            single_flow_latency_p95_ms=float(np.percentile(latencies_ms, 95)),
            single_flow_latency_p99_ms=float(np.percentile(latencies_ms, 99)),
            batch_throughput_samples_per_sec=throughput,
            batch_1000_duration_sec=duration_1000,
        )

    def benchmark_latency(
        self,
        model: Any,
        X_test: np.ndarray,
        samples: int = 100,
    ) -> Tuple[List[float], float, float]:
        """
        Benchmarks inference latency for single flows and batch throughput.
        Validates TC-021 (p95 <= 25ms) and TC-022 (batch 1000 <= 2.0s).
        """
        latencies = []
        n_available = len(X_test)

        # 1. Single sample latency benchmark
        for i in range(samples):
            idx = i % n_available
            sample = X_test[idx:idx+1]
            t0 = time.perf_counter()
            _ = model.predict(sample)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        # 2. Batch throughput benchmark (1000 flows)
        batch_size = 1000
        if n_available < batch_size:
            repeat_factor = int(np.ceil(batch_size / n_available))
            batch_data = np.tile(X_test, (repeat_factor, 1))[:batch_size]
        else:
            batch_data = X_test[:batch_size]

        t_start = time.perf_counter()
        _ = model.predict(batch_data)
        t_end = time.perf_counter()

        duration = t_end - t_start
        throughput = batch_size / duration if duration > 0 else 0.0

        return latencies, throughput, duration

    def compare_models(
        self,
        results: List[EvaluationMetrics],
    ) -> pd.DataFrame:
        """Constructs a consolidated comparison table across evaluated models."""
        rows = []
        for r in results:
            rows.append({
                "Model": r.model_name,
                "Accuracy": f"{r.accuracy * 100:.2f}%",
                "F1-Macro": f"{r.f1_macro * 100:.2f}%",
                "Precision": f"{r.precision_macro * 100:.2f}%",
                "Recall": f"{r.recall_macro * 100:.2f}%",
                "FPR": f"{r.false_positive_rate * 100:.2f}%",
                "ROC-AUC": f"{r.roc_auc:.4f}" if r.roc_auc else "N/A",
                "p95 Latency": f"{r.single_flow_latency_p95_ms:.2f} ms",
                "Throughput": f"{r.batch_throughput_samples_per_sec:.0f} flows/s",
            })
        return pd.DataFrame(rows)
