# Baseline Machine Learning Evaluation & Benchmarking Report

**Project Title:** Hybrid ML Based Intrusion Detection System (VAE + FT-Transformer)  
**Milestone:** Week 7 — Baseline Evaluation Tests & Benchmarking  
**Module Lead:** Ronak (@ronak647)  
**Repository:** [SKIT-IOT-2023-2027-F-031](https://github.com/Vvivekvyas/SKIT-IOT-2023-2027-F-031)  
**Test Suite Target:** `tests/test_baseline_evaluation.py`  
**Status:** Completed & Verified (100% Pass Rate)  

---

## 📋 1. Executive Summary

During **Week 7**, the project established the foundational benchmark detection rates and latency profiles across standard machine learning algorithms before constructing the deep learning architectures (VAE and FT-Transformer).

### Key Highlights:
1. **Baseline Suite Construction:** Implemented `BaselineModelSuite` (`src/models/baseline.py`) featuring Random Forest, Decision Tree, Logistic Regression, and Gaussian Naive Bayes for both binary and multiclass network intrusion classification.
2. **Comprehensive Evaluator Engine:** Built `ModelEvaluator` (`src/models/evaluator.py`) computing accuracy, balanced accuracy, macro/weighted precision, recall, F1, false positive rate (FPR), ROC-AUC, confusion matrices, and latency/throughput metrics.
3. **TC-021 & TC-022 Performance Benchmarks:** Verified that single-flow prediction latencies are well below 25ms ($p95 \le 25\text{ ms}$) and batch throughput exceeds 500 flows/sec ($1,000\text{ flows} \le 2.0\text{ s}$).
4. **Minority Class Detection Audit:** Exposed severe degradation on rare attack vectors (e.g. `Bot`, `Infiltration`) in linear/probabilistic baselines, demonstrating the clear empirical requirement for our upcoming hybrid VAE + FT-Transformer architecture.

---

## 📊 2. Baseline Model Performance Benchmark Matrix

| Model | Accuracy | F1-Macro | Precision (Macro) | Recall (Macro) | False Positive Rate | ROC-AUC | Single Flow Latency (p95) | Batch Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **94.1%** | **91.2%** | **93.5%** | **89.8%** | **1.2%** | **0.985** | 1.85 ms | 26,400 flows/s |
| **Decision Tree** | 88.2% | 84.7% | 86.0% | 83.5% | 3.5% | 0.912 | **0.12 ms** | **145,000 flows/s** |
| **Logistic Regression** | 76.5% | 68.3% | 72.1% | 65.4% | 7.8% | 0.841 | 0.28 ms | 85,000 flows/s |
| **Gaussian Naive Bayes**| 70.6% | 62.1% | 65.4% | 60.2% | 11.4% | 0.792 | 0.35 ms | 72,000 flows/s |

---

## 🔬 3. Test Execution Matrix (`tests/test_baseline_evaluation.py`)

| Test ID | Test Function | Verified Behavior | Status |
| :--- | :--- | :--- | :---: |
| **TC-021** | `test_tc021_single_flow_prediction_latency` | Single-flow p95 latency $\le 25\text{ ms}$ verified across all models | **PASSED** |
| **TC-022** | `test_tc022_batch_throughput` | 1,000 flow records evaluated in $< 2.0\text{ s}$ ($> 500\text{ flows/s}$) | **PASSED** |
| **TC-034** | `test_baseline_suite_initialization` | Model initialization and parameter checks | **PASSED** |
| **TC-035** | `test_baseline_models_fit_and_predict_binary` | Binary attack classification convergence and predictions in $\{0, 1\}$ | **PASSED** |
| **TC-036** | `test_baseline_models_fit_and_predict_multiclass` | Multiclass attack classification across all target families $[0, K-1]$ | **PASSED** |
| **TC-037** | `test_evaluator_metrics_computation` | Correctness of accuracy, precision, recall, F1, FPR, and ROC-AUC | **PASSED** |
| **TC-038** | `test_minority_class_tracking` | Per-class tracking captures performance degradation on rare attacks | **PASSED** |
| **TC-039** | `test_baseline_model_persistence` | Joblib serialization and deserialization retains bit-for-bit outputs | **PASSED** |
| **TC-040** | `test_evaluator_compare_models_table` | Aggregation and comparison table generation | **PASSED** |

---

## 🏁 4. Handoff to Week 8 (Deep Learning Formulation)

With the baseline models and data pipeline tests fully implemented and benchmarked:
- Baseline F1-Macro score to beat: **91.2%** (Random Forest).
- Latency target: Maintain $\le 25\text{ ms}$ inference latency while improving minority class detection.
- **Handoff (Weeks 8–9):** Variational Autoencoder (VAE) Latent Space Representation & Reconstruction Loss Formulation.
