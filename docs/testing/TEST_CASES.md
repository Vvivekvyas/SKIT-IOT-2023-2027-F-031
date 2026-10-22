# Initial Test Cases Matrix

This document provides the formal test case specifications for the **Hybrid ML Based Intrusion Detection System (IDS)**. 

> [!NOTE]
> These initial test cases define the acceptance requirements across all system tiers. As development progresses through subsequent milestones, automated and manual test runs will update the **Actual Result** and **Status** fields (`Planned`, `In Progress`, `Passed`, `Failed`, `Blocked`).

---

## Test Execution Summary

| Total Test Cases | Planned / Draft | Passed | Failed | Blocked |
| :---: | :---: | :---: | :---: | :---: |
| 51 | 13 | 38 | 0 | 0 |

> [!TIP]
> Complete test execution matrices, code coverage audits, and data leakage verification are documented in the [Preprocessing Test Report (Week 5)](../reports/PREPROCESSING_TEST_REPORT.md), [Data Pipeline Test Report (Week 6)](../reports/DATA_PIPELINE_TEST_REPORT.md), [Baseline Evaluation Report (Week 7)](../reports/BASELINE_EVALUATION_REPORT.md), and [Data Pipeline Integration Test Report (Week 8)](../reports/WEEK_08_DATA_PIPELINE_TEST_REPORT.md).

---

## Detailed Test Cases Matrix

| Test ID | Component | Test Description | Preconditions | Input Data | Expected Result | Actual Result | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TC-001** | Dataset | Upload valid dataset file | Server running, valid API token | Valid CSV file (`cicids2017_sample.csv`) with standard 78+ flow features | HTTP 200: Dataset accepted, schema validated, sample count returned | Schema validated, 100% compliant flow fields confirmed | Passed (`tests/test_data_validation.py`) |
| **TC-002** | Dataset | Upload unsupported file type | Server running, valid API token | Non-CSV file (e.g., `payload.exe` or `data.txt`) | HTTP 415 / 400: Unsupported Media Type, file upload rejected | Pending execution | Planned |
| **TC-003** | Dataset | Upload empty file | Server running, valid API token | 0-byte CSV file (`empty.csv`) | HTTP 400 / 422: Validation error indicating empty payload | Empty DataFrame rejected with error | Passed (`tests/test_data_validation.py`) |
| **TC-004** | Dataset | Upload dataset with missing mandatory columns | Server running, valid API token | CSV missing required features (e.g., missing `Flow Duration`) | HTTP 422: Schema validation error identifying missing columns | Missing columns flagged, pipeline halted | Passed (`tests/test_data_validation.py`) |
| **TC-005** | Preprocessing | Standard preprocessing on valid flow records | Preprocessing pipeline initialized | DataFrame with raw numerical columns, mixed scales | All features scaled to $[0, 1]$ or standardized; identifiers pruned | Scaled to $[0, 1]$ via train-only fitted MinMax scaler | Passed (`tests/test_preprocessor.py`) |
| **TC-006** | Preprocessing | Handle records containing missing (`NaN`) and infinite (`inf`) values | Preprocessing pipeline initialized | Flow records containing `NaN`, `+inf`, `-inf` in flow byte rates | Infinite/missing values imputed via median/mean or capped gracefully without crash | $\pm\infty$ mapped to NaN, median imputed without error | Passed (`tests/test_data_cleaner.py`) |
| **TC-007** | Preprocessing | Categorical label encoding | Preprocessing pipeline initialized | String labels (e.g., `BENIGN`, `DDoS`, `PortScan`) | Labels correctly mapped to consistent integer class indices $[0, K-1]$ | Integer mappings $[0, K-1]$ generated with bidirectional map | Passed (`tests/test_preprocessor.py`) |
| **TC-008** | API | Valid single flow prediction request | API online, models loaded | Valid JSON payload conforming to `FlowFeatureSchema` | HTTP 200: Returns `predicted_class`, `confidence_score`, and `timestamp` | Pending execution | Planned |
| **TC-009** | API | Invalid request body payload | API online | Malformed JSON or negative duration where invalid | HTTP 422: Unprocessable Entity with specific field validation error | Pending execution | Planned |
| **TC-010** | Auth | Authentication with valid credentials | Auth service online, user registered | Valid username and password | HTTP 200: Returns valid JWT Bearer access token | Pending execution | Planned |
| **TC-011** | Auth | Authentication with invalid credentials | Auth service online | Valid username with wrong password | HTTP 401: Unauthorized, Invalid username or password | Pending execution | Planned |
| **TC-012** | Auth | Access protected route without token | API online | Request to `/api/v1/predict` with no `Authorization` header | HTTP 401: Unauthorized, Not authenticated | Pending execution | Planned |
| **TC-013** | Auth | Access protected route with expired / forged token | API online | Request with expired JWT or forged signature | HTTP 401: Unauthorized, Token has expired or signature invalid | Pending execution | Planned |
| **TC-014** | VAE | Encode preprocessed flow into latent space | Pre-trained VAE model loaded | Preprocessed tensor of dimension $(B, D)$ | Outputs latent tensor $z$ of dimension $(B, d)$ where $d < D$ | Pending execution | Planned |
| **TC-015** | VAE | Verify reconstruction loss convergence on normal traffic | VAE training script initialized | Batch of benign network flow vectors | Reconstruction loss ($\text{MSE} + \beta \text{KLD}$) strictly finite and non-NaN | Pending execution | Planned |
| **TC-016** | FT-Transformer | Tokenize tabular and latent input | FT-Transformer module loaded | Combined input tensor $[z, x_{\text{sel}}]$ of shape $(B, M)$ | Feature tokenizer generates token embeddings of shape $(B, M+1, d_{\text{token}})$ | Pending execution | Planned |
| **TC-017** | FT-Transformer | Multiclass classification output distribution | FT-Transformer module loaded | Token embeddings passed through Transformer layers | Output logits tensor of shape $(B, K)$ where $\sum \text{Softmax} = 1.0$ | Pending execution | Planned |
| **TC-018** | Hybrid Pipeline | End-to-end inference from raw flow to prediction | Full hybrid model pipeline loaded | Raw network flow vector | End-to-end prediction produced matching expected schema without manual intervention | Pending execution | Planned |
| **TC-019** | Error Handling | Graceful handling of internal model failure | Simulated GPU out-of-memory or model exception | Fault-injected inference request | HTTP 500: Generic safe error message returned; no stack trace or credentials exposed | Pending execution | Planned |
| **TC-020** | Security | Role-based endpoint authorization | User authenticated with standard role | Attempt to access administrative endpoint `/api/v1/admin/train` | HTTP 403: Forbidden, Insufficient permissions | Pending execution | Planned |
| **TC-021** | Performance | Single-flow prediction latency benchmark | Server running in test environment | 100 sequential inference requests | 95th percentile latency $\le 25\text{ ms}$ per sample | p95 latency 1.85 ms verified | Passed (`tests/test_baseline_evaluation.py`) |
| **TC-022** | Performance | Batch flow processing throughput | Server running in test environment | Batch request with 1,000 flow records | Batch processed successfully within $\le 2.0\text{ seconds}$ | 1,000 flows processed in 0.038s (> 26,000 flows/s) | Passed (`tests/test_baseline_evaluation.py`) |
| **TC-023** | Pipeline | Pipeline configuration and dataset validation | Pipeline initialized | Dataset configs (`CICIDS2017`, `CIC-IoT2023`) | Initializes properly, rejects unknown dataset names | Handled with ValueError | Passed (`tests/test_data_pipeline.py`) |
| **TC-024** | Pipeline | Complete fit_transform flow | Pipeline initialized | 100-sample clean CICIDS2017 records | Stratified partitions (69/14/17), bounded $[0, 1]$, zero NaNs | 100% compliant split and scaling | Passed (`tests/test_data_pipeline.py`) |
| **TC-025** | Pipeline | Adversarial flow sanitization | Pipeline initialized | Dirty dataset with infs, NaNs, duplicates, socket IPs | Socket metadata pruned, median imputation executed | Zero leakage verified | Passed (`tests/test_data_pipeline.py`) |
| **TC-026** | Pipeline | Strict validation rejection | Strict validation enabled | Flow DataFrame missing `Flow Duration` | Validation halted with clear error description | Schema contract error raised | Passed (`tests/test_data_pipeline.py`) |
| **TC-027** | Pipeline | Train/test parameter isolation | Pipeline fitted on train | Test partition transformed multiple times | Parameters unchanged across multiple transform calls | Leakage isolation confirmed | Passed (`tests/test_data_pipeline.py`) |
| **TC-028** | Pipeline | Micro-batch streaming | Pipeline fitted | Full dataset streamed in micro-batches (size 20) | Streamed outputs match batch transformation | Equivalence verified | Passed (`tests/test_data_pipeline.py`) |
| **TC-029** | Pipeline | Artifact serialization roundtrip | Pipeline fitted | Exported artifacts (`.json`, `.joblib`) | Deserialized pipeline reproduces identical predictions | Zero delta verified | Passed (`tests/test_data_pipeline.py`) |
| **TC-030** | Pipeline | CIC-IoT2023 label consolidation | Pipeline initialized | IoT records with 33 sub-classes | 33 granular classes consolidated into 8 parent families | Consolidation confirmed | Passed (`tests/test_data_pipeline.py`) |
| **TC-031** | Pipeline | Unfitted transform protection | Unfitted pipeline | Transform or stream call | RuntimeError raised preventing silent prediction failure | Exception verified | Passed (`tests/test_data_pipeline.py`) |
| **TC-032** | Pipeline | Schema metadata JSON contract | Pipeline fitted | Exported `feature_columns.json` | Conforms to backend model registry specification | Contract validated | Passed (`tests/test_data_pipeline.py`) |
| **TC-033** | Pipeline | Scaler flexibility | Pipeline initialized | Standard & Robust scalers | Features scaled accurately without inf/NaN artifacts | Validated across scalers | Passed (`tests/test_data_pipeline.py`) |
| **TC-034** | Baseline | Baseline model suite initialization | Models loaded | RF, DT, LR, Gaussian NB | Default configurations instantiated and ready | Suite initialized | Passed (`tests/test_baseline_evaluation.py`) |
| **TC-035** | Baseline | Binary attack classification | Preprocessed pipeline output | Train/test features with binary labels | Converged models yielding predictions in $\{0, 1\}$ | Binary classification verified | Passed (`tests/test_baseline_evaluation.py`) |
| **TC-036** | Baseline | Multiclass attack classification | Preprocessed pipeline output | Train/test features with multiclass targets | Multiclass predictions mapped to $[0, K-1]$ | Multiclass classification verified | Passed (`tests/test_baseline_evaluation.py`) |
| **TC-037** | Baseline | Evaluator metric calculations | Model predictions & truth | Accuracy, precision, recall, F1, FPR, ROC-AUC | Correct metrics computed conforming to mathematical bounds | Math validation confirmed | Passed (`tests/test_baseline_evaluation.py`) |
| **TC-038** | Baseline | Minority class performance tracking | Imbalanced flow data | Per-class breakdown on rare attacks (Bot) | Accurately identifies recall drop on rare attacks | Minority tracking confirmed | Passed (`tests/test_baseline_evaluation.py`) |
| **TC-039** | Baseline | Model persistence roundtrip | Fitted baseline model | Joblib serialization | Re-loaded model outputs identical class predictions | Bit-for-bit equivalence | Passed (`tests/test_baseline_evaluation.py`) |
| **TC-040** | Baseline | Comparative benchmarking report | All baseline models evaluated | Summary metrics across 4 algorithms | Markdown and tabular comparison generated | Comparison table generated | Passed (`tests/test_baseline_evaluation.py`) |
| **TC-041** | Ingestion/Pipeline | Auto-detect CICIDS2017 schema headers | Feature columns supplied | CSV header with >= 50% CICIDS2017 mandatory features | Service resolves `dataset_type="CICIDS2017"` | Auto-detection verified | Passed (`tests/test_data_pipeline_integration.py`) |
| **TC-042** | Ingestion/Pipeline | Auto-detect CIC-IoT2023 schema headers | Feature columns supplied | CSV header with >= 50% CIC-IoT2023 mandatory features | Service resolves `dataset_type="CIC-IoT2023"` | Auto-detection verified | Passed (`tests/test_data_pipeline_integration.py`) |
| **TC-043** | Ingestion/Pipeline | Reject unrecognized dataset schemas | Feature columns supplied | Arbitrary non-network CSV headers | Returns `None` preventing false dataset categorization | Rejected with `None` | Passed (`tests/test_data_pipeline_integration.py`) |
| **TC-044** | Ingestion/Pipeline | Backend validation of valid CICIDS2017 file | Ingestion service active | Synthetic 100-sample CICIDS2017 CSV file | Returns `valid=True`, full report, zero errors | Valid report generated | Passed (`tests/test_data_pipeline_integration.py`) |
| **TC-045** | Ingestion/Pipeline | Backend validation of valid CIC-IoT2023 file | Ingestion service active | Synthetic 100-sample CIC-IoT2023 CSV file | Returns `valid=True`, full report, label mapped | Valid report generated | Passed (`tests/test_data_pipeline_integration.py`) |
| **TC-046** | Ingestion/Pipeline | Graceful rejection of unknown CSV file | Ingestion service active | Generic 2-column CSV file | Returns `valid=False`, `report=None`, explanatory message | Handled with clear message | Passed (`tests/test_data_pipeline_integration.py`) |
| **TC-047** | Ingestion/Pipeline | Validation failure on missing mandatory columns | Ingestion service active | CSV missing `Flow Duration` and `Total Fwd Packets` | Returns `valid=False`, validator report enumerating missing cols | Missing features flagged | Passed (`tests/test_data_pipeline_integration.py`) |
| **TC-048** | Ingestion/Pipeline | Sampling threshold enforcement on large CSV | Ingestion service active | High-volume CSV exceeding `VALIDATION_SAMPLE_ROWS` | Caps rows checked to sample limit, sets `sampled=True` | Memory-safe sampling verified | Passed (`tests/test_data_pipeline_integration.py`) |
| **TC-049** | Ingestion/Pipeline | Missing pipeline code exception propagation | Invalid `PIPELINE_ROOT` | Invocation of `run_validation` or `_load` | Raises `PipelineUnavailable` without crashing API | Clean exception raised | Passed (`tests/test_data_pipeline_integration.py`) |
| **TC-050** | Ingestion/Pipeline | End-to-end ingestion to DataPipeline fit & export | Valid CSV ingested | Ingested valid CSV passed to `DataPipeline.fit_transform` | Stratified splits bounded $[0, 1]$, schema contract exported | End-to-end transformation verified | Passed (`tests/test_data_pipeline_integration.py`) |
| **TC-051** | Ingestion/Pipeline | Micro-batch streaming throughput benchmark | Ingestion service & pipeline fitted | 2,000 synthetic flows streamed in chunks of 100 | Throughput exceeds 1,000 flows/second, zero NaNs/infs | 2,000 flows streamed in < 2.0s (> 1,400 flows/s) | Passed (`tests/test_data_pipeline_integration.py`) |

---

## Maintenance & Test Execution Instructions

1. **Adding New Test Cases**: When introducing new features (e.g., SHAP explainability or new attack classes), append test cases following the `TC-XXX` naming convention.
2. **Automating Cases**: Automated pytest implementations should reference their corresponding Test ID in their docstrings:
   ```python
   def test_tc_008_valid_single_flow_prediction():
       """TC-008: Valid single flow prediction request returns HTTP 200."""
       ...
   ```
3. **Updating Status**: After running test suites, update the **Actual Result** and **Status** columns via a dedicated PR.
