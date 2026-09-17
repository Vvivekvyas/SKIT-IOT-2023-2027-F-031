# Initial Test Cases Matrix

This document provides the formal test case specifications for the **Hybrid ML Based Intrusion Detection System (IDS)**. 

> [!NOTE]
> These initial test cases define the acceptance requirements across all system tiers. As development progresses through subsequent milestones, automated and manual test runs will update the **Actual Result** and **Status** fields (`Planned`, `In Progress`, `Passed`, `Failed`, `Blocked`).

---

## Test Execution Summary

| Total Test Cases | Planned / Draft | Passed | Failed | Blocked |
| :---: | :---: | :---: | :---: | :---: |
| 22 | 22 | 0 | 0 | 0 |

---

## Detailed Test Cases Matrix

| Test ID | Component | Test Description | Preconditions | Input Data | Expected Result | Actual Result | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TC-001** | Dataset | Upload valid dataset file | Server running, valid API token | Valid CSV file (`cicids2017_sample.csv`) with standard 78+ flow features | HTTP 200: Dataset accepted, schema validated, sample count returned | Pending execution | Planned |
| **TC-002** | Dataset | Upload unsupported file type | Server running, valid API token | Non-CSV file (e.g., `payload.exe` or `data.txt`) | HTTP 415 / 400: Unsupported Media Type, file upload rejected | Pending execution | Planned |
| **TC-003** | Dataset | Upload empty file | Server running, valid API token | 0-byte CSV file (`empty.csv`) | HTTP 400 / 422: Validation error indicating empty payload | Pending execution | Planned |
| **TC-004** | Dataset | Upload dataset with missing mandatory columns | Server running, valid API token | CSV missing required features (e.g., missing `Flow Duration`) | HTTP 422: Schema validation error identifying missing columns | Pending execution | Planned |
| **TC-005** | Preprocessing | Standard preprocessing on valid flow records | Preprocessing pipeline initialized | DataFrame with raw numerical columns, mixed scales | All features scaled to $[0, 1]$ or standardized; identifiers pruned | Pending execution | Planned |
| **TC-006** | Preprocessing | Handle records containing missing (`NaN`) and infinite (`inf`) values | Preprocessing pipeline initialized | Flow records containing `NaN`, `+inf`, `-inf` in flow byte rates | Infinite/missing values imputed via median/mean or capped gracefully without crash | Pending execution | Planned |
| **TC-007** | Preprocessing | Categorical label encoding | Preprocessing pipeline initialized | String labels (e.g., `BENIGN`, `DDoS`, `PortScan`) | Labels correctly mapped to consistent integer class indices $[0, K-1]$ | Pending execution | Planned |
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
| **TC-021** | Performance | Single-flow prediction latency benchmark | Server running in test environment | 100 sequential inference requests | 95th percentile latency $\le 25\text{ ms}$ per sample | Pending execution | Planned |
| **TC-022** | Performance | Batch flow processing throughput | Server running in test environment | Batch request with 1,000 flow records | Batch processed successfully within $\le 2.0\text{ seconds}$ | Pending execution | Planned |

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
