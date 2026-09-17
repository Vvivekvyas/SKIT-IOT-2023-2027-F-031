# Test Strategy

## 1. Purpose

The purpose of this testing strategy is to establish a rigorous quality assurance methodology to verify that the **Hybrid ML Based Intrusion Detection System (IDS)** operates accurately, securely, and reliably under varied network traffic conditions and deployment environments.

---

## 2. Testing Scope

The testing lifecycle spans all software engineering and machine learning layers of the system:

- **Dataset Ingestion & Validation**: Verifying file parsing, header consistency, missing data handling, and schema integrity.
- **Data Preprocessing**: Validating scaling, encoding, infinite-value replacement, and leakage prevention.
- **Variational Autoencoder (VAE)**: Checking encoder/decoder tensor shapes, latent dimensionality, and reconstruction loss convergence.
- **FT-Transformer**: Verifying feature tokenization, attention weight computation, multi-class output probabilities, and classification heads.
- **Hybrid Pipeline Integration**: End-to-end testing from raw network flow features to final attack class output.
- **API Functionality**: Validating request routing, status codes, payload serialization, and asynchronous handling.
- **Authentication & Authorization**: Testing JWT token issuance, verification, expiration, and access control.
- **Security & Robustness**: Testing against input tampering, malicious payload injection, and malformed CSV uploads.
- **Performance & Latency**: Assessing throughput (flows/second) and single-flow inference latency.
- **Error Handling**: Verifying graceful degradation and descriptive error responses without leaking internal stack traces.

---

## 3. Testing Levels

### 3.1 Unit Testing
Individual modules, mathematical transformations, and utility functions are isolated and tested independently:
- Preprocessing transformer units (e.g., custom scalers, imputers).
- VAE reparameterization trick function.
- FT-Transformer feature tokenizer and individual attention blocks.
- API utility functions (e.g., token decoding, schema validators).

### 3.2 Integration Testing
Verifies proper data contract exchange and tensor dimension alignment between interconnected sub-systems:
- **Pipeline Flow**: `Raw Flow -> Preprocessing -> VAE Latent Vector -> FT-Transformer -> Prediction`.
- **Service Flow**: `Client Request -> FastAPI Endpoint -> Model Inference Engine -> Database Logging -> Client Response`.

### 3.3 API Testing
Verifies REST API compliance using `pytest` and `httpx` / `FastAPI TestClient`:
- **Valid Requests**: Correct payload structure returns HTTP 200 with appropriate class and confidence score.
- **Malformed Payloads**: Missing fields, out-of-range numerical values, or invalid types return HTTP 422 Unprocessable Entity.
- **Authentication Failures**: Missing or expired bearer tokens return HTTP 401 Unauthorized.
- **Forbidden Actions**: Insufficient role permissions return HTTP 403 Forbidden.

### 3.4 Model Testing & Validation
Evaluation of model predictive capabilities using standardized cybersecurity machine learning metrics:
- **Metrics Evaluated**:
  - Accuracy (Overall classification accuracy)
  - Precision, Recall, Specificity per class
  - Macro F1-score & Weighted F1-score (critical due to severe class imbalance)
  - Multiclass Confusion Matrix
  - Receiver Operating Characteristic (ROC-AUC) curves
- **Data Partitions**:
  - Training Set (70%)
  - Validation Set (15%) - For early stopping and hyperparameter tuning
  - Test Set (15%) - Strictly held out until final evaluation

### 3.5 Security Testing
Evaluates resilience against malicious exploits:
- **Authentication & Session**: Token tampering, invalid signatures, expired tokens.
- **Input Validation**: Extremely large file uploads (DoS resistance), non-CSV file extensions, SQL/NoSQL injection in parameters.
- **Information Leakage**: Ensuring error messages do not disclose database credentials, internal server paths, or framework versions.

### 3.6 Performance Testing
- Measuring batch scoring throughput on simulated network flow bursts.
- Benchmarking single-flow inference response time to determine suitability for real-time edge or gateway inspection (target: $< 20\text{ ms}$ per sample).

---

## 4. Test Environment

| Component | Technology / Tool |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Test Runner** | `pytest`, `pytest-cov`, `pytest-asyncio` |
| **API Testing** | `httpx`, `FastAPI TestClient` |
| **Deep Learning Framework** | `PyTorch` |
| **Data Processing** | `pandas`, `numpy`, `scikit-learn` |
| **Database** | PostgreSQL / SQLite (in-memory for unit testing) |
| **Benchmark Datasets** | CICIDS2017, CIC-IoT2023 |

---

## 5. Test Data Management

- **Synthetic Test Vectors**: Dedicated fixtures representing single-flow benign and attack records with controlled values.
- **Edge-Case Vectors**: Floats with `NaN`, `inf`, `-inf`, out-of-boundary values, negative packet lengths, and duplicate rows.
- **Benchmark Partitions**: Controlled subsamples of CICIDS2017 and CIC-IoT2023 stored under version-controlled test assets.

---

## 6. Defect Management & Reporting

- Any discrepancy, crash, performance regression, or validation failure must be logged via **GitHub Issues** using the standard issue templates:
  - `[BUG]`: For defects, crashes, incorrect predictions, or test failures.
  - `[FEATURE]`: For enhancements to preprocessing, new model layers, or API routes.
  - `[TASK]`: For backlog tracking and scheduled engineering work.
- Each issue must include reproduction steps, environment details, and relevant stack traces.

---

## 7. Test Results & Tracking

- Test cases are formally cataloged in [`docs/testing/TEST_CASES.md`](TEST_CASES.md).
- Status for each test case is tracked across iterations (`Planned`, `In Progress`, `Passed`, `Failed`, `Blocked`).
- Continuous Integration (CI) reports will record test passes, coverage percentages, and execution times upon each Pull Request.
