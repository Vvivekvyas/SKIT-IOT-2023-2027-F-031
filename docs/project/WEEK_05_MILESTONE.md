# Project Milestone 5: Preprocessing Test Report & Quality Assurance

**Milestone Identifier:** `MS-05`  
**Milestone Name:** Week 5: Preprocessing Test Report & Quality Assurance  
**Lead Assignee:** Ronak ([@ronak647](https://github.com/ronak647))  
**Repository:** [SKIT-IOT-2023-2027-F-031](https://github.com/Vvivekvyas/SKIT-IOT-2023-2027-F-031)  
**Timeline / Sprint:** Week 5  
**Milestone State:** `Closed / Completed`  

---

## 🎯 1. Milestone Overview & Objectives

The primary objective of **Milestone 5** is to establish full verification, regression testing, and quality assurance for the data sanitization, leakage elimination, and feature scaling subsystems developed in Week 4.

Network flow records from **CICIDS2017** and **CIC-IoT2023** exhibit severe data hazards:
- Socket metadata (IPs, Ports, Timestamps) that provoke model shortcut learning.
- Micro-burst flow records yielding infinite flow byte and packet rates ($\pm\infty$).
- Heavy-tailed numerical distributions requiring robust scaling bounded strictly within $[0.0, 1.0]$.
- Highly skewed class distributions requiring strict stratification.

This milestone ensures that every component is rigorously tested against both nominal and adversarial inputs, that test results are fully documented in a technical test report, and that 0 data leakage enters downstream training folds.

---

## 📋 2. Linked Milestone Issues & Deliverables

| Issue # | Issue Title | Module / Target | Assignee | Status |
| :---: | :--- | :--- | :---: | :---: |
| **#10** | **Automated Test Execution for Data Sanitization & Leakage** | `tests/test_data_cleaner.py` | Ronak | **Closed** |
| **#11** | **Feature Scaling Bounds & Stratification Split Testing** | `tests/test_preprocessor.py` | Ronak | **Closed** |
| **#12** | **Schema Contract & Anomaly Detection Verification** | `tests/test_data_validation.py` | Ronak | **Closed** |
| **#13** | **Technical Preprocessing Test Report Compilation** | `docs/reports/PREPROCESSING_TEST_REPORT.md` | Ronak | **Closed** |
| **#14** | **Data Leakage & Train-Only Parameter Isolation Audit** | `src/data/` | Ronak | **Closed** |

---

## 🔍 3. Issue Breakdown & Scope of Work

### Issue #10: Data Sanitization & Cleaning Engine Verification
- **Target:** `src/data/cleaner.py` via `tests/test_data_cleaner.py`
- **Scope:**
  - Verify removal of all socket identifiers (`Source IP`, `Destination IP`, `Source Port`, `Timestamp`).
  - Verify exact row deduplication (105 rows $\to$ 100 rows).
  - Verify median imputation of $\pm\infty$ and $\text{NaN}$ values strictly computed from training folds.
  - Verify dropping of constant zero-variance features (`Fwd PSH Flags`).
  - Verify consolidation of 33 granular IoT attacks into 8 macro families.
- **Outcome:** All 5 unit test cases passing.

### Issue #11: Feature Scaling & Stratified Split Testing
- **Target:** `src/data/preprocessor.py` via `tests/test_preprocessor.py`
- **Scope:**
  - Verify 70/15/15 stratified train/val/test split preserves class representation across all target categories.
  - Verify MinMax scaling bounds $X_{\text{train}}, X_{\text{test}} \in [0.0, 1.0]$ with outlier clipping.
  - Verify bidirectional label encoding ($y \in [0, K-1]$) with invertible class mapping.
  - Verify guard against unfitted `transform()` calls (`RuntimeError`).
- **Outcome:** All 4 unit test cases passing.

### Issue #12: Schema Contract & Anomaly Detection Verification
- **Target:** `src/data/validator.py`, `src/data/schemas.py` via `tests/test_data_validation.py`
- **Scope:**
  - Verify TC-001 (CICIDS2017 compliant schema validation).
  - Verify TC-001b (CIC-IoT2023 compliant schema validation).
  - Verify TC-003 (Empty dataset rejection).
  - Verify TC-004 (Missing mandatory flow features detection).
  - Verify TC-005 (Missing label column detection).
  - Verify TC-006 (Infinite rate detection and NaN ratio threshold enforcement).
  - Verify TC-007 (Duplicate row detection).
- **Outcome:** All 7 validation test cases passing.

### Issue #13: Technical Preprocessing Test Report Compilation
- **Target:** `docs/reports/PREPROCESSING_TEST_REPORT.md`
- **Scope:**
  - Author comprehensive test report detailing executive summary, test execution matrices, component-by-component evaluations, quantitative coverage metrics, and reproducibility commands.
- **Outcome:** Published and archived in `docs/reports/`.

---

## ✅ 4. Definition of Done (DoD) & Acceptance Criteria

- [x] **Zero Test Failures:** All 16 automated tests pass cleanly with 100% success rate.
- [x] **Zero Data Leakage:** Mathematical verification that `fit()` is computed strictly on training records, and zero socket metadata enters feature tensors.
- [x] **Strict Bounding:** Transformed features strictly fall in $[0.0, 1.0]$.
- [x] **Invertible Encoding:** Bijective mapping from integer index to attack class name verified without information loss.
- [x] **Formal Reporting:** Comprehensive test report compiled and indexed in repository documentation.
- [x] **Traceability:** Test cases directly mapped to Initial Test Cases Matrix (TC-001 through TC-007).

---

## 📈 5. Milestone Metrics

```
Total Milestone Tasks:   5 / 5 Completed (100%)
Automated Test Cases:    16 / 16 Passed (100%)
Test Execution Duration: 0.48s
Data Leakage Rate:       0.0%
Code Coverage:           > 96% across src/data/
```

---

## 🗺️ 6. Ronak's Sprint Roadmap

```
├── Week 4:  Data Validation Test Suite + Preprocessing Pipeline       [COMPLETED]
├── Week 5:  Preprocessing Test Report + GitHub Milestone (MS-05)     [COMPLETED]
├── Week 6:  EDA Documentation + Reproducibility Checks               [PLANNED]
└── Week 7:  Data Pipeline Tests                                      [PLANNED]
```

* **Next Sprint (Week 6):** Exploratory Data Analysis (EDA) documentation, feature distribution profiling, attack frequency visualizations, and cross-platform reproducibility validation.
