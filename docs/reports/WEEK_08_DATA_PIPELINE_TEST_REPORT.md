# Week 8: Data Pipeline Integration & Dataset Ingestion Test Report

**Project Title:** Hybrid ML Based Intrusion Detection System (VAE + FT-Transformer)  
**Milestone:** Week 8 — Data Pipeline Integration & Backend Ingestion Verification  
**Module Lead:** Ronak (@ronak647)  
**Repository:** [SKIT-IOT-2023-2027-F-031](https://github.com/Vvivekvyas/SKIT-IOT-2023-2027-F-031)  
**Test Suite Target:** `tests/test_data_pipeline_integration.py`  
**Status:** Completed & Verified (100% Pass Rate across 53 System Tests)  

---

## 📋 1. Executive Summary

During **Week 8**, the primary focus was establishing robust automated integration tests bridging the ML Data Pipeline (`src/data/`) with the backend API dataset ingestion service (`backend/app/services/data_pipeline.py`).

Following the backend dataset endpoint implementation (Week 8), this sprint ensured that:
1. Uploaded network flow captures are deterministically auto-detected (`CICIDS2017` vs `CIC-IoT2023`) based on schema feature overlap ratios.
2. The ML team's `DatasetValidator` executes seamlessly through the backend service interface (`run_validation`), generating structured diagnostics, handling schema corruption, and rejecting non-network payloads.
3. Memory boundaries are strictly maintained via row-sampling limits (`VALIDATION_SAMPLE_ROWS`) when processing high-volume traffic captures.
4. Validated flow records flow directly into `DataPipeline.fit_transform()` to generate stratified partitions, scale transformations, and serialize `feature_columns.json` schema contracts for downstream model consumption.
5. Micro-batch streaming throughput meets near-real-time ingestion criteria (> 1,000 flows/sec).

---

## 📊 2. Test Execution Matrix (TC-041 to TC-051)

| Test ID | Test Function | Target Scope | Expected Result | Status |
| :--- | :--- | :--- | :--- | :---: |
| **TC-041** | `test_tc_041_detect_dataset_type_cicids2017` | Dataset type detection | Resolves `CICIDS2017` when column match $\ge 50\%$ | **PASSED** |
| **TC-042** | `test_tc_042_detect_dataset_type_ciciot2023` | Dataset type detection | Resolves `CIC-IoT2023` when column match $\ge 50\%$ | **PASSED** |
| **TC-043** | `test_tc_043_detect_dataset_type_unknown_columns` | Schema classification | Returns `None` for unrecognised/sub-threshold columns | **PASSED** |
| **TC-044** | `test_tc_044_run_validation_valid_cicids2017` | Ingestion validation | Generates valid report for clean CICIDS2017 CSV, `valid=True` | **PASSED** |
| **TC-045** | `test_tc_045_run_validation_valid_ciciot2023` | Ingestion validation | Generates valid report for clean CIC-IoT2023 CSV, `valid=True` | **PASSED** |
| **TC-046** | `test_tc_046_run_validation_unrecognized_dataset` | Rejection & error handling | Returns `valid=False`, `report=None`, explanatory message | **PASSED** |
| **TC-047** | `test_tc_047_run_validation_corrupted_dataset` | Mandatory column checks | Flags missing features, returns comprehensive error report | **PASSED** |
| **TC-048** | `test_tc_048_run_validation_sampling_enforcement` | Memory optimization | Caps rows checked to sample limit, sets `sampled=True` | **PASSED** |
| **TC-049** | `test_tc_049_pipeline_unavailable_error_handling` | Graceful degradation | Raises `PipelineUnavailable` on missing components | **PASSED** |
| **TC-050** | `test_tc_050_end_to_end_validated_dataset_to_model_pipeline` | Full lifecycle integration | Validated CSV $\to$ DataPipeline fit $\to$ artifacts export | **PASSED** |
| **TC-051** | `test_tc_051_pipeline_microbatch_streaming_throughput` | Streaming performance | Throughput $> 1,000$ flows/s with zero NaNs/infs | **PASSED** |

---

## 📈 3. Performance & Audit Metrics

```
Total Automated Test Cases: 53 / 53 Passed (100%)
New Week 8 Test Cases:      11 / 11 Passed (100%)
Suite Execution Duration:   12.08s
Micro-Batch Throughput:     > 1,400 flows/sec (Chunk size: 100)
Data Leakage Rate:          0.0%
Feature Scaling Bounds:     [0.0, 1.0] strictly verified
```

---

## 🏁 4. Handoff to Week 9 (Deep Hybrid Modeling)

With the end-to-end data pipeline thoroughly validated and connected to backend services:
- **Ingestion Contract:** Ingested flow datasets reliably serialize `feature_columns.json` and fitted scalers.
- **Handoff (Weeks 8–9):** Variational Autoencoder (VAE) Latent Space Formulation & FT-Transformer classification heads.
