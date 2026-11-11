# Hybrid IDS Prediction Model Checkpoint & Service Specification

**Module:** Machine Learning & Backend Model Service Interface  
**Author:** Nancy ([@NancyJain1234](https://github.com/NancyJain1234))  
**Consumers:** `backend/app/services/predictor.py`, `backend/app/api/v1/predict.py`  
**Standard Artifact Location:** `ml/artifacts/hybrid/hybrid_model.pt`

---

## 1. Checkpoint Bundle Schema Contract

The serialized checkpoint file is saved via `torch.save()` and loaded using `torch.load(path, weights_only=True)`. It must contain the following keys:

| Key | Type | Description |
| :--- | :--- | :--- |
| `feature_names` | `list[str]` | Ordered list of expected input tabular network features ($1 \le N \le 500$). |
| `class_names` | `list[str]` | Multiclass label mapping (e.g. `["BENIGN", "DDoS", "PortScan"]`). |
| `classifier_config` | `dict` | Hyperparameters for `FTTransformer` (`embedding_dim`, `num_heads`, `num_layers`, `ffn_factor`, `dropout`). |
| `classifier_state` | `dict` | Serialized `state_dict` of the trained `FTTransformer` classifier. |
| `autoencoder_config` | `dict` | Hyperparameters for `DeepAutoencoder` (`latent_dim`, `hidden_dims`, `dropout`). |
| `autoencoder_state` | `dict` | Serialized `state_dict` of the trained `DeepAutoencoder`. |
| `anomaly_threshold` | `float` | Calibrated reconstruction error threshold ($99^{\text{th}}$ percentile on benign traffic). |
| `confidence_threshold` | `float` | Minimum classifier confidence threshold (default `0.85`). |
| `anomaly_flag_threshold` | `float` | Anomaly score cutoff for flagging zero-day threats (default `0.70`). |
| `scaler_center` | `list[float]` or `None` | Preprocessing scaler mean/center per feature vector. |
| `scaler_scale` | `list[float]` or `None` | Preprocessing scaler scale/standard deviation per feature vector. |
| `model_version` | `str` | Semantic model version identifier (e.g. `"hybrid-ids-v1.0.0"`). |

---

## 2. Security & Model Integrity Validation

To satisfy the system security requirements (OWASP API Top 10 and Model-Serving Security):
1. **Cryptographic Checksums:** Each production artifact is published with an accompanying SHA-256 hash manifest (`hybrid_model.pt.sha256` or `manifest.json`).
2. **Safe Deserialization:** The backend service loads weights with `weights_only=True`, preventing arbitrary code execution.
3. **Input Sanitization:** Features are scrubbed of `NaN`/`Inf` and clipped against extreme outliers prior to inference via `FeatureSanitizer`.
4. **Threat Triage:** Anomalous traffic classified as benign is triaged as `ZERO_DAY_SUSPECT` to isolate novel attack patterns.

---

## 3. REST API Contract Alignment

The backend prediction endpoint `POST /api/v1/predict` receives flat feature payloads and returns the triaged classification:

```json
{
  "features": {
    "flow_duration": 1420.0,
    "total_fwd_packets": 12.0,
    "total_bwd_packets": 8.0
  }
}
```

Response payload:
```json
{
  "prediction_id": "84cfbbd5-f857-4ba4-9720-c3d3f9b20bfa",
  "classification": "DDoS",
  "confidence": 0.9625,
  "anomaly_score": 0.1240,
  "is_alert": true,
  "model_version": "hybrid-ids-v1.0.0"
}
```
