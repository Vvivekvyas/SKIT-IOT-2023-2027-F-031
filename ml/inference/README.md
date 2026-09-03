\# ML Inference



This module will provide the inference pipeline for the trained

Hybrid FT-Transformer + Autoencoder IDS model.



\## Expected Flow



Input Network Features
      ↓

Preprocessing
      ↓

Trained Hybrid Model
      ↓

Prediction
      ↓

Attack Class
      ↓

Confidence / Anomaly Score
      ↓

Backend



\## Expected Output



```json

{

  "prediction": "DDoS",

  "confidence": 0.96,

  "anomaly\_score": 0.12

}

