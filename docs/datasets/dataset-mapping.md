# IDS Dataset Mapping

## 1. Datasets Used

The Hybrid ML-Based IDS uses two network intrusion detection
datasets:

1. CICIDS2017
2. CIC-IoT2023

## 2. Dataset Comparison

+----------------------+------------------------------+------------------------------+
| Property             | CICIDS2017                   | CIC-IoT2023                  |
+----------------------+------------------------------+------------------------------+
| Domain               | Network intrusion detection  | IoT network intrusion        |
|                      |                              | detection                    |
+----------------------+------------------------------+------------------------------+
| Format               | CSV                          | CSV                          |
+----------------------+------------------------------+------------------------------+
| Organization         | Multiple traffic/capture     | Multiple traffic/attack      |
|                      | files                        | files                        |
+----------------------+------------------------------+------------------------------+
| Traffic Type         | Network traffic              | IoT network traffic          |
+----------------------+------------------------------+------------------------------+
| Benign Traffic       | Available                    | Available                    |
+----------------------+------------------------------+------------------------------+
| Attack Traffic       | Available                    | Available                    |
+----------------------+------------------------------+------------------------------+
| ML Usage             | Classification and           | Classification and           |
|                      | anomaly detection            | anomaly detection            |
+----------------------+------------------------------+------------------------------+
| Primary Application  | General network intrusion    | IoT intrusion detection      |
|                      | detection                    |                              |
+----------------------+------------------------------+------------------------------+
| Feature Structure    | Network flow-based features  | IoT network flow/traffic     |
|                      |                              | features                     |
+----------------------+------------------------------+------------------------------+
| Label Representation | Attack/benign labels         | Attack/benign attack labels  |
+----------------------+------------------------------+------------------------------+
| Project Role         | Training and evaluation      | Training and evaluation      |
+----------------------+------------------------------+------------------------------+

## 3. Project Usage

Both datasets will pass through a common ML processing pipeline:

Dataset
→ Loading
→ Cleaning
→ Feature selection
→ Encoding
→ Scaling
→ Data splitting
→ Model training
→ Evaluation

## 4. Model Usage

The datasets will support development and evaluation of:

- Baseline ML models
- FT-Transformer
- Autoencoder
- Hybrid FT-Transformer + Autoencoder

## 5. Important Considerations

The datasets have different file organizations and potentially
different feature/label representations.


Therefore, dataset-specific loading and validation logic may be
required before both datasets are passed to the common ML pipeline.

## 6. Raw Data Policy

Raw datasets will be stored locally and will not be committed
to the Git repository.

Only dataset documentation, loading scripts, analysis notebooks,
and configuration files will be committed.