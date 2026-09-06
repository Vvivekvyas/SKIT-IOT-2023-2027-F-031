# CICIDS2017 Dataset Study

## 1. Dataset Purpose

CICIDS2017 is used as one of the network intrusion detection datasets
for the Hybrid ML-Based Intrusion Detection System.

The dataset contains benign network traffic as well as multiple
types of malicious network attacks.

## 2. Dataset Organization

The dataset is available as multiple CSV files.

Examples include:

- Monday-WorkingHours.pcap_ISCX.csv
- Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
- Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
- Other CICIDS2017 CSV files

## 3. Observed Structure

Each CSV contains network-flow records represented by multiple
features and a label column.

The original CSV files contain approximately 79 columns.

## 4. Observed Attack Categories

The dataset contains labels including:

- BENIGN
- DDoS
- PortScan
- DoS Hulk
- DoS GoldenEye
- FTP-Patator
- SSH-Patator
- DoS slowloris
- DoS Slowhttptest
- Bot
- Web Attack – Brute Force
- Web Attack – XSS
- Infiltration
- Web Attack – Sql Injection
- Heartbleed

## 5. Dataset Usage in the Project

CICIDS2017 will be used for:

1. Dataset analysis
2. Data preprocessing
3. Exploratory data analysis
4. Baseline model training
5. FT-Transformer training
6. Model evaluation
7. Hybrid IDS development

## 6. Initial Observations

The dataset contains significant class imbalance.

Benign traffic represents a large portion of the dataset, while
some attack classes contain relatively few samples.

Therefore, class distribution and imbalance will be investigated
before model training.

## 7. Planned Processing

The raw dataset will not be modified directly.

The planned pipeline is:

Raw CSV
→ Loading
→ Cleaning
→ Feature/label identification
→ Encoding
→ Scaling
→ Train/Validation/Test split
→ Model training

Detailed preprocessing will be implemented in later project weeks.