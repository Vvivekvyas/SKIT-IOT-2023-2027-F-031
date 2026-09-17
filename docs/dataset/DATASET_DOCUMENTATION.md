# Dataset Documentation

## 1. Purpose

This document provides technical documentation for the benchmark datasets utilized to train, validate, and evaluate the **Hybrid ML Based Intrusion Detection System (IDS)**. It details dataset origins, attack class taxonomies, the end-to-end data pipeline, data leakage safeguards, and security measures.

---

## 2. CICIDS2017 Dataset

### 2.1 Overview & Source
- **Provider**: Canadian Institute for Cybersecurity (CIC), University of New Brunswick (UNB).
- **Domain**: Modern enterprise network environments.
- **Acquisition Protocol**: Captured using a B-Profile system covering realistic background user traffic (HTTP, HTTPS, FTP, SSH, Email) combined with synchronized attack profiles executed over five consecutive days.
- **Format**: 8 CSV flow capture files extracted using CICFlowMeter, generating 78+ statistical network flow attributes.

### 2.2 Attack Profiles & Class Distribution
The dataset captures the following core attack profiles:
1. **Benign**: Normal operational network activities.
2. **Brute Force**: FTP-Patator, SSH-Patator.
3. **DoS / DDoS**: DoS Slowloris, DoS Slowhttptest, DoS Hulk, DoS GoldenEye, Heartbleed, DDoS LOIC.
4. **Web Attacks**: SQL Injection, Cross-Site Scripting (XSS), Web Brute Force.
5. **Infiltration**: Internal network reconnaissance and privilege escalation.
6. **Botnet**: ARES botnet command-and-control communications.
7. **PortScan**: Stealthy and aggressive port sweeps.

### 2.3 Preprocessing Strategy
- **Header Sanitization**: Strip leading/trailing whitespace from column headers (e.g., `' Destination Port'`, `' Flow Duration'`).
- **Anomalous Values**: Replace `+inf`, `-inf`, and divide-by-zero artifacts with numerical nulls (`NaN`), followed by feature-wise median imputation fitted solely on training splits.
- **Identifier Pruning**: Exclude non-generalizable network identifiers (Source IP, Destination IP, Timestamp, Source Port) to force the model to learn flow behavioral patterns rather than memorize IP addresses.
- **Normalization**: Apply `RobustScaler` or `StandardScaler` to handle extreme outliers in flow durations and packet byte counts.

---

## 3. CIC-IoT2023 Dataset

### 3.1 Overview & Source
- **Provider**: Canadian Institute for Cybersecurity (CIC).
- **Domain**: Internet of Things (IoT) and Industrial IoT infrastructure.
- **Topology**: Captured across an extensive testbed incorporating 105 distinct smart devices (cameras, smart home sensors, microcontrollers) executing 33 diverse cyberattacks alongside legitimate smart traffic.

### 3.2 Attack Taxonomies
CIC-IoT2023 groups its 33 granular attack variants into 7 macro categories:
1. **DDoS Attacks**: ICMP Flood, UDP Flood, TCP SYN Flood, HTTP Flood, etc.
2. **DoS Attacks**: TCP DoS, UDP DoS, ICMP DoS.
3. **Reconnaissance**: OS Fingerprinting, Host Discovery, Vulnerability Scanning, Ping Sweep.
4. **Web-Based Attacks**: SQL Injection, Command Injection, Backdoor.
5. **Brute Force**: Telnet, SSH, and Web credential attacks on IoT devices.
6. **Spoofing**: ARP Spoofing, DNS Spoofing.
7. **Mirai Botnet**: Mirai-greeth_flood, Mirai-udpplain, Mirai-ackflag.

### 3.3 Target Usage
CIC-IoT2023 evaluates the hybrid model's ability to generalize to resource-constrained IoT architectures where traffic patterns feature high periodicity, small packet bursts, and high susceptibility to volumetric botnet swarms.

---

## 4. End-to-End Dataset Pipeline

```text
  +--------------------------------------------+
  |              Raw Dataset CSVs              |
  |         (CICIDS2017 / CIC-IoT2023)         |
  +---------------------+----------------------+
                        |
                        v
  +--------------------------------------------+
  |              Data Sanitization             |
  | - Remove duplicate records                 |
  | - Handle NaN / Infinity values             |
  | - Strip non-generalizable identifiers      |
  +---------------------+----------------------+
                        |
                        v
  +--------------------------------------------+
  |        Train / Val / Test Partition        |
  | - Stratified Split: 70% Train, 15% Val, 15%|
  +---------------------+----------------------+
                        |
         +--------------+--------------+
         | (Fit & Transform)           | (Transform Only)
         v                             v
  +--------------------+      +--------------------+
  | Feature Scaling &  |      | Validation & Test  |
  | Label Encoding on  |      | Transformations    |
  | Training Partition |      | (Zero Data Leakage)|
  +---------+----------+      +---------+----------+
            |                           |
            +-------------+-------------+
                          |
                          v
  +--------------------------------------------+
  |         Engineered Tensors for VAE         |
  |              & FT-Transformer              |
  +--------------------------------------------+
```

---

## 5. Data Leakage Prevention Safeguards

To maintain strict scientific validity and avoid data leakage:
1. **Split-Before-Fit Principle**: Stratified splitting into Train (70%), Validation (15%), and Test (15%) sets must occur **before** computing any scaling parameters (mean, standard deviation, min, max) or imputing missing values.
2. **Stateless Test Transformations**: The test and validation sets must only be transformed using statistics learned from the training set. No test distribution data is ever exposed during feature engineering.
3. **Temporal Awareness**: When chronological order is preserved, sequence-based splits must avoid using future packet bursts to predict past anomalies.

---

## 6. Dataset Security & Ingestion Safety

When accepting external CSV files via APIs:
- **File Validation**: Enforce strict MIME-type checking (`text/csv`), file extension validation, and maximum upload size limits ($< 250\text{ MB}$).
- **DoS Prevention**: Stream CSV ingestion chunk-by-chunk using generator streams rather than reading entire large files into RAM at once.
- **Malicious Payload Guard**: Sanitize CSV content against formula injection (e.g., entries starting with `=`, `+`, `-`, `@`).

---

## 7. Dataset Versioning & Experiment Tracking

Every training run records:
- SHA-256 checksum of raw dataset files used.
- Configuration file (`config.yaml`) specifying:
  - Selected feature subset.
  - Imputation strategy.
  - Scaler type.
  - Random seed used for stratified partitioning.
  - Class mapping dictionary.

---

## 8. Post-Processing Profile (To Be Populated)

> [!NOTE]
> The exact empirical statistics below will be calculated and documented following execution of the automated exploratory data analysis (EDA) pipeline in Weeks 4–5.

- **Total Samples Cleaned**: `[Pending EDA]`
- **Feature Count (Numerical / Categorical)**: `[Pending EDA]`
- **Attack Class Distribution (Sample Count & %)**: `[Pending EDA]`
- **Null Value / Imputation Count**: `[Pending EDA]`
- **Train / Validation / Test Sample Counts**: `[Pending EDA]`
