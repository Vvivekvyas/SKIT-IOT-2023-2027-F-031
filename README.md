# Hybrid ML Based Intrusion Detection System

An advanced, deep-learning-driven Network Intrusion Detection System (NIDS) designed to detect and categorize malicious network activities and cyber threats across enterprise and Internet of Things (IoT) environments. The system leverages a **Hybrid Deep Learning Architecture** combining a **Variational Autoencoder (VAE)** for unsupervised non-linear latent feature extraction and an **FT-Transformer (Feature Tokenizer Transformer)** for high-precision multiclass classification of tabular network flow data.

---

## 📌 Project Overview

Network security operations face severe challenges due to exponential increases in encrypted traffic volumes, zero-day vulnerabilities, evolving attack vectors, and high data dimensionality. Traditional signature-based systems fail against novel attacks, while standard machine learning classifiers struggle with tabular network flow characteristics, feature correlation complexity, and class imbalances.

This project introduces a hybrid framework:
1. **Dimensionality Reduction & Representation Learning**: A **Variational Autoencoder (VAE)** compresses raw, high-dimensional tabular flow features into a dense, regularized latent space, mitigating noise and capturing complex underlying distribution manifolds.
2. **Tabular Deep Learning Classification**: The **FT-Transformer (Feature Tokenizer Transformer)** processes the learned latent representations and critical tabular features using multi-head self-attention mechanisms, capturing subtle inter-feature dependencies to achieve robust multi-class threat classification.

---

## 🎯 Objectives

- **High Detection Accuracy**: Maximize detection rates on low-footprint, sophisticated modern attacks (e.g., Botnets, Infiltration, Port Scans, DDoS).
- **Reduced False Alarm Rate (FAR)**: Minimize false positive rates on benign traffic to avoid security analyst fatigue.
- **Robust Feature Representation**: Overcome tabular data limitations through latent embeddings extracted via VAE.
- **Support for Heterogeneous Environments**: Evaluate on both enterprise network traffic (**CICIDS2017**) and IoT sensor networks (**CIC-IoT2023**).
- **Scalable Modular Design**: Decouple data ingestion, preprocessing, inference, and API serving to enable seamless production deployment.

---

## 🧠 Proposed Architecture

The core pipeline combines unsupervised representation learning with attention-driven tabular classification:

```text
  +-------------------------------+
  |     Network Traffic Data      |
  |    (CICIDS2017 / CIC-IoT2023) |
  +---------------+---------------+
                  |
                  v
  +-------------------------------+
  |       Data Preprocessing      |
  |  (Cleaning, Scaling, Encoding)|
  +---------------+---------------+
                  |
                  v
  +-------------------------------+
  |  Variational Autoencoder(VAE) |
  |  - Encoder (q_phi(z|x))       |
  |  - Reparameterization (mu, sig|
  |  - Regularized Latent Space   |
  +---------------+---------------+
                  |
                  v
  +-------------------------------+
  |     Latent Representation     |
  |    (Low-dimensional Vector z) |
  +---------------+---------------+
                  |
                  v
  +-------------------------------+
  |        FT-Transformer         |
  |  - Feature Tokenization Layer |
  |  - Multi-Head Self-Attention  |
  |  - Feed-Forward Transformer   |
  +---------------+---------------+
                  |
                  v
  +-------------------------------+
  | Multiclass Classification Head|
  |   (Softmax Probability Layer) |
  +---------------+---------------+
                  |
                  v
  +-------------------------------+
  |   Predicted Attack Category   |
  |  (Benign / DDoS / Recon / etc)|
  +-------------------------------+
```

---

## 🔄 System Workflow

1. **Ingestion**: Network packet captures or flow records (CSV/NetFlow/IPFIX) are collected.
2. **Preprocessing**:
   - Infinity and missing value treatment.
   - Removal of duplicate records and identifier columns (e.g., IP addresses, ports, timestamps) to prevent model overfitting and shortcut learning.
   - Numerical feature normalization (Min-Max Scaling / Robust Scaling).
   - Categorical feature encoding.
3. **VAE Encoding**:
   - High-dimensional input vector $x$ is mapped to mean ($\mu$) and log-variance ($\log \sigma^2$) vectors.
   - Sampled latent vector $z = \mu + \sigma \odot \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$ creates a regularized low-dimensional representation.
4. **FT-Transformer Processing**:
   - The latent vector and selected prominent tabular features are transformed into dense continuous embeddings via the Feature Tokenizer.
   - Stacked Transformer layers apply multi-head self-attention to model inter-feature interactions.
5. **Inference & Alerting**:
   - Classification head outputs class probabilities.
   - Alerts are logged and surfaced through the REST API.

---

## 📊 Datasets

The system is developed and benchmarked on two benchmark intrusion detection datasets:

| Dataset | Domain | Scope & Characteristics | Target Attack Classes |
| :--- | :--- | :--- | :--- |
| **CICIDS2017** | Enterprise Network | Comprehensive real-world traffic generated by the Canadian Institute for Cybersecurity. | Benign, DoS/DDoS, PortScan, Botnet, Infiltration, Web Attacks, Brute Force |
| **CIC-IoT2023** | IoT Infrastructure | Scalable IoT security dataset collected from 105 smart devices under 33 diverse cyberattacks. | Benign, DDoS/DoS, Mirai, Reconnaissance, Spoofing, Brute Force, Web-based |

*Detailed specifications, feature distributions, and leakage prevention rules are documented in [`docs/dataset/DATASET_DOCUMENTATION.md`](docs/dataset/DATASET_DOCUMENTATION.md).*

---

## 🤖 Machine Learning Models

### 1. Variational Autoencoder (VAE)
- **Role**: Unsupervised feature representation and dimensionality reduction.
- **Components**:
  - **Encoder Network**: Maps $D$-dimensional input $x$ to latent parameters $\mu(x)$ and $\Sigma(x)$.
  - **Reparameterization Trick**: Ensures end-to-end gradient backpropagation.
  - **Decoder Network**: Reconstructs original features from latent vector $z$.
  - **Loss Function**: $\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{reconstruction}} + \beta \cdot D_{\text{KL}}(q_\phi(z|x) \parallel p(z))$.

### 2. Feature Tokenizer Transformer (FT-Transformer)
- **Role**: Multiclass classification on tabular and latent representations.
- **Components**:
  - **Feature Tokenizer**: Projects continuous and discrete features into an embedding space of dimension $d_{\text{token}}$.
  - **Transformer Encoder Blocks**: Multi-head self-attention (MHSA) capturing feature-to-feature interactions followed by feedforward layers and LayerNorm.
  - **[CLS] Token Classification Head**: Collects aggregated context for softmax multiclass output.



## 🔐 Security

- **Safe Parsing & Uploads**: Rigorous schema validation and file size restrictions for uploaded dataset files.
- **Input Sanitization**: Rejection of malformed flow records and malicious data payloads.
- **Authentication**: JWT-based authentication for backend API endpoints.
- **Data Privacy**: No sensitive raw packet payload or identifiable personal information is stored.

---

## 📈 Expected Outcomes

- A validated hybrid deep learning pipeline demonstrating superior detection rates on minority attack classes compared to baseline classifiers.
- High-throughput flow processing capable of near real-time intrusion categorization.
- Comprehensive end-to-end documentation, test suites, and repeatable experiment pipelines.

---

## 🚀 Future Scope

- **Real-Time Stream Processing**: Integration with Apache Kafka or eBPF packet capture engines.
- **Explainable AI (XAI)**: Integration of SHAP/LIME to explain attention weights and individual feature contributions to security analysts.
- **Edge Deployment**: Model quantization and pruning (ONNX/TensorRT) for deployment on IoT gateways. 
