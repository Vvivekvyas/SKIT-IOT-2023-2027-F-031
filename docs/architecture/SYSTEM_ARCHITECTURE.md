# System Architecture

## 1. Overview

The **Hybrid ML Based Intrusion Detection System (IDS)** combines an unsupervised **Variational Autoencoder (VAE)** with a state-of-the-art tabular **Feature Tokenizer Transformer (FT-Transformer)** to deliver high-performance, multiclass network intrusion detection.

Tabular network flow data is characterized by high feature dimensionality, non-linear inter-feature correlations, extreme class imbalance, and heterogeneous numerical scales. This architecture addresses these challenges by first learning a continuous, regularized latent manifold using a VAE, and subsequently applying multi-head self-attention via an FT-Transformer to capture high-order feature interactions for fine-grained attack classification.

---

## 2. High-Level Architecture

```text
       +------------------------------------+
       |        Network Traffic Data        |
       |     (Flow Records / PCAP / CSV)    |
       +-----------------+------------------+
                         |
                         v
       +------------------------------------+
       |       Data Processing Engine       |
       |  - Missing / Infinity Cleaning     |
       |  - Robust / MinMax Feature Scaling |
       |  - Identifier Column Pruning       |
       |  - One-Hot / Target Label Encoding |
       +-----------------+------------------+
                         |
                         v
       +------------------------------------+
       |    Variational Autoencoder (VAE)   |
       |  +------------------------------+  |
       |  |  Encoder q_phi(z|x)          |  |
       |  |  Computes mu(x) & log_var(x) |  |
       |  +--------------+---------------+  |
       |                 |                  |
       |                 v                  |
       |  +------------------------------+  |
       |  | Reparameterization Trick:    |  |
       |  | z = mu + sigma (*) epsilon   |  |
       |  +------------------------------+  |
       +-----------------+------------------+
                         |
                         v
       +------------------------------------+
       |     Dense Latent Representation    |
       |         (Latent Vector z)          |
       +-----------------+------------------+
                         |
                         v
       +------------------------------------+
       |           FT-Transformer           |
       |  +------------------------------+  |
       |  | Feature Tokenizer            |  |
       |  | Linear projection per column |  |
       |  +--------------+---------------+  |
       |                 |                  |
       |                 v                  |
       |  +------------------------------+  |
       |  | Multi-Head Self-Attention    |  |
       |  | Captures inter-feature deps  |  |
       |  +--------------+---------------+  |
       |                 |                  |
       |                 v                  |
       |  +------------------------------+  |
       |  | Transformer Encoder Stack    |  |
       |  | LayerNorm + MLP + Residual   |  |
       |  +------------------------------+  |
       +-----------------+------------------+
                         |
                         v
       +------------------------------------+
       |   Multiclass Classification Head   |
       |      (Softmax Probabilities)       |
       +-----------------+------------------+
                         |
                         v
       +------------------------------------+
       |         Prediction Output          |
       | (Benign / PortScan / DDoS / etc.)  |
       +------------------------------------+
```

---

## 3. Core Components

### 3.1 Data Processing Engine
- **Ingestion**: Accepts tabular network flow records (e.g., CICFlowMeter format).
- **Sanitization**: Strips infinite values (`+inf`, `-inf`) and handles missing entries (`NaN`). Drops identifier artifacts (source/destination IP, ports, timestamps) to prevent model bias.
- **Normalization**: Scales continuous numerical features to zero-mean unit-variance or bounded $[0, 1]$ intervals, fitted strictly on the training partition.
- **Label Processing**: Encodes categorical labels into integer class indices for multiclass optimization.

### 3.2 Variational Autoencoder (VAE)
- **Role**: Performs non-linear dimensionality reduction and unsupervised manifold learning.
- **Mechanism**:
  - **Encoder**: Compresses input feature vector $x \in \mathbb{R}^D$ into distribution parameters: mean vector $\boldsymbol{\mu} \in \mathbb{R}^d$ and log-variance vector $\log \boldsymbol{\sigma}^2 \in \mathbb{R}^d$, where $d \ll D$.
  - **Reparameterization Trick**: Generates sample $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$ with $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$, allowing standard backpropagation.
  - **Regularization**: Penalizes deviation of the approximate posterior from a standard Gaussian prior using Kullback-Leibler (KL) divergence, preventing overfitting to anomalous noise.

### 3.3 FT-Transformer (Feature Tokenizer Transformer)
- **Role**: High-capacity classifier tailored for tabular and latent feature representations.
- **Mechanism**:
  - **Feature Tokenizer**: Projects each continuous feature (including latent dimensions $z$) into an embedding space $\mathbb{R}^E$ using dedicated linear layers, and appends a learnable `[CLS]` token.
  - **Self-Attention Layers**: Stacked multi-head self-attention blocks allow all features to attend to all other features dynamically, capturing subtle multi-feature attack signatures.
  - **Classification Head**: Extracts the transformed `[CLS]` token representation and passes it through an MLP with Softmax activation to generate probability distributions across attack classes.

### 3.4 Backend & API Service
- **Framework**: **FastAPI** asynchronous application.
- **Endpoints**:
  - `POST /api/v1/predict`: Real-time single flow classification.
  - `POST /api/v1/predict/batch`: High-throughput batch dataset classification.
  - `POST /api/v1/dataset/validate`: Validates uploaded CSV structure and schema.
  - `GET /api/v1/health`: System health and model loading status.

### 3.5 Storage & Audit Logging
- **Database**: **PostgreSQL** or structured append-only logging for:
  - Recorded security incidents and detected anomalies.
  - User and API authentication credentials.
  - Model versioning metadata and execution metrics.
