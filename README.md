# 🛡️ Hybrid Machine Learning Based Intrusion Detection System (IDS)

> 🚀 A hybrid deep learning-based Intrusion Detection System that combines **Variational Autoencoder (VAE)** for representation learning with **FT-Transformer** for multiclass network intrusion classification.

---

## 📌 About The Project

Modern computer networks are becoming larger, faster, and more complex, while cyberattacks are becoming increasingly sophisticated.

Traditional Intrusion Detection Systems often face challenges such as:

* 🔹 High-dimensional network traffic
* 🔹 Noisy and redundant features
* 🔹 Rare or minority attack classes
* 🔹 Attacks that closely resemble normal traffic
* 🔹 Complex relationships between network traffic features

To address these challenges, this project proposes a **Hybrid Machine Learning Based Intrusion Detection System** that combines two deep learning approaches:

* 🧠 **Variational Autoencoder (VAE)** for learning compact and meaningful representations of network traffic
* 🎯 **FT-Transformer** for attention-based multiclass classification

The goal is to develop an IDS that can accurately detect different types of network attacks while also exploring **model interpretability and statistical reliability**.

---

## 🎯 Objectives

The main objectives of this project are:

* 🧠 Develop a hybrid **VAE + FT-Transformer** architecture.
* 🔍 Learn compact and meaningful representations of network traffic.
* 🎯 Perform **multiclass network intrusion detection**.
* ⚠️ Improve detection of minority and overlapping attack classes.
* 📊 Compare the proposed model with existing baseline models.
* 📈 Evaluate the model using multiple performance metrics.
* 🧪 Analyze statistical significance using **bootstrap confidence intervals** and **paired hypothesis testing**.
* 🔎 Explore FT-Transformer attention weights for model interpretability.
* 🌐 Provide a secure backend/API for interacting with the IDS model.

---

## 🏗️ Proposed Architecture

```text
              🌐 Network Traffic
                      │
                      ▼
           📂 CICIDS2017 / CIC-IoT2023
                      │
                      ▼
             ⚙️ Data Preprocessing
                      │
                      ▼
             ┌─────────────────┐
             │       VAE       │
             │                 │
             │ Encoder →       │
             │ Latent Space →  │
             │ Decoder          │
             └────────┬────────┘
                      │
                      ▼
            🧩 Latent Representation
                      │
                      ▼
             ┌─────────────────┐
             │ FT-Transformer  │
             │                 │
             │ Self-Attention  │
             │       ↓         │
             │ Classification  │
             └────────┬────────┘
                      │
                      ▼
              🚨 Attack Detection
                      │
                      ▼
             📊 Multiclass Prediction
```

---

## 🧠 Machine Learning Approach

### 1️⃣ Variational Autoencoder (VAE)

The **VAE** is used as the representation learning component of the system.

It learns a compact representation of network traffic by encoding the input features into a probabilistic latent space.

```text
Network Features
       ↓
    Encoder
       ↓
  Latent Space
       ↓
    Decoder
```

The learned latent representation is then passed to the FT-Transformer.

---

### 2️⃣ FT-Transformer

The **FT-Transformer** is used as the classification component.

It uses the Transformer architecture and **self-attention mechanisms** to learn relationships between network traffic features.

```text
VAE Latent Features
        ↓
   FT-Transformer
        ↓
   Self-Attention
        ↓
 Multiclass Classification
```

---

### 3️⃣ Hybrid Model

The proposed system combines both models:

```text
             Input Traffic
                  ↓
                 VAE
                  ↓
       Latent Representation
                  ↓
          FT-Transformer
                  ↓
        Attack Classification
```

This combination aims to benefit from:

* 🧩 Generative representation learning
* 🎯 Attention-based classification

---

## 📚 Datasets

The project will use established network intrusion detection datasets, including:

### 🔹 CICIDS2017

A widely used network intrusion detection dataset containing **benign traffic and multiple types of network attacks**.

### 🔹 CIC-IoT2023

A network traffic dataset focused on **IoT environments** and different attack scenarios.

The datasets will be preprocessed before being provided to the proposed hybrid model.

---

## 🚀 Expected Outcome

The final system aims to provide a complete hybrid IDS capable of:

```text
📥 Receive Network Traffic
          ↓
⚙️ Preprocess Features
          ↓
🧩 Learn Latent Representation
       using VAE
          ↓
🧠 Classify using FT-Transformer
          ↓
🚨 Detect Network Intrusions
          ↓
📊 Generate Prediction & Confidence
          ↓
🔐 Securely Serve Results through API
```

The research will mainly focus on:

* Minority attack detection
* Overlapping attack classes
* Statistical reliability
* Model interpretability

---

## 👥 Project Information

| **Field**            | **Details**                                              |
| -------------------- | -------------------------------------------------------- |
| **Project Title**    | Hybrid Machine Learning Based Intrusion Detection System |
| **Project Type**     | 🎓 Academic / Research Project                           |
| **Focus**            | 🛡️ Network Security                                     |
| **Machine Learning** | 🤖 Machine Learning & Deep Learning                      |
| **Architecture**     | 🧠 VAE + FT-Transformer                                  |
| **Task**             | 🎯 Multiclass Intrusion Detection                        |
| **Datasets**         | 📚 CICIDS2017 & CIC-IoT2023                              |

---

## 🛡️ Project Focus

**Network Security • Machine Learning • Deep Learning • Cybersecurity • Intrusion Detection**
