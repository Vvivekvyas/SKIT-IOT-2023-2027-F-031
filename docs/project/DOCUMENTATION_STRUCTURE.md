# Documentation Structure

This document outlines the organization, scope, and maintenance standards for all documentation within the **Hybrid ML Based Intrusion Detection System (IDS)** project repository.

---

## 📁 Repository Documentation Layout

The project maintains modular documentation under the `docs/` directory to facilitate collaborative engineering, reproducible research, and maintainable software architecture.

```text
docs/
├── architecture/
│   └── SYSTEM_ARCHITECTURE.md
├── dataset/
│   └── DATASET_DOCUMENTATION.md
├── project/
│   └── DOCUMENTATION_STRUCTURE.md
└── testing/
    ├── TEST_STRATEGY.md
    └── TEST_CASES.md
```

---

## 📂 Directory Descriptions

### 1. `docs/architecture/`
- **Purpose**: Houses the high-level system architecture, component diagrams, deep learning pipeline design, and dataflow specifications.
- **Key Documents**:
  - `SYSTEM_ARCHITECTURE.md`: Complete blueprint detailing the integration between the Preprocessing Engine, Variational Autoencoder (VAE), Latent Space representation, FT-Transformer tabular classifier, and the Backend API.
- **Audience**: ML engineers, backend developers, system architects.

### 2. `docs/dataset/`
- **Purpose**: Details the datasets used for model training, validation, and benchmarking.
- **Key Documents**:
  - `DATASET_DOCUMENTATION.md`: Technical breakdowns of **CICIDS2017** and **CIC-IoT2023**, including feature summaries, attack class taxonomies, data cleaning rules, normalization strategies, and strict data leakage prevention mechanisms.
- **Audience**: Data engineers, ML researchers.

### 3. `docs/project/`
- **Purpose**: Contains project governance, planning, team guidelines, and documentation policies.
- **Key Documents**:
  - `DOCUMENTATION_STRUCTURE.md`: This file, serving as the index and formatting standard for all project documentation.
- **Audience**: All team members, academic evaluators, and project mentors.

### 4. `docs/testing/`
- **Purpose**: Covers all verification, validation, and quality assurance strategies for the software and machine learning models.
- **Key Documents**:
  - `TEST_STRATEGY.md`: Outlines the multi-tier testing methodology (Unit, Integration, API, Model Evaluation, Security, and Performance).
  - `TEST_CASES.md`: Detailed test case registry providing step-by-step inputs, preconditions, and expected outcomes across all system components.
- **Audience**: QA engineers, test developers, reviewers.

---

## 🚀 Future Documentation Folders

As the project advances through subsequent development milestones, the following documentation directories will be added:

- **`docs/api/`**: API endpoint specifications, OpenAPI / Swagger schemas, authentication protocols, and request/response payloads for the FastAPI backend.
- **`docs/models/`**: Deep learning model training logs, hyperparameter configurations, loss curves, confusion matrices, and ablation study reports.
- **`docs/security/`**: Threat modeling, vulnerability assessments, input sanitization protocols, and security audit reports.
- **`docs/deployment/`**: Containerization instructions (Docker), CI/CD pipelines (GitHub Actions), environment configurations, and edge deployment guides.

---

## ✍️ Documentation Guidelines & Standards

1. **Format**: All documentation must be written in standard GitHub-Flavored Markdown (`.md`).
2. **Clarity**: Keep sections well-structured using descriptive headings, bullet points, code blocks, and markdown tables.
3. **Integrity**: Never state hypothetical or unachieved experimental results as completed facts. Clearly label draft or planned metrics.
4. **Maintenance**: When modifying code, updating pipelines, or altering schemas, the corresponding documentation in `docs/` must be updated within the same Pull Request.
