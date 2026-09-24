# ML-Based Intrusion Detection System — Backend

FastAPI backend for the final-year project (SKIT, Dept. of CSE).
Serves REST endpoints for traffic classification, alert history, admin
login, and the server-rendered dashboard.

## Architecture

```
ids-backend/
├── app/
│   ├── main.py              # FastAPI app, mounts routers + static files
│   ├── core/
│   │   ├── config.py        # env-driven settings
│   │   ├── database.py      # SQLAlchemy engine/session
│   │   └── security.py      # password hashing + JWT
│   ├── models/traffic.py    # ORM: User, TrafficRecord, Alert
│   ├── schemas/traffic.py   # Pydantic request/response contracts
│   ├── api/
│   │   ├── deps.py          # shared dependencies (auth, DB session)
│   │   └── v1/
│   │       ├── router.py    # wires all endpoint modules together
│   │       └── endpoints/   # auth.py, predict.py, traffic.py, dashboard.py
│   ├── ml/
│   │   ├── model_registry.py  # loads whichever model is marked "active"
│   │   ├── inference.py       # feature alignment + predict()
│   │   └── artifacts/         # trained model files go here (gitignored)
│   ├── data/
│   │   ├── loaders.py             # loads/concats CICIDS2017 + CIC-IoT2023 CSVs
│   │   ├── label_mapping.py       # raw attack names -> Normal/DoS/Probe/R2L/U2R
│   │   ├── preprocessing.py       # clean, encode, select features, scale
│   │   ├── pipeline.py            # orchestrates the above + zero-day holdout split
│   │   ├── feature_quality.py     # feature quality + leakage/security checks
│   │   ├── feature_quality_report.py  # renders the analysis as Markdown
│   │   ├── raw/                   # put downloaded dataset CSVs here (gitignored)
│   │   └── processed/             # pipeline output: train/test/zero_day CSVs + report
│   ├── templates/           # Jinja2: base.html, login.html, dashboard.html
│   └── static/               # css/js for the dashboard
├── scripts/
│   ├── run_data_pipeline.py            # Dataset Loading Pipeline CLI
│   ├── run_feature_quality_analysis.py # Feature-quality & security analysis CLI
│   └── train_models.py                 # Module 2 model training entrypoint (placeholder)
├── tests/
├── requirements.txt
├── .env.example
├── docker-compose.yml       # local Postgres
└── README.md
```

**Design decisions and why:**
- **FastAPI** — matches the proposal's tech stack, async-friendly, auto-generates
  OpenAPI docs at `/docs` for the rest of the team to test endpoints without a UI.
- **SQLAlchemy + PostgreSQL** — `TrafficRecord` stores every classified flow,
  `Alert` stores only the malicious ones (drives the dashboard's alert panel).
- **Model registry pattern** — the API never imports a specific model class.
  It loads whatever `.pkl` is at `ACTIVE_MODEL_PATH`. This means whoever wins
  the model comparison in Module 2 just drops a file in and updates `.env` —
  zero backend code changes.
- **Jinja2 + Tailwind (CDN)** — matches "HTML, Tailwind CSS, JS, Jinja2 Templates"
  from the Technology Stack slide, server-rendered so it works without a
  separate frontend build step.

## 1. Repo setup (do this once)

```bash
# create the repo (or clone if a teammate already created it on GitHub)
git init ids-backend
cd ids-backend
git branch -M main
```

If a teammate already created the GitHub repo:
```bash
git clone <repo-url> ids-backend
cd ids-backend
```

Copy all the generated files/folders into this directory, then:
```bash
git add .
git commit -m "Initial backend architecture and repo scaffold"
git remote add origin <repo-url>   # skip if you cloned
git push -u origin main
```

Agree on a branching convention with your team, e.g.:
- `main` — always working
- `feature/dataset-pipeline`, `feature/model-training`, `feature/dashboard-ui`
- open PRs into `main`

## 2. Local environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

```bash
cp .env.example .env
# edit .env: set a real SECRET_KEY, adjust DATABASE_URL if needed
```

## 3. Database (PostgreSQL via Docker)

```bash
docker compose up -d
```

This starts Postgres on `localhost:5432` with the credentials already
matching `.env.example`. Tables are auto-created on first API startup
(`Base.metadata.create_all` in `main.py`) — fine for development; swap to
Alembic migrations before anything resembling production.

## 4. Create an admin user (one-time, for the login page)

Run this once with the venv active:
```bash
python -c "
from app.core.database import SessionLocal, Base, engine
from app.core.security import hash_password
from app.models.traffic import User

Base.metadata.create_all(bind=engine)
db = SessionLocal()
db.add(User(username='admin', hashed_password=hash_password('admin123'), is_admin=True))
db.commit()
"
```

## 5. Run the server

```bash
uvicorn app.main:app --reload
```

- API docs: http://localhost:8000/docs
- Dashboard: http://localhost:8000/dashboard
- Login page: http://localhost:8000/login (admin / admin123 from step 4)
- Health check: http://localhost:8000/health

## 6. Run the dataset loading pipeline

Download the datasets and place the CSVs here (not committed to git):
```
app/data/raw/cicids2017/*.csv    <- https://www.unb.ca/cic/datasets/ids-2017.html
app/data/raw/ciciot2023/*.csv    <- https://www.unb.ca/cic/datasets/iotdataset-2023.html
```

Then run:
```bash
python scripts/run_data_pipeline.py --dataset both --k-features 30 --zero-day-categories U2R
```

This loads both datasets, maps their raw attack labels onto the 5 output
categories (Normal/DoS/Probe/R2L/U2R — see `app/data/label_mapping.py`),
cleans (drops inf/NaN/duplicate rows), holds out the `U2R` category entirely
for zero-day evaluation, encodes categoricals, picks the top-k features
(ANOVA F-value), scales, and writes:
- `app/data/processed/train.csv`, `test.csv`, `zero_day_test.csv`
- `app/ml/artifacts/feature_columns.json` — read automatically by `model_registry.py`
- `app/ml/artifacts/scaler.pkl`, `label_encoders.pkl`

Run `--dataset cicids2017` or `--dataset ciciot2023` alone if you only have
one dataset downloaded yet.

## 7. Try the classification endpoint

The model registry needs an actual trained model to serve predictions.
Once step 6 has produced `train.csv`, run:

```bash
python scripts/train_models.py
```

(This trains a baseline Random Forest on the real pipeline output — swap in
the real model comparison from Module 2 later; the file contract stays the same.
If you haven't run step 6 yet, this still works using dummy data so the API
stays testable end-to-end.)

Then:
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"source_ip": "10.0.0.5", "protocol": "TCP", "features": {"duration": 1.2, "src_bytes": 500, "dst_bytes": 200}}'
```

Refresh http://localhost:8000/dashboard to see it appear in the table (and
in the alerts panel if it was classified as an attack).

## 8. Feature quality & security analysis (Week 4)

Once the pipeline has produced `app/data/processed/train.csv`, run:

```bash
python scripts/run_feature_quality_analysis.py
```

This checks the exact feature set the model will train on for:
- **Near-zero-variance features** — columns carrying almost no signal
- **Redundant feature pairs** — correlation ≥ 0.95, candidates to drop one of each pair
- **Feature importance** — Random Forest importance + mutual information ranking
- **Outlier proportion per feature** — IQR-rule based
- **Security/leakage risk:**
  - Identity-style columns (IP/port/flow-ID/timestamp/MAC) that risk the model
    memorizing hosts from the lab capture instead of learning attack *behavior*
  - Any single feature that separates classes almost perfectly on its own
    (AUC ≥ 0.98) — a red flag for either a genuinely strong signal or a leak

Output: `app/data/processed/feature_quality_report.md` — share this with your
supervisor/team alongside the preprocessing report from Module 2.

## 9. Run tests

```bash
pytest
```

## Handoff points for teammates

- **Model Training (Module 2):** `scripts/train_models.py` is the contract —
  fill in the real RF/SVM/Decision Tree/XGBoost/FT-Transformer+Autoencoder
  comparison, but keep saving the winner to `app/ml/artifacts/best_model.pkl`.
  It already reads `app/data/processed/train.csv` — no changes needed there.
- **Dashboard UI polish:** templates live in `app/templates/`, styled with
  Tailwind CDN classes — safe to restyle without touching Python code.
- **Label mapping review:** `app/data/label_mapping.py` maps CIC-IoT2023's 33
  attack types onto Normal/DoS/Probe/R2L/U2R as a first pass — some (like the
  Mirai botnet family) don't map cleanly onto that classic taxonomy. Worth a
  team review before final training.