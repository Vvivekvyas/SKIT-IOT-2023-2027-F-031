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
│   ├── data/                # dataset loading pipeline lands here (Task 2)
│   ├── templates/           # Jinja2: base.html, login.html, dashboard.html
│   └── static/               # css/js for the dashboard
├── scripts/train_models.py  # Module 2 model training entrypoint (placeholder)
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

## 6. Try the classification endpoint

The model registry needs an actual trained model to serve predictions.
Until Module 2 delivers the real one, generate a dummy model so the pipeline
is testable end-to-end:

```bash
python scripts/train_models.py
```

Then:
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"source_ip": "10.0.0.5", "protocol": "TCP", "features": {"duration": 1.2, "src_bytes": 500, "dst_bytes": 200}}'
```

Refresh http://localhost:8000/dashboard to see it appear in the table (and
in the alerts panel if it was classified as an attack).

## 7. Run tests

```bash
pytest
```

## Handoff points for teammates

- **Dataset Loading Pipeline (your other task):** write loaders under `app/data/`
  for CIC-IoT2023 / CICIDS2017, and have preprocessing output
  `feature_columns.json` in the exact format `app/ml/model_registry.py` expects.
- **Model Training (Module 2):** `scripts/train_models.py` is the contract —
  fill in real training/comparison logic, but keep saving to
  `app/ml/artifacts/best_model.pkl` + `feature_columns.json`.
- **Dashboard UI polish:** templates live in `app/templates/`, styled with
  Tailwind CDN classes — safe to restyle without touching Python code.
