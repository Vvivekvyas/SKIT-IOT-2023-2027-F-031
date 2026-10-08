# ids-api — Week 6: Dataset upload/processing endpoints

## What's new this week

All under `/api/v1/datasets`:

| Endpoint | Who | What it does |
|---|---|---|
| `POST /upload` | analyst, admin | Upload a CSV (validated, size-limited) |
| `GET /` | analyst, admin | List datasets, newest first (`limit`, `offset`) |
| `GET /{id}` | analyst, admin | Name, status, row count, file size, upload time |
| `GET /{id}/status` | analyst, admin | Processing status |
| `GET /{id}/preview` | analyst, admin | Column names + first N rows (`limit`, 1-50) |
| `POST /{id}/process` | analyst, admin | Re-run processing (409 if already running) |
| `DELETE /{id}` | admin only | Delete the dataset row and its file |

The seed script now also creates `admin@example.com` (needed to test delete).

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env   # then set a real JWT_SECRET_KEY
python -m app.db.seed  # creates tables + test users (safe to re-run)
uvicorn app.main:app --reload
```

Docs: http://localhost:8000/docs

Test logins (password `changeme123` for both):
- `analyst@example.com`
- `admin@example.com`

## Also available

- `POST /api/v1/auth/login`, `POST /api/v1/auth/refresh`
- `GET /api/v1/me`, `GET /health`

## Layout

```
app/
  core/       config, JWT + password hashing, auth dependency, rate limiter
  db/         database.py (engine/session), seed.py (setup script)
  models/     SQLAlchemy table definitions
  api/v1/     auth.py, datasets.py
  schemas/    Pydantic request/response models
```

## Not yet wired up

The `Prediction` table and schemas exist, but there is no `/predict` endpoint
yet — that lands once the ML side has a trained model to call.
