# ids-api — Week 5: Data ingestion API

## What's new this week

- `POST /api/v1/datasets/upload` — real, working endpoint: accepts a CSV,
  validates file type and size, saves it to disk, creates a `Dataset` row,
  and processes it in the background (counts rows, flips status to
  `processed` or `failed`)
- `GET /api/v1/datasets/{id}/status` — poll for the processing result
- Both require `analyst` or `admin` role

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env   # then set a real JWT_SECRET_KEY
python -m app.db.seed  # creates tables + test user (run once)
uvicorn app.main:app --reload
```

Docs: http://localhost:8000/docs

Test login: `analyst@example.com` / `changeme123`

## What's implemented

- `POST /api/v1/auth/login` — queries the real `users` table
- `POST /api/v1/auth/refresh` — token rotation
- `GET /api/v1/me` — protected test route
- `GET /health` — public health check
- `POST /api/v1/datasets/upload`, `GET /api/v1/datasets/{id}/status` — dataset ingestion
- `app/models/` — SQLAlchemy table definitions (User, Dataset, Prediction)
- `app/schemas/` — Pydantic request/response models for auth, datasets, predictions/alerts

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

The `Prediction` table and schema exist, but there's no `/predict` endpoint
yet — that lands once the ML side (VAE + FT-Transformer) has a trained
model to actually call, per the project's later weeks.
