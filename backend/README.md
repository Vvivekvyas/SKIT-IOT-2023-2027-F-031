# ids-api — Week 4: Database schema + Pydantic models

## What's new this week

- Real SQLite database (SQLAlchemy) replacing Week 3's fake in-memory user dict
- Tables: `User`, `Dataset`, `Prediction`
- Pydantic schemas for dataset and prediction/alert endpoints
- A seed script to set up the DB and a test login

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

- `POST /api/v1/auth/login` — now queries the real `users` table
- `POST /api/v1/auth/refresh` — token rotation, unchanged from Week 3
- `GET /api/v1/me` — protected test route
- `GET /health` — public health check
- `app/models/` — SQLAlchemy table definitions (User, Dataset, Prediction)
- `app/schemas/` — Pydantic request/response models for auth, datasets, predictions/alerts

## Layout

```
app/
  core/       config, JWT + password hashing, auth dependency, rate limiter
  db/         database.py (engine/session), seed.py (setup script)
  models/     SQLAlchemy table definitions
  api/v1/     versioned route modules
  schemas/    Pydantic request/response models
```

## Not yet wired up

The `Dataset` and `Prediction` tables and schemas exist, but there are no
endpoints reading/writing them yet — those land when the dataset-upload and
prediction routes get built (later weeks, once the ML side has models to
call). This week's job was just the schema + models.
