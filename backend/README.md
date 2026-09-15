# ids-api

Week 1 deliverable — API scaffolding + working auth/security layer for the
hybrid FT-Transformer + VAE IDS project.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env   # then set a real JWT_SECRET_KEY
uvicorn app.main:app --reload
```

Docs: http://localhost:8000/docs

## What's implemented

- `POST /api/v1/auth/login` — real JWT issuance, argon2 password hashing,
  rate-limited (5/min)
- `POST /api/v1/auth/refresh` — refresh token rotation
- `GET /api/v1/me` — example protected route using the shared auth dependency
- `GET /health` — public health check
- Role-based access control (`viewer` / `analyst` / `admin`) via
  `app/core/deps.py::require_role`
- CORS locked to `ALLOWED_ORIGINS`, no wildcard

User lookup is a temporary in-memory dict (`app/api/v1/auth.py`) — swap for
Pranjal's DB models in Week 2.

## Layout

```
app/
  core/       config, JWT + password hashing, auth dependency, rate limiter
  api/v1/     versioned route modules (auth.py so far)
  schemas/    Pydantic request/response models
```
