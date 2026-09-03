# API Architecture — Hybrid FT-Transformer + VAE Intrusion Detection System

## 1. Purpose

This document defines the API layer that sits between the frontend dashboard,
the ML inference engine (VAE + FT-Transformer hybrid), and the backend/data
layer. It is the contract every other component builds against.

## 2. Design goals

- **Separation of concerns.** The API never embeds ML logic or business rules — it
  validates, authenticates, routes, and shapes responses. Model inference lives behind
  a dedicated model-serving interface.
- **Security by default.** Every endpoint is authenticated unless explicitly marked
  public (e.g. `/health`). No endpoint trusts client-supplied identifiers without
  authorization checks.
- **Versioned and stable.** All routes are prefixed `/api/v1/...` so the contract can
  evolve without breaking the frontend or backend.
- **Traceable.** Every request/response is logged with a correlation ID for debugging
  during model integration.

## 3. Tech stack

| Layer | Choice | Rationale |
|---|---|---|
| Framework | FastAPI (Python) | Async, auto-generated OpenAPI docs, native Pydantic validation |
| Data validation | Pydantic v2 | Type-safe request/response schemas, shared with backend DB models |
| Auth | OAuth2 password flow + JWT (access + refresh tokens) | Stateless, works cleanly with a dashboard + API client model |
| API docs | OpenAPI/Swagger (auto from FastAPI) | Zero-maintenance docs for frontend integration |
| Server | Uvicorn/Gunicorn | Standard FastAPI deployment target |

## 4. API surface

High-level route groups the system exposes:

| Group | Endpoints (indicative) |
|---|---|
| Auth | `POST /api/v1/auth/login`, `/refresh`, `/logout` |
| Dataset ingestion | `POST /api/v1/datasets/upload`, `GET /api/v1/datasets/{id}/status` |
| Prediction (secured) | `POST /api/v1/predict`, `GET /api/v1/predict/{id}` |
| Model serving/versioning | `GET /api/v1/models`, `POST /api/v1/models/{id}/activate` |
| Autoencoder service | `POST /api/v1/latent/encode` |
| Hybrid prediction | `POST /api/v1/hybrid/predict` |
| Alerts/history | `GET /api/v1/alerts`, `GET /api/v1/alerts/{id}/history` |
| Dashboard aggregates | `GET /api/v1/dashboard/summary`, `/metrics` |

## 5. Request flow

1. Frontend authenticates via `/auth/login`, receives a JWT.
2. Every subsequent request carries `Authorization: Bearer <token>`.
3. The gateway validates the token, checks scope/role, then forwards to the
   relevant service (dataset, prediction, dashboard).
4. Prediction requests route to the model-serving interface, which loads the
   active VAE + FT-Transformer pipeline and returns a classification with
   confidence and an attention-based interpretability signal.
5. Responses use a consistent envelope: `{ "data": ..., "meta": {...}, "error": null }`.
