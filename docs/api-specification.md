# API Specification — Hybrid IDS

Detailed contract for each endpoint: method, path, request body, response body,
status codes, and auth requirement. Builds directly on `api-architecture.md`.

## Conventions

- Base path: `/api/v1`
- All requests/responses: `application/json`
- Auth: `Authorization: Bearer <access_token>` header, unless marked **Public**
- Standard response envelope:
```json
{
  "data": { },
  "meta": { },
  "error": null
}
```
- Standard error shape:
```json
{
  "data": null,
  "meta": { },
  "error": { "code": "string", "message": "string" }
}
```

## Auth

### POST /auth/login — Public

Request:
```json
{ "username": "analyst@example.com", "password": "string" }
```
Response `200`:
```json
{ "access_token": "string", "refresh_token": "string", "token_type": "bearer" }
```
Errors: `401` invalid credentials, `429` rate limit exceeded.

### POST /auth/refresh — Public

Request:
```json
{ "refresh_token": "string" }
```
Response `200`: same shape as `/auth/login`.
Errors: `401` invalid or expired refresh token.

### POST /auth/logout — Auth required

Response `204`. Invalidates the current refresh token server-side.

## Datasets

### POST /datasets/upload — Auth required (analyst, admin)

Request: `multipart/form-data`, field `file` (CSV).
Response `202`:
```json
{ "dataset_id": "string", "status": "processing" }
```
Errors: `400` invalid file type/size, `413` file too large.

### GET /datasets/{id}/status — Auth required (analyst, admin)

Response `200`:
```json
{ "dataset_id": "string", "status": "processing|processed|failed", "rows": 0 }
```
Errors: `404` not found, `403` not authorized for this dataset.

## Prediction

### POST /predict — Auth required (analyst, admin)

Request:
```json
{ "features": { } }
```
Response `200`:
```json
{
  "prediction_id": "string",
  "classification": "Normal|DoS|Probe|R2L|U2R",
  "confidence": 0.0
}
```
Errors: `400` malformed feature payload, `429` rate limit exceeded.

### GET /predict/{id} — Auth required (analyst, admin)

Response `200`: same shape as the `POST /predict` response, retrieved by ID.
Errors: `404` not found.

## Models

### GET /models — Auth required (admin)

Response `200`:
```json
{ "models": [ { "id": "string", "version": "string", "active": true } ] }
```

### POST /models/{id}/activate — Auth required (admin)

Response `200`:
```json
{ "id": "string", "active": true }
```
Errors: `404` model not found.

## Alerts

### GET /alerts — Auth required (viewer, analyst, admin)

Query params: `limit` (default 20), `cursor` (pagination token).
Response `200`:
```json
{ "alerts": [ { "id": "string", "classification": "string", "confidence": 0.0, "timestamp": "ISO8601" } ], "next_cursor": "string|null" }
```

### GET /alerts/{id} — Auth required (viewer, analyst, admin)

Response `200`: full alert detail including source feature summary.
Errors: `404` not found.

## Dashboard

### GET /dashboard/summary — Auth required (viewer, analyst, admin)

Response `200`:
```json
{
  "total_scanned": 0,
  "active_alerts": 0,
  "active_model_version": "string",
  "class_distribution": { "Normal": 0, "DoS": 0, "Probe": 0 }
}
```

## Health

### GET /health — Public

Response `200`:
```json
{ "status": "ok" }
```
