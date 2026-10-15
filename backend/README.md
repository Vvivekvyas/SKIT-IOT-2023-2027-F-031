# ids-api — Week 7: API validation + error handling

## What's new this week

**One error format for everything.** Every error looks like this:

```json
{
  "data": null,
  "meta": {"request_id": "3f2a..."},
  "error": {"code": "not_found", "message": "Dataset not found", "details": null}
}
```

| Status | `error.code` |
|---|---|
| 400 | `bad_request` |
| 401 | `unauthorized` |
| 403 | `forbidden` |
| 404 | `not_found` |
| 405 | `method_not_allowed` |
| 409 | `conflict` |
| 413 | `payload_too_large` |
| 422 | `validation_error` (lists each bad field in `details`) |
| 429 | `rate_limited` |
| 500 | `internal_error` (generic message; the real error goes only to the server log) |

**Request IDs.** Every response has an `X-Request-ID` header. The same ID is in the
error body and in the server log line, so one failing request can be traced.
A client may send its own `X-Request-ID` (letters, digits, dashes, max 64).

**Stricter input checks**
- Login: valid email, password 1-128 characters, unknown fields rejected
- Refresh token: 1-2048 characters, unknown fields rejected
- Dataset ID in the URL must be a UUID (otherwise 422)
- Upload: `.csv` extension and allowed content type required, file not empty,
  not binary, header line present, size limit. File name is cleaned (no folders,
  no control characters). A failed upload never leaves a file on disk
- Predict request (for the future `/predict`): flat feature dictionary, 1-500
  features, no nested values, unknown fields rejected

Validation errors never echo the submitted value back (so a password is not repeated).

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env   # then set a real JWT_SECRET_KEY
python -m app.db.seed
uvicorn app.main:app --reload
```

Docs: http://localhost:8000/docs

## Layout

```
app/
  core/       config, security, auth dependency, rate limiter,
              errors.py (error format), middleware.py (request IDs + logging)
  db/         database.py, seed.py
  models/     SQLAlchemy tables
  api/v1/     auth.py, datasets.py
  schemas/    Pydantic request/response models
```

## Not yet done

- Successful responses are still plain JSON (only errors use the envelope above).
  Wrapping them too needs a decision with the frontend first.
- `/predict` does not exist yet (waits for a trained model).
