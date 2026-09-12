# Security Requirements — Hybrid FT-Transformer + VAE Intrusion Detection System

## 1. Scope

This is a system that itself detects attacks, so its own API is a plausible target.
These requirements cover the API layer, the model-serving path, and the data
ingestion path. They're written before endpoints exist, so security is designed
in rather than patched on later.

## 2. Threat model

| Threat | Relevant surface | Mitigation direction |
|---|---|---|
| Unauthorized access to predictions/alerts | All `/api/v1/*` routes | JWT auth + role-based access control |
| Credential stuffing / brute force on login | `/auth/login` | Rate limiting, account lockout, password hashing (bcrypt/argon2) |
| Injection via uploaded dataset/CSV fields | `/datasets/upload` | Strict Pydantic schemas, size limits, content-type checks, sandboxed parsing |
| Model extraction / excessive query probing | `/predict`, `/hybrid/predict` | Per-user/API-key rate limiting, response throttling on repeated near-boundary queries |
| Adversarial input crafted to evade the IDS classifier | `/predict` | Input validation ranges, logging of anomalous request patterns |
| Data exfiltration of network traffic datasets | Dataset storage/API | Encryption at rest, access-scoped download endpoints, audit logging |
| Token theft / replay | JWT-based auth | Short-lived access tokens, refresh token rotation, HTTPS-only cookies where applicable |
| Man-in-the-middle | All traffic | TLS 1.2+ enforced, HSTS |

## 3. Authentication & authorization requirements

- All non-public endpoints require a valid JWT.
- Access tokens expire in ≤15 minutes; refresh tokens are rotated on use and revocable.
- Passwords hashed with bcrypt or argon2 — never stored or logged in plaintext.
- Role-based access control minimum viable set: `admin`, `analyst`, `viewer`.
  - `viewer`: read-only dashboard/alerts access.
  - `analyst`: can trigger predictions, view full alert history.
  - `admin`: model version management, user management.

## 4. API-level requirements (OWASP API Security Top 10 aligned)

1. **Broken object level authorization** — every resource fetch (`/alerts/{id}`,
   `/datasets/{id}`) must verify the requester owns or is scoped to that resource.
2. **Broken authentication** — enforce token expiry, no fallback to unauthenticated
   access paths.
3. **Excessive data exposure** — response schemas explicitly whitelist fields; never
   return raw ORM/DB objects.
4. **Lack of resource & rate limiting** — apply per-IP and per-token rate limits,
   especially on `/predict` and `/auth/login`.
5. **Mass assignment** — Pydantic models define exactly which fields are writable.
6. **Security misconfiguration** — no debug mode, no verbose stack traces, in
   production; CORS locked to known frontend origins.
7. **Injection** — parameterized queries only (via ORM), strict input validation on
   all uploaded data.
8. **Improper inventory management** — all endpoints documented in OpenAPI, no
   undocumented/shadow routes.
9. **Insufficient logging & monitoring** — every auth event, prediction request, and
   admin action logged with timestamp, actor, and correlation ID (no sensitive
   payload data in logs).

## 5. Data protection requirements

- Datasets (CICIDS2017/CIC-IoT2023-derived) encrypted at rest.
- TLS enforced in transit for all API traffic — no plaintext HTTP in any environment.
- Secrets (DB credentials, JWT signing keys) stored in environment variables /
  secret manager, never committed to the repo.
- PII is not expected in these network-traffic datasets, but any user-account data
  (analyst emails, etc.) follows minimal-collection principles.

## 6. Model-serving security considerations

- Model artifacts (VAE, FT-Transformer weights) are not directly downloadable via
  the API — only inference results are exposed.
- Prediction endpoints log input feature summaries (not raw payloads) to support
  anomaly detection on the API itself without duplicating full traffic data.
- Confidence scores and attention-weight interpretability output are
  access-controlled the same as predictions — they can leak information about
  decision boundaries if exposed too broadly.

## 7. Compliance/process notes

- No compliance certification (SOC2/ISO) is targeted — this is an academic/research
  project — but the above practices mirror OWASP API Security Top 10 as the baseline
  standard referenced in the report.
