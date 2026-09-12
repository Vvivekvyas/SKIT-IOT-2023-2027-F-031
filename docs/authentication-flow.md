# Authentication Flow — Hybrid IDS

Concrete flow for how a client authenticates and stays authenticated,
implementing the auth requirements from `security-requirements.md`.

## 1. Token model

Two tokens are issued together on login:

| Token | Lifetime | Purpose |
|---|---|---|
| Access token | 15 minutes | Sent with every API request to prove identity |
| Refresh token | 7 days | Used only to get a new access token, never sent to feature endpoints |

Both are JWTs signed with a server-side secret, containing `sub` (username),
`role`, `type` (`access` or `refresh`), `iat`, and `exp`.

## 2. Login sequence

1. Client sends `POST /auth/login` with username + password.
2. Server verifies the password against the stored argon2 hash.
3. If valid, server issues a new access token + refresh token pair and returns both.
4. If invalid, server returns `401` with a generic "incorrect username or password"
   message — it never reveals whether the username exists, to prevent user
   enumeration.
5. Login attempts are rate-limited (5/minute per IP) to blunt brute-force attempts.

## 3. Using the access token

1. Client stores the access token in memory (not localStorage, to reduce
   XSS token-theft risk).
2. Every request to a protected endpoint includes:
   `Authorization: Bearer <access_token>`
3. Server middleware decodes the token, checks `type == "access"`, checks
   expiry, and checks the role against the endpoint's required role.
4. If the token is missing, expired, or invalid → `401`.
5. If the token is valid but the role is insufficient → `403`.

## 4. Refreshing an expired access token

1. When a request comes back `401` due to an expired access token, the client
   calls `POST /auth/refresh` with the stored refresh token.
2. Server verifies the refresh token (`type == "refresh"`, not expired,
   not revoked).
3. Server issues a **new** access token and a **new** refresh token
   (rotation — the old refresh token is invalidated so it can't be reused).
4. Client replaces both stored tokens and retries the original request.
5. If the refresh token itself is invalid/expired → client is logged out and
   must re-authenticate via `/auth/login`.

## 5. Logout

1. Client calls `POST /auth/logout` with the current access token.
2. Server invalidates the associated refresh token server-side (added to a
   revocation list / removed from the active-sessions store).
3. Client discards both tokens locally.

## 6. Role enforcement

Three roles, checked on every protected route:

| Role | Can access |
|---|---|
| `viewer` | Dashboard, alerts (read-only) |
| `analyst` | Everything `viewer` can, plus triggering predictions, uploading datasets |
| `admin` | Everything `analyst` can, plus model activation, user management |

Role checks happen at the API layer (not just hidden in the frontend), so a
viewer calling `/predict` directly gets `403` regardless of what the UI shows.

## 7. Sequence summary

```
Client                         Server
  |--- POST /auth/login ------->|
  |<-- access + refresh --------|
  |                              |
  |--- GET /alerts (Bearer) --->|
  |<-- 200 alerts ---------------|
  |                              |
  |    ... 15 min later ...      |
  |--- GET /alerts (Bearer) --->|
  |<-- 401 expired --------------|
  |--- POST /auth/refresh ----->|
  |<-- new access + refresh ----|
  |--- GET /alerts (Bearer) --->|
  |<-- 200 alerts ---------------|
```
