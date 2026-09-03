# UI Wireframe (Low-Fidelity) — Hybrid IDS Dashboard

## 1. Purpose

A low-fidelity wireframe of the core screens, done so the API contract is shaped
around what the UI actually needs to render. Full visual design and component
build-out happens later. This is intentionally rough — layout and data needs,
not polish.

## 2. Screen 1 — Login

```
+--------------------------------------------------+
|                   IDS Console                     |
|                                                    |
|   [ Email/username        ]                       |
|   [ Password               ]                       |
|                                                    |
|              ( Log in )                            |
|                                                    |
+--------------------------------------------------+
```
Data need: `POST /auth/login` → JWT access + refresh token.

## 3. Screen 2 — Dashboard home

```
+--------------------------------------------------------------+
| IDS Console        Nav: [Dashboard][Alerts][Datasets][Admin]  |
+--------------------------------------------------------------+
| Summary cards:                                                |
|  [ Total traffic scanned ] [ Active alerts ] [ Model version ]|
+--------------------------------------------------------------+
| Attack-class distribution (chart)  |  Recent alerts (list)    |
|                                     |  - Alert #123  DoS  9:02 |
|                                     |  - Alert #122  Probe 8:47|
+--------------------------------------------------------------+
```
Data need: `GET /dashboard/summary`, `GET /alerts?limit=10`.

## 4. Screen 3 — Alert detail / attack history

```
+--------------------------------------------------------------+
| < Back to alerts        Alert #123                            |
+--------------------------------------------------------------+
| Classification: DoS      Confidence: 0.94                     |
| Timestamp: 2026-09-03 09:02:11                                |
| Source features (summary table)                               |
| Attention/interpretability signal                              |
+--------------------------------------------------------------+
```
Data need: `GET /alerts/{id}`, later `GET /alerts/{id}/interpretability`.

## 5. Screen 4 — Dataset management (analyst/admin only)

```
+--------------------------------------------------------------+
| Datasets                                        [ Upload ]    |
+--------------------------------------------------------------+
| Name              Status        Rows        Uploaded          |
| CICIDS2017-batch1 Processed     2.3M         2026-09-05        |
| CIC-IoT2023-b2    Processing    -            2026-09-06        |
+--------------------------------------------------------------+
```
Data need: `POST /datasets/upload`, `GET /datasets/{id}/status`.

## 6. Navigation & role visibility

| Screen | viewer | analyst | admin |
|---|---|---|---|
| Dashboard | ✓ | ✓ | ✓ |
| Alerts | ✓ | ✓ | ✓ |
| Datasets | – | ✓ | ✓ |
| Admin (model/user mgmt) | – | – | ✓ |

## 7. What this drives in the API design

- Confirms `/dashboard/summary` needs aggregate counts + a class-distribution
  breakdown in one call (avoid N+1 requests from the frontend).
- Confirms alert list needs pagination (`limit`/`cursor`) from day one.
- Confirms role checks belong at the API layer, not just hidden in the UI.
