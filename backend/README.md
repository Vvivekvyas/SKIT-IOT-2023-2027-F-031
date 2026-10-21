# ids-api — Week 8: Backend integration for dataset endpoints

## What's new this week

The dataset endpoints now use the ML team's real data code (`src/data` in the repo
root) instead of only counting rows.

After an upload, the backend runs in the background:
1. counts the rows,
2. works out which dataset it is (CICIDS2017 or CIC-IoT2023) from the column names,
   or uses the `dataset_type` you chose at upload,
3. runs the ML team's `DatasetValidator` and saves the report.

Only the first 100,000 rows are checked (setting `VALIDATION_SAMPLE_ROWS`), so a
huge CSV cannot use all the memory. The message says when a file was sampled.

### Dataset status values

| Status | Meaning |
|---|---|
| `processing` | still running |
| `processed` | finished. `is_valid` is `true`, or `null` if validation was skipped |
| `invalid` | finished, but the file failed validation or matches no known dataset |
| `failed` | the file could not be read as a CSV |

### Endpoint changes

- `POST /datasets/upload` — new optional form field `dataset_type`
  (`CICIDS2017` or `CIC-IoT2023`). Leave it empty to auto-detect.
- `GET /datasets/{id}/validation` — **new.** The full validation report
  (missing columns, leakage columns, NaN/infinite counts, duplicates, bad labels,
  negative values). `409` while still running, `404` if there is no report
  (the message says why).
- `GET /datasets`, `GET /datasets/{id}`, `GET /datasets/{id}/status` now also return
  `dataset_type`, `is_valid` and `message`.
- `POST /datasets/{id}/process` clears the old result and validates again.

### If the ML code is not available

If `src/data` is missing or `pandas` is not installed, uploads still work. The
dataset ends as `processed` with `is_valid: null` and a message saying validation
was skipped. Nothing crashes.

## Database

`python -m app.db.seed` now also adds the new columns to an existing database, so
you do **not** need to delete `ids_dev.db`. The server does the same check on start-up.

## Run it

```bash
pip install -r requirements.txt   # adds pandas, numpy, joblib
python -m app.db.seed
uvicorn app.main:app --reload
```

Run it from inside `backend/`. The backend finds `src/data` by itself (the repo root
is one folder above `backend`). To point somewhere else, set `PIPELINE_ROOT` in `.env`.

Docs: http://localhost:8000/docs

## Layout

```
app/
  core/       config, security, auth dependency, rate limiter, errors, middleware
  db/         database.py, schema.py (tables + new columns), seed.py
  models/     SQLAlchemy tables
  services/   data_pipeline.py (the link to src/data)
  api/v1/     auth.py, datasets.py
  schemas/    Pydantic request/response models
```

## Not yet done

- `/predict` does not exist yet (waits for a trained model).
- The validator checks a dataset but the API does not run the full cleaning/scaling
  pipeline (`DataPipeline`) yet. That is a training-time step on the ML side.
