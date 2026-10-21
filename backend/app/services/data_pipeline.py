"""
Connects the API to the data pipeline written by the ML team (the repo's
top-level `src/data` package).

What it does:
  1. Finds the repo root and makes `src.data` importable.
  2. Works out which dataset a CSV is (CICIDS2017 or CIC-IoT2023) from its
     column names, unless the uploader says which one.
  3. Runs the ML team's DatasetValidator and returns its report.

If `src/data` or pandas is not available, PipelineUnavailable is raised and
the dataset endpoints keep working without validation.
"""
import importlib
import logging
import sys
from pathlib import Path
from typing import Any

from app.core.config import settings

logger = logging.getLogger("ids.api")

# Keep in sync with DATASET_CONFIGS in src/data/schemas.py
KNOWN_DATASET_TYPES = ("CICIDS2017", "CIC-IoT2023")

# Share of a dataset's required columns that must be present to recognise it.
MIN_COLUMN_MATCH = 0.5


class PipelineUnavailable(Exception):
    """The ML team's data pipeline code can't be used on this machine."""


_cache: tuple[Any, dict, Any] | None = None


def _repo_root() -> Path:
    if settings.PIPELINE_ROOT:
        return Path(settings.PIPELINE_ROOT).resolve()
    # backend/app/services/data_pipeline.py -> repo root is 3 levels above `services`
    return Path(__file__).resolve().parents[3]


def _load() -> tuple[Any, dict, Any]:
    """Import pandas and the ML team's validator once, on first use."""
    global _cache
    if _cache is not None:
        return _cache

    root = _repo_root()
    if not (root / "src" / "data" / "validator.py").exists():
        raise PipelineUnavailable(f"data pipeline code not found in {root / 'src' / 'data'}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        pd = importlib.import_module("pandas")
        schemas = importlib.import_module("src.data.schemas")
        validator = importlib.import_module("src.data.validator")
    except ImportError as exc:
        raise PipelineUnavailable(f"a package the data pipeline needs is missing ({exc.name})") from exc

    _cache = (pd, schemas.DATASET_CONFIGS, validator.DatasetValidator)
    return _cache


def detect_dataset_type(columns: list[str], configs: dict) -> str | None:
    """Pick the dataset whose required columns best match the CSV header."""
    present = {c.strip() for c in columns}
    best_name, best_ratio = None, 0.0
    for name, config in configs.items():
        required = config["mandatory_features"]
        ratio = sum(1 for col in required if col.strip() in present) / max(len(required), 1)
        if ratio > best_ratio:
            best_name, best_ratio = name, ratio
    return best_name if best_ratio >= MIN_COLUMN_MATCH else None


def run_validation(csv_path: str, dataset_type: str | None, total_rows: int) -> dict:
    """
    Validate a CSV with the ML team's validator.

    Returns {"dataset_type", "valid", "report", "message"}:
      - valid is True/False, report is the validator's report as a dict
      - for a file that matches no known dataset: valid False, report None
    Raises PipelineUnavailable if the pipeline can't be used here.
    Any other exception (e.g. unreadable CSV) is for the caller to handle.
    """
    pd, configs, validator_cls = _load()

    header = pd.read_csv(csv_path, nrows=0, encoding="utf-8", encoding_errors="replace")
    resolved = dataset_type or detect_dataset_type(list(header.columns), configs)
    if resolved is None:
        known = " or ".join(configs.keys())
        return {
            "dataset_type": None,
            "valid": False,
            "report": None,
            "message": f"Columns do not match a known dataset ({known}).",
        }

    sample_rows = settings.VALIDATION_SAMPLE_ROWS
    df = pd.read_csv(
        csv_path,
        nrows=sample_rows,
        low_memory=False,
        encoding="utf-8",
        encoding_errors="replace",
    )
    report = validator_cls().validate(df, resolved).to_dict()
    report["rows_checked"] = len(df)
    report["sampled"] = total_rows > len(df)

    message = None
    if report["sampled"]:
        message = f"Validated the first {len(df)} of {total_rows} rows."
    if not report["is_valid"]:
        first_error = report["errors"][0] if report["errors"] else "validation failed"
        message = f"{first_error} {message or ''}".strip()

    return {"dataset_type": resolved, "valid": report["is_valid"], "report": report, "message": message}
