"""
Creates the tables and adds any columns that were introduced later.

SQLAlchemy's create_all() only creates missing TABLES. It never changes a
table that already exists, so a database made in an earlier week would be
missing new columns. ensure_schema() adds them, so nobody has to delete
their database after an update. Safe to run any number of times.
"""
from sqlalchemy import inspect, text

from app.db.database import Base, engine
from app.models.dataset import Dataset  # noqa: F401 - imported so create_all sees it
from app.models.prediction import Prediction  # noqa: F401
from app.models.user import User  # noqa: F401

# table -> {column: SQL type} for columns added after the table was first created
LATER_COLUMNS = {
    "datasets": {
        "dataset_type": "VARCHAR",
        "is_valid": "BOOLEAN",
        "validation_report": "TEXT",
        "message": "VARCHAR",
    },
    "predictions": {
        "anomaly_score": "FLOAT",
    },
}


def ensure_schema() -> list[str]:
    """Create missing tables, add missing columns. Returns what was added."""
    Base.metadata.create_all(bind=engine)

    added = []
    inspector = inspect(engine)
    for table, columns in LATER_COLUMNS.items():
        existing = {c["name"] for c in inspector.get_columns(table)}
        for name, sql_type in columns.items():
            if name not in existing:
                with engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}"))
                added.append(f"{table}.{name}")
    return added
