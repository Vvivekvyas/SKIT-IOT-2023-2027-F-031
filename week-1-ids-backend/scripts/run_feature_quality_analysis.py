"""
Runs feature-quality + security/leakage analysis on the processed training
data and writes a Markdown report.

Run after scripts/run_data_pipeline.py has produced
app/data/processed/train.csv and app/ml/artifacts/feature_columns.json:

    python scripts/run_feature_quality_analysis.py

Output: app/data/processed/feature_quality_report.md
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from app.data.feature_quality import analyze  # noqa: E402
from app.data.feature_quality_report import to_markdown  # noqa: E402

TRAIN_PATH = Path("app/data/processed/train.csv")
FEATURE_COLUMNS_PATH = Path("app/ml/artifacts/feature_columns.json")
REPORT_PATH = Path("app/data/processed/feature_quality_report.md")


def main() -> None:
    if not TRAIN_PATH.exists() or not FEATURE_COLUMNS_PATH.exists():
        raise SystemExit(
            "Missing app/data/processed/train.csv or app/ml/artifacts/feature_columns.json.\n"
            "Run `python scripts/run_data_pipeline.py` first."
        )

    df = pd.read_csv(TRAIN_PATH)
    with open(FEATURE_COLUMNS_PATH) as f:
        feature_columns = json.load(f)

    report = analyze(df, feature_columns)
    markdown = to_markdown(report)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(markdown)

    print(markdown)
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()