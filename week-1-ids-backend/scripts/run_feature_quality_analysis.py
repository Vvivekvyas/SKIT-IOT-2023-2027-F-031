"""
Runs feature-quality + security/leakage analysis on the processed training
data and writes a comprehensive Markdown report.

Usage:
    python scripts/run_feature_quality_analysis.py \
        --train-path app/data/processed/train.csv \
        --features-path app/ml/artifacts/feature_columns.json \
        --report-path app/data/processed/feature_quality_report.md
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to system path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from app.data.feature_quality import analyze  # noqa: E402
from app.data.feature_quality_report import to_markdown  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

DEFAULT_TRAIN_PATH = Path("app/data/processed/train.csv")
DEFAULT_FEATURE_COLUMNS_PATH = Path("app/ml/artifacts/feature_columns.json")
DEFAULT_REPORT_PATH = Path("app/data/processed/feature_quality_report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run feature-quality and security/leakage analysis on training data."
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help="Path to the processed training CSV file."
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=DEFAULT_FEATURE_COLUMNS_PATH,
        help="Path to the feature columns JSON file."
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path where the output Markdown report will be saved."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.train_path.exists() or not args.features_path.exists():
        logger.error(
            f"Missing required files:\n- {args.train_path}\n- {args.features_path}\n"
            "Please run `python scripts/run_data_pipeline.py` first."
        )
        raise SystemExit(1)

    logger.info(f"Loading training data from {args.train_path}...")
    df = pd.read_csv(args.train_path)

    logger.info(f"Loading feature columns from {args.features_path}...")
    with open(args.features_path, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    logger.info("Running feature quality and security/leakage analysis...")
    report = analyze(df, feature_columns)
    
    logger.info("Generating Markdown report...")
    markdown = to_markdown(report)

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(markdown, encoding="utf-8")

    logger.info(f"Feature quality analysis report successfully saved to {args.report_path}")
    print("\n" + markdown)


if __name__ == "__main__":
    main()