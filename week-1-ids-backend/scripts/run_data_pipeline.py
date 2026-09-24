"""
CLI to run the full dataset loading pipeline.

Examples:
    # both datasets, default 30 features, hold out U2R for zero-day testing
    python scripts/run_data_pipeline.py --dataset both --zero-day-categories U2R

    # just CICIDS2017, 20 features, no zero-day holdout
    python scripts/run_data_pipeline.py --dataset cicids2017 --k-features 20
"""
import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/run_data_pipeline.py` from the repo root
# without needing to set PYTHONPATH manually.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.pipeline import run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the IDS dataset loading pipeline.")
    parser.add_argument("--dataset", choices=["cicids2017", "ciciot2023", "both"], default="both")
    parser.add_argument("--k-features", type=int, default=30)
    parser.add_argument(
        "--zero-day-categories",
        nargs="*",
        default=[],
        help="Attack categories (e.g. U2R R2L) to exclude entirely from training, "
             "held out for zero-day evaluation instead.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(
        dataset=args.dataset,
        k_features=args.k_features,
        zero_day_categories=args.zero_day_categories,
        test_size=args.test_size,
    )

    print(f"Train rows:     {result.n_train}")
    print(f"Test rows:      {result.n_test}")
    print(f"Zero-day rows:  {result.n_zero_day}")
    print(f"Selected features ({len(result.feature_columns)}): {result.feature_columns}")
    print(f"Class balance (pre-split, excluding zero-day holdout): {result.class_balance}")
    print("\nSaved:")
    print("  app/data/processed/train.csv")
    print("  app/data/processed/test.csv")
    print("  app/data/processed/zero_day_test.csv")
    print("  app/ml/artifacts/feature_columns.json")
    print("  app/ml/artifacts/scaler.pkl")
    print("  app/ml/artifacts/label_encoders.pkl")


if __name__ == "__main__":
    main()