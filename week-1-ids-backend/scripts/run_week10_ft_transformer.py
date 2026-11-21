import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.ft_transformer.data_prep import FTDataPrep
from ml.ft_transformer.feature_analysis import analyze, save_report
from ml.ft_transformer.model import build_model, fit, save_model


def find_csvs(path):
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.csv"))


def load_data(path, max_files, rows_per_file):
    files = find_csvs(path)[:max_files]
    if not files:
        raise FileNotFoundError(f"no CSV found at {path}")
    frames = [pd.read_csv(f, nrows=rows_per_file) for f in files]
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "raw" / "CIC-IoT2023"))
    parser.add_argument("--target", default="label")
    parser.add_argument("--task", default="classification", choices=["classification", "regression"])
    parser.add_argument("--max-files", type=int, default=5)
    parser.add_argument("--rows-per-file", type=int, default=200000)
    parser.add_argument("--max-rows", type=int, default=300000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--d-token", type=int, default=64)
    parser.add_argument("--scaler", default="quantile", choices=["standard", "quantile"])
    parser.add_argument("--artifacts", default=str(ROOT / "ml" / "artifacts" / "ft_transformer"))
    parser.add_argument("--reports", default=str(ROOT / "docs" / "week10"))
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    df = load_data(args.data, args.max_files, args.rows_per_file)
    print("loaded:", df.shape)

    prep = FTDataPrep(target=args.target, task=args.task, numeric_scaler=args.scaler)
    frames, tensors = prep.prepare_splits(df, max_rows=args.max_rows)
    print("train/val/test:", {k: len(v) for k, v in frames.items()}, "stratified:", prep.stratified)

    report = analyze(frames, tensors, prep, d_token=args.d_token)
    save_report(report, args.reports)
    layout = report["token_layout"]
    print("sequence length:", layout["sequence_length"], "| tensor:", layout["input_tensor_shape_per_batch"])
    print("top features by mutual information:", report["mutual_information_ranking"][:5])
    print("highly correlated pairs:", len(report["high_correlation_pairs"]))

    prep.save(args.artifacts)
    if not args.skip_train:
        model = build_model(prep, d_token=args.d_token)
        history = fit(model, prep, tensors, epochs=args.epochs)
        save_model(model, Path(args.artifacts) / "model.pt")
        (Path(args.reports) / "training_history.json").write_text(json.dumps(history, indent=2))
        print("last epoch:", history[-1])


if __name__ == "__main__":
    main()
