"""
Loaders for the two datasets named in the proposal's Technical Scope slide:
CIC-IoT2023 and CICIDS2017. Both ship as multiple CSV files (one per
capture/attack session) — these functions just find and concatenate them.

Download the datasets yourself (they're large, not included here):
  CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
  CIC-IoT2023: https://www.unb.ca/cic/datasets/iotdataset-2023.html

Place the raw CSVs under:
  app/data/raw/cicids2017/*.csv
  app/data/raw/ciciot2023/*.csv
"""
from pathlib import Path

import pandas as pd

from app.data.label_mapping import map_label


def _load_and_concat_csvs(folder: Path) -> pd.DataFrame:
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {folder}. Download the dataset and place "
            f"its CSVs there first (see app/data/loaders.py docstring)."
        )
    frames = [pd.read_csv(f, low_memory=False) for f in csv_files]
    df = pd.concat(frames, ignore_index=True)
    # Both datasets are notorious for stray leading/trailing spaces in column names.
    df.columns = [c.strip() for c in df.columns]
    return df


def load_cicids2017(raw_dir: str | Path = "app/data/raw/cicids2017") -> pd.DataFrame:
    df = _load_and_concat_csvs(Path(raw_dir))
    if "Label" not in df.columns:
        raise KeyError("Expected a 'Label' column in CICIDS2017 CSVs, none found.")
    df = df.rename(columns={"Label": "raw_label"})
    df["label"] = df["raw_label"].apply(lambda x: map_label(str(x), "cicids2017"))
    df["dataset_source"] = "cicids2017"
    return df


def load_cic_iot2023(raw_dir: str | Path = "app/data/raw/ciciot2023") -> pd.DataFrame:
    df = _load_and_concat_csvs(Path(raw_dir))
    label_col = "label" if "label" in df.columns else "Label"
    if label_col not in df.columns:
        raise KeyError("Expected a 'label'/'Label' column in CIC-IoT2023 CSVs, none found.")
    df = df.rename(columns={label_col: "raw_label"})
    df["label"] = df["raw_label"].apply(lambda x: map_label(str(x), "ciciot2023"))
    df["dataset_source"] = "ciciot2023"
    return df


def load_combined(
    cicids_dir: str | Path = "app/data/raw/cicids2017",
    ciciot_dir: str | Path = "app/data/raw/ciciot2023",
    include: tuple[str, ...] = ("cicids2017", "ciciot2023"),
) -> pd.DataFrame:
    """Loads whichever of the two datasets are requested and stacks them into one frame."""
    frames = []
    if "cicids2017" in include:
        frames.append(load_cicids2017(cicids_dir))
    if "ciciot2023" in include:
        frames.append(load_cic_iot2023(ciciot_dir))
    if not frames:
        raise ValueError("include must contain at least one of 'cicids2017', 'ciciot2023'")

    # Only keep columns common to whichever datasets were loaded — the two
    # datasets don't share an identical feature schema, so a union would
    # leave the other dataset's rows full of NaNs for those columns.
    common_cols = set(frames[0].columns)
    for f in frames[1:]:
        common_cols &= set(f.columns)
    common_cols = sorted(common_cols)

    return pd.concat([f[common_cols] for f in frames], ignore_index=True)