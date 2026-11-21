import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler


class FTDataPrep:
    def __init__(
        self,
        target,
        task="classification",
        numeric_cols=None,
        categorical_cols=None,
        max_cat_cardinality=50,
        numeric_scaler="standard",
        seed=42,
    ):
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        self.target = target
        self.task = task
        self.numeric_cols = list(numeric_cols) if numeric_cols is not None else None
        self.categorical_cols = list(categorical_cols) if categorical_cols is not None else None
        self.max_cat_cardinality = max_cat_cardinality
        self.numeric_scaler = numeric_scaler
        self.seed = seed
        self.medians = {}
        self.category_maps = {}
        self.classes = []
        self.scaler = None
        self.fitted = False
        self.stratified = False

    @property
    def feature_cols(self):
        return list(self.numeric_cols) + list(self.categorical_cols)

    @property
    def cardinalities(self):
        return [len(self.category_maps[c]) + 1 for c in self.categorical_cols]

    @property
    def n_outputs(self):
        return len(self.classes) if self.task == "classification" else 1

    def infer_columns(self, df):
        features = [c for c in df.columns if c != self.target]
        if self.numeric_cols is None and self.categorical_cols is None:
            self.numeric_cols = [
                c for c in features if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
            ]
            self.categorical_cols = [c for c in features if c not in self.numeric_cols]
        elif self.numeric_cols is None:
            self.numeric_cols = [c for c in features if c not in self.categorical_cols]
        elif self.categorical_cols is None:
            self.categorical_cols = [c for c in features if c not in self.numeric_cols]
        for c in self.categorical_cols:
            if df[c].nunique(dropna=True) > self.max_cat_cardinality:
                raise ValueError(f"column '{c}' has too many categories for embedding")

    def fit(self, df):
        if self.target not in df.columns:
            raise KeyError(f"target column '{self.target}' not found")
        self.infer_columns(df)
        for c in self.numeric_cols:
            col = pd.to_numeric(df[c], errors="coerce")
            self.medians[c] = float(col.median()) if col.notna().any() else 0.0
        for c in self.categorical_cols:
            values = sorted(df[c].dropna().astype(str).unique())
            self.category_maps[c] = {v: i + 1 for i, v in enumerate(values)}
        if self.numeric_cols:
            raw = self._numeric_frame(df)
            if self.numeric_scaler == "quantile":
                self.scaler = QuantileTransformer(
                    n_quantiles=min(1000, max(10, len(df))),
                    output_distribution="normal",
                    random_state=self.seed,
                )
            else:
                self.scaler = StandardScaler()
            self.scaler.fit(raw)
        if self.task == "classification":
            self.classes = [str(v) for v in sorted(df[self.target].dropna().astype(str).unique())]
        self.fitted = True
        return self

    def _numeric_frame(self, df):
        out = np.zeros((len(df), len(self.numeric_cols)), dtype=np.float64)
        for j, c in enumerate(self.numeric_cols):
            col = pd.to_numeric(df[c], errors="coerce").astype(float)
            col = col.replace([np.inf, -np.inf], np.nan).fillna(self.medians[c])
            out[:, j] = col.to_numpy()
        return out

    def transform_features(self, df):
        if not self.fitted:
            raise RuntimeError("call fit() before transform")
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"missing feature columns: {missing}")
        if self.numeric_cols:
            x_num = self.scaler.transform(self._numeric_frame(df)).astype(np.float32)
        else:
            x_num = np.zeros((len(df), 0), dtype=np.float32)
        x_cat = np.zeros((len(df), len(self.categorical_cols)), dtype=np.int64)
        for j, c in enumerate(self.categorical_cols):
            mapping = self.category_maps[c]
            x_cat[:, j] = df[c].map(lambda v: mapping.get(str(v), 0) if pd.notna(v) else 0).to_numpy()
        return torch.from_numpy(x_num), torch.from_numpy(x_cat)

    def transform_target(self, df):
        if self.task == "classification":
            index = {c: i for i, c in enumerate(self.classes)}
            y = df[self.target].astype(str).map(index)
            if y.isna().any():
                raise ValueError("target contains labels unseen during fit")
            return torch.tensor(y.to_numpy(), dtype=torch.long)
        return torch.tensor(pd.to_numeric(df[self.target]).to_numpy(), dtype=torch.float32)

    def transform(self, df):
        x_num, x_cat = self.transform_features(df)
        y = self.transform_target(df) if self.target in df.columns else None
        return x_num, x_cat, y

    def prepare_splits(self, df, val_size=0.15, test_size=0.15, max_rows=None):
        df = df.dropna(subset=[self.target]).reset_index(drop=True)
        if max_rows and len(df) > max_rows:
            if self.task == "classification":
                frac = max_rows / len(df)
                parts = [
                    g.sample(n=max(1, int(round(len(g) * frac))), random_state=self.seed)
                    for _, g in df.groupby(self.target)
                ]
                df = pd.concat(parts).reset_index(drop=True)
            else:
                df = df.sample(n=max_rows, random_state=self.seed).reset_index(drop=True)
        stratify = None
        if self.task == "classification" and df[self.target].value_counts().min() >= 3:
            stratify = df[self.target]
        self.stratified = stratify is not None
        train_df, rest_df = train_test_split(
            df, test_size=val_size + test_size, random_state=self.seed, stratify=stratify
        )
        stratify_rest = None
        if self.stratified and rest_df[self.target].value_counts().min() >= 2:
            stratify_rest = rest_df[self.target]
        val_df, test_df = train_test_split(
            rest_df,
            test_size=test_size / (val_size + test_size),
            random_state=self.seed,
            stratify=stratify_rest,
        )
        self.fit(train_df)
        frames = {"train": train_df, "val": val_df, "test": test_df}
        tensors = {name: self.transform(frame) for name, frame in frames.items()}
        return frames, tensors

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "target": self.target,
            "task": self.task,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "max_cat_cardinality": self.max_cat_cardinality,
            "numeric_scaler": self.numeric_scaler,
            "seed": self.seed,
            "medians": self.medians,
            "category_maps": self.category_maps,
            "classes": self.classes,
            "cardinalities": self.cardinalities,
        }
        (directory / "prep_meta.json").write_text(json.dumps(meta, indent=2))
        joblib.dump(self.scaler, directory / "numeric_scaler.joblib")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = json.loads((directory / "prep_meta.json").read_text())
        prep = cls(
            target=meta["target"],
            task=meta["task"],
            numeric_cols=meta["numeric_cols"],
            categorical_cols=meta["categorical_cols"],
            max_cat_cardinality=meta["max_cat_cardinality"],
            numeric_scaler=meta["numeric_scaler"],
            seed=meta["seed"],
        )
        prep.medians = meta["medians"]
        prep.category_maps = meta["category_maps"]
        prep.classes = meta["classes"]
        prep.scaler = joblib.load(directory / "numeric_scaler.joblib")
        prep.fitted = True
        return prep