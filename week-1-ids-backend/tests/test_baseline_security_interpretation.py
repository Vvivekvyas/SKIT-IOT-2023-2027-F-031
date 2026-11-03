"""
Tests for scripts/run_baseline_security_interpretation.py
Place in: tests/test_baseline_security_interpretation.py
Run:      pytest tests/test_baseline_security_interpretation.py
"""

import importlib.util
import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "run_baseline_security_interpretation.py"
)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("baseline_sec", SCRIPT_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture()
def workspace(tmp_path):
    """Synthetic artifacts: 4 classes, one identity-style feature (src_port)."""
    rng = np.random.default_rng(0)
    feature_cols = ["duration", "src_bytes", "dst_bytes", "src_port"]
    classes = ["Normal", "DoS", "Probe", "R2L"]

    def make(n):
        y = rng.choice(classes, size=n, p=[0.5, 0.25, 0.15, 0.10])
        X = rng.normal(size=(n, len(feature_cols)))
        X[:, 0] += np.where(y == "DoS", 3, 0)
        X[:, 1] += np.where(y == "Probe", 3, 0)
        return pd.DataFrame(X, columns=feature_cols).assign(label=y)

    train, test = make(600), make(300)
    zero_day = make(50).assign(label="U2R")

    art = tmp_path / "artifacts"
    proc = tmp_path / "processed"
    art.mkdir()
    proc.mkdir()

    model = RandomForestClassifier(n_estimators=20, random_state=0)
    model.fit(train[feature_cols], train["label"])
    joblib.dump(model, art / "best_model.pkl")
    with open(art / "feature_columns.json", "w") as f:
        json.dump({"feature_columns": feature_cols}, f)

    test.to_csv(proc / "test.csv", index=False)
    zero_day.to_csv(proc / "zero_day_test.csv", index=False)
    return tmp_path, art, proc


def test_full_run_writes_report(mod, workspace):
    tmp_path, art, proc = workspace
    docs = tmp_path / "docs"
    mod.run(str(art), str(proc), str(docs))

    report = (docs / "baseline_security_interpretation.md").read_text(encoding="utf-8")
    assert "Dangerous false negatives" in report
    assert "Zero-day holdout" in report
    assert "Feature reliance" in report


def test_miss_rates_are_valid_percentages(mod, workspace):
    _, art, proc = workspace
    model, cols, test_df, _ = mod.load_artifacts(str(art), str(proc))
    _, _, misses = mod.analyze_false_negatives(model, cols, test_df)

    assert "Normal" not in misses
    assert set(misses) <= {"DoS", "Probe", "R2L"}
    for info in misses.values():
        assert 0 <= info["miss_rate_percent"] <= 100
        assert info["missed_as_normal"] <= info["total_samples"]


def test_zero_day_counts_add_up(mod, workspace):
    _, art, proc = workspace
    model, cols, _, zero_day_df = mod.load_artifacts(str(art), str(proc))
    res = mod.analyze_zero_day(model, cols, zero_day_df)

    assert res["total_zero_day_samples"] == 50
    assert res["flagged_as_any_attack"] + res["flagged_as_normal_missed"] == 50


def test_zero_day_missing_file_is_skipped(mod, workspace):
    _, art, proc = workspace
    os.remove(proc / "zero_day_test.csv")
    _, _, _, zero_day_df = mod.load_artifacts(str(art), str(proc))
    assert zero_day_df is None
    assert mod.analyze_zero_day(None, [], zero_day_df) is None


def test_identity_feature_is_flagged(mod):
    class FakeModel:
        feature_importances_ = np.array([0.1, 0.2, 0.1, 0.6])

    top, flagged = mod.analyze_feature_reliance(
        FakeModel(), ["duration", "src_bytes", "dst_bytes", "src_port"]
    )
    assert "src_port" in flagged
    assert top.index[0] == "src_port"


def test_non_tree_model_returns_none(mod):
    class NoImportances:
        pass

    top, flagged = mod.analyze_feature_reliance(NoImportances(), ["a", "b"])
    assert top is None and flagged == []
    