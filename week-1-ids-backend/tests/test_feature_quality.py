import numpy as np
import pandas as pd

from app.data.feature_quality import analyze
from app.data.feature_quality_report import to_markdown


def _synthetic_df(n=300, seed=1):
    rng = np.random.RandomState(seed)
    label = rng.choice(["Normal", "DoS", "Probe", "R2L"], size=n, p=[0.6, 0.2, 0.1, 0.1])

    df = pd.DataFrame({
        "flow_duration": rng.exponential(1000, n),
        "fwd_packets": rng.poisson(5, n),
    })
    df["flow_duration_dup"] = df["flow_duration"] * 1.0001  # redundant
    df["constant_col"] = 1.0                                 # near-zero variance

    label_to_num = {"Normal": 0.0, "DoS": 10.0, "Probe": 20.0, "R2L": 30.0}
    df["leak_col"] = [label_to_num[l] + rng.normal(0, 0.01) for l in label]  # planted leak

    df["Destination Port"] = rng.randint(1, 65535, n)         # identity-style name
    df["label"] = label
    return df


def test_analyze_flags_all_planted_issues():
    df = _synthetic_df()
    feature_columns = [
        "flow_duration", "fwd_packets", "flow_duration_dup",
        "constant_col", "leak_col", "Destination Port",
    ]
    report = analyze(df, feature_columns)

    assert "constant_col" in report.near_zero_variance
    assert any({"flow_duration", "flow_duration_dup"} == {a, b} for a, b, _ in report.redundant_pairs)
    assert "Destination Port" in report.identity_leak_columns
    assert any(name == "leak_col" for name, _auc in report.single_feature_leak_risk)


def test_analyze_clean_data_has_no_false_positive_leak():
    rng = np.random.RandomState(2)
    n = 300
    df = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
        "label": rng.choice(["Normal", "DoS"], size=n),
    })
    report = analyze(df, ["a", "b"])
    assert report.single_feature_leak_risk == []
    assert report.near_zero_variance == []
    assert report.redundant_pairs == []


def test_to_markdown_produces_nonempty_report():
    df = _synthetic_df()
    report = analyze(df, ["flow_duration", "fwd_packets", "leak_col"])
    md = to_markdown(report)
    assert "# Feature Quality & Security Analysis Report" in md
    assert "leak_col" in md
    