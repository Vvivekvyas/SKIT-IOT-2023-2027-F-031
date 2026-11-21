import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.ft_transformer.data_prep import FTDataPrep
from ml.ft_transformer.feature_analysis import analyze, save_report
from ml.ft_transformer.model import build_model, fit, save_model
from ml.ft_transformer.service import MAX_RECORDS, create_app

API_KEY = "test-key-123"
HEADERS = {"X-API-Key": API_KEY}


@pytest.fixture(scope="module")
def raw_df():
    rng = np.random.default_rng(0)
    n = 600
    temperature = rng.normal(30, 5, n)
    humidity = rng.normal(55, 10, n)
    vibration = rng.gamma(2.0, 1.5, n)
    device = rng.choice(["node_a", "node_b", "node_c"], n)
    zone = rng.choice(["north", "south"], n)
    score = 0.1 * temperature + 0.05 * humidity + vibration + (device == "node_c") * 2
    label = np.where(score + rng.normal(0, 0.5, n) > np.median(score), "fault", "normal")
    df = pd.DataFrame(
        {
            "temperature": temperature,
            "humidity": humidity,
            "vibration": vibration,
            "device": device,
            "zone": zone,
            "status": label,
        }
    )
    df.loc[rng.choice(n, 20, replace=False), "humidity"] = np.nan
    return df


@pytest.fixture(scope="module")
def prepared(raw_df):
    prep = FTDataPrep(target="status", task="classification")
    frames, tensors = prep.prepare_splits(raw_df)
    return prep, frames, tensors


@pytest.fixture(scope="module")
def artifact_dir(tmp_path_factory, prepared):
    prep, frames, tensors = prepared
    directory = tmp_path_factory.mktemp("artifacts")
    model = build_model(prep, d_token=16, n_blocks=1, n_heads=2)
    fit(model, prep, tensors, epochs=2)
    prep.save(directory)
    save_model(model, directory / "model.pt")
    return directory


@pytest.fixture(scope="module")
def client(artifact_dir):
    return TestClient(create_app(artifact_dir, api_key=API_KEY))


def valid_record(**overrides):
    record = {
        "temperature": 31.2,
        "humidity": 54.0,
        "vibration": 2.8,
        "device": "node_a",
        "zone": "north",
    }
    record.update(overrides)
    return record


def test_split_shapes_and_dtypes(prepared):
    prep, frames, tensors = prepared
    x_num, x_cat, y = tensors["train"]
    assert x_num.dtype == torch.float32
    assert x_cat.dtype == torch.long
    assert x_num.shape[1] == 3
    assert x_cat.shape[1] == 2
    assert x_num.shape[0] == x_cat.shape[0] == y.shape[0]


def test_no_nan_or_inf_after_prep(prepared):
    _, _, tensors = prepared
    for x_num, x_cat, _ in tensors.values():
        assert torch.isfinite(x_num).all()
        assert (x_cat >= 0).all()


def test_train_numeric_is_standardised(prepared):
    _, _, tensors = prepared
    x_num = tensors["train"][0]
    assert np.allclose(x_num.mean(dim=0).numpy(), 0, atol=1e-4)
    assert np.allclose(x_num.std(dim=0, unbiased=False).numpy(), 1, atol=1e-3)


def test_unseen_category_maps_to_zero(prepared, raw_df):
    prep, _, _ = prepared
    df = raw_df.drop(columns=["status"]).head(3).copy()
    df["device"] = "brand_new_device"
    _, x_cat = prep.transform_features(df)
    assert (x_cat[:, prep.categorical_cols.index("device")] == 0).all()


def test_prep_save_load_roundtrip(prepared, raw_df, tmp_path):
    prep, _, _ = prepared
    prep.save(tmp_path)
    loaded = FTDataPrep.load(tmp_path)
    df = raw_df.drop(columns=["status"]).head(20)
    a = prep.transform_features(df)
    b = loaded.transform_features(df)
    assert torch.allclose(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_feature_analysis_report(prepared, tmp_path):
    prep, frames, tensors = prepared
    report = analyze(frames, tensors, prep, d_token=16)
    layout = report["token_layout"]
    assert layout["sequence_length"] == 3 + 2 + 1
    assert layout["input_tensor_shape_per_batch"] == "[batch, 6, 16]"
    assert len(report["mutual_information_ranking"]) == 5
    assert report["tensor_sanity"]["train"]["num_nan"] == 0
    out = save_report(report, tmp_path)
    assert (out / "feature_analysis.json").exists()
    assert (out / "feature_summary.csv").exists()


def test_service_refuses_to_start_without_key(artifact_dir, monkeypatch):
    monkeypatch.delenv("FT_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        create_app(artifact_dir)


def test_health_is_open(client):
    assert client.get("/health").status_code == 200


def test_predict_requires_key(client):
    r = client.post("/predict", json={"records": [valid_record()]})
    assert r.status_code == 401


def test_predict_rejects_wrong_key(client):
    r = client.post("/predict", json={"records": [valid_record()]}, headers={"X-API-Key": "nope"})
    assert r.status_code == 401


def test_schema_requires_key(client):
    assert client.get("/schema").status_code == 401
    assert client.get("/schema", headers=HEADERS).status_code == 200


def test_predict_valid_batch(client):
    r = client.post("/predict", json={"records": [valid_record(), valid_record(device="node_c")]}, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["n_records"] == 2
    assert set(body["predictions"]) <= set(body["classes"])
    for row in body["probabilities"]:
        assert abs(sum(row) - 1.0) < 1e-5


def test_predict_handles_missing_numeric_and_unseen_category(client):
    rec = valid_record(humidity=None, device="never_seen")
    r = client.post("/predict", json={"records": [rec]}, headers=HEADERS)
    assert r.status_code == 200


def test_predict_rejects_missing_column(client):
    rec = valid_record()
    rec.pop("vibration")
    r = client.post("/predict", json={"records": [rec]}, headers=HEADERS)
    assert r.status_code == 422


def test_predict_rejects_extra_column(client):
    r = client.post("/predict", json={"records": [valid_record(debug="x")]}, headers=HEADERS)
    assert r.status_code == 422


def test_predict_rejects_non_numeric(client):
    r = client.post("/predict", json={"records": [valid_record(temperature="hot")]}, headers=HEADERS)
    assert r.status_code == 422


def test_predict_rejects_boolean_as_number(client):
    r = client.post("/predict", json={"records": [valid_record(temperature=True)]}, headers=HEADERS)
    assert r.status_code == 422


def test_predict_rejects_oversized_batch(client):
    r = client.post("/predict", json={"records": [valid_record()] * (MAX_RECORDS + 1)}, headers=HEADERS)
    assert r.status_code == 422


def test_predict_rejects_empty_batch(client):
    r = client.post("/predict", json={"records": []}, headers=HEADERS)
    assert r.status_code == 422


def test_predict_rejects_long_string(client):
    r = client.post("/predict", json={"records": [valid_record(device="x" * 500)]}, headers=HEADERS)
    assert r.status_code == 422


def test_docs_endpoints_disabled(client):
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


@pytest.fixture(scope="module")
def iot_df():
    rng = np.random.default_rng(1)
    n = 900
    df = pd.DataFrame(
        {
            "flow_duration": rng.exponential(2.0, n),
            "Header_Length": rng.gamma(2.0, 20.0, n),
            "Rate": rng.exponential(50.0, n),
            "syn_count": rng.poisson(3, n).astype(float),
            "Tot size": rng.normal(120, 30, n),
        }
    )
    df["Rate_copy"] = df["Rate"] * 1.001
    df["label"] = np.select(
        [df["Rate"] > 80, df["syn_count"] > 5],
        ["DDoS-SYN_Flood", "Recon-PortScan"],
        default="BenignTraffic",
    )
    df.loc[rng.choice(n, 10, replace=False), "Rate"] = np.inf
    return df


def test_numeric_only_iot_pipeline(iot_df, tmp_path):
    prep = FTDataPrep(target="label", numeric_scaler="quantile")
    frames, tensors = prep.prepare_splits(iot_df)
    x_num, x_cat, y = tensors["train"]
    assert x_cat.shape[1] == 0
    assert x_num.shape[1] == 6
    assert torch.isfinite(x_num).all()
    model = build_model(prep, d_token=16, n_blocks=1, n_heads=2)
    out = model(x_num[:8], x_cat[:8])
    assert out.shape == (8, 3)
    report = analyze(frames, tensors, prep, d_token=16)
    assert report["token_layout"]["sequence_length"] == 7
    assert {"a", "b"} <= set(report["high_correlation_pairs"][0])
    save_report(report, tmp_path)


def test_rare_class_falls_back_to_unstratified(iot_df):
    df = iot_df.copy()
    df.loc[df.index[:1], "label"] = "RareAttack"
    prep = FTDataPrep(target="label")
    frames, tensors = prep.prepare_splits(df)
    assert prep.stratified is False
    assert sum(len(f) for f in frames.values()) == len(df)


def test_max_rows_keeps_every_class(iot_df):
    prep = FTDataPrep(target="label")
    frames, _ = prep.prepare_splits(iot_df, max_rows=300)
    total = sum(len(f) for f in frames.values())
    assert total <= 305
    assert set(pd.concat(frames.values())["label"]) == set(iot_df["label"])


def test_numeric_only_service_roundtrip(iot_df, tmp_path):
    prep = FTDataPrep(target="label", numeric_scaler="quantile")
    frames, tensors = prep.prepare_splits(iot_df)
    model = build_model(prep, d_token=16, n_blocks=1, n_heads=2)
    fit(model, prep, tensors, epochs=1)
    prep.save(tmp_path)
    save_model(model, tmp_path / "model.pt")
    api = TestClient(create_app(tmp_path, api_key=API_KEY))
    record = {c: 1.0 for c in prep.numeric_cols}
    r = api.post("/predict", json={"records": [record]}, headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["predictions"][0] in prep.classes
