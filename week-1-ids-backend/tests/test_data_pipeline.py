import numpy as np
import pandas as pd

from app.data.label_mapping import map_label
from app.data.preprocessing import clean_dataframe, encode_categoricals


def test_label_mapping_known_and_unknown():
    assert map_label("BENIGN", "cicids2017") == "Normal"
    assert map_label("DDoS", "cicids2017") == "DoS"
    assert map_label("SomeNewAttackType", "cicids2017") == "Other"
    assert map_label("BenignTraffic", "ciciot2023") == "Normal"
    assert map_label("Mirai-udpplain", "ciciot2023") == "U2R"


def test_clean_dataframe_drops_inf_nan_and_duplicates():
    df = pd.DataFrame({
        "a": [1, 2, np.inf, 2],
        "b": [1, np.nan, 3, 1],
        "label": ["Normal", "DoS", "Normal", "Normal"],
    })
    cleaned = clean_dataframe(df)
    # row 1 (nan) and row 2 (inf) get dropped; rows 0 and 3 are distinct (a differs), both survive
    assert len(cleaned) == 2
    assert not cleaned.isna().any().any()
    assert not np.isinf(cleaned.to_numpy(dtype=object, na_value=0)[:, :2].astype(float)).any()


def test_clean_dataframe_drops_exact_duplicate_rows():
    df = pd.DataFrame({
        "a": [1, 1, 2],
        "b": [5, 5, 6],
        "label": ["Normal", "Normal", "DoS"],
    })
    cleaned = clean_dataframe(df)
    assert len(cleaned) == 2


def test_encode_categoricals_only_touches_object_columns():
    df = pd.DataFrame({
        "protocol": ["TCP", "UDP", "TCP"],
        "duration": [1.0, 2.0, 3.0],
        "label": ["Normal", "DoS", "Normal"],
    })
    encoded, encoders = encode_categoricals(df)
    assert encoded["protocol"].dtype != object
    assert "protocol" in encoders
    assert "duration" not in encoders
    