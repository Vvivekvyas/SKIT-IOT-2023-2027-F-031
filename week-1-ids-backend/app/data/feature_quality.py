"""
Feature-quality and security analysis for the preprocessed dataset.

Two different concerns, both asked for by the Week 4 task:

1. FEATURE QUALITY — is each feature actually useful and well-behaved?
   (near-zero variance, redundant/duplicate features, ranked importance,
   outlier-heavy columns, missing-value residue)

2. SECURITY / LEAKAGE RISK — could a feature let the model cheat instead
   of learning real attack behavior? The classic failure mode on
   CICIDS2017 / CIC-IoT2023-style data:
     - Source/Destination IP or Port columns let the model memorize which
       *hosts* generated attacks in the lab setup, instead of learning the
       *traffic pattern* — looks great on the test split, fails completely
       on any network the model hasn't seen before.
     - A single feature that near-perfectly separates classes on its own
       is either a genuinely powerful signal or a leak (e.g. a flow-ID-like
       column, or a flag that's only ever set by the attack-generation
       script). Worth a human look either way.
     - Near-duplicate rows across what should be independent classes can
       inflate reported accuracy without the model learning anything
       generalizable.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Column names (case-insensitive substring match) that commonly leak
# host/session identity rather than describing traffic *behavior*.
IDENTITY_LEAK_PATTERNS = ["ip", "port", "flow id", "flow_id", "timestamp", "src mac", "dst mac"]


@dataclass
class FeatureQualityReport:
    n_rows: int
    n_features: int
    missing_pct: dict[str, float]
    near_zero_variance: list[str]
    redundant_pairs: list[tuple[str, str, float]]
    importance_ranking: list[tuple[str, float]]           # RandomForest importance
    mutual_info_ranking: list[tuple[str, float]]
    outlier_pct: dict[str, float]
    class_balance: dict[str, int]
    identity_leak_columns: list[str]                       # security concern
    single_feature_leak_risk: list[tuple[str, float]]       # security concern (feature, AUC)
    exact_duplicate_rows: int
    recommendations: list[str] = field(default_factory=list)


def _missing_value_report(df: pd.DataFrame) -> dict[str, float]:
    return (df.isna().mean() * 100).round(2).to_dict()


def _near_zero_variance(X: pd.DataFrame, threshold: float = 1e-4) -> list[str]:
    variances = X.var(numeric_only=True)
    return list(variances[variances < threshold].index)


def _redundant_pairs(X: pd.DataFrame, threshold: float = 0.95) -> list[tuple[str, str, float]]:
    """Pairs of features correlated above `threshold` — one of each pair is likely redundant."""
    numeric_X = X.select_dtypes(include=[np.number])
    if numeric_X.shape[1] < 2:
        return []
    corr = numeric_X.corr().abs()
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if val >= threshold:
                pairs.append((cols[i], cols[j], round(float(val), 4)))
    return sorted(pairs, key=lambda x: -x[2])


def _importance_ranking(X: pd.DataFrame, y: pd.Series) -> list[tuple[str, float]]:
    numeric_X = X.select_dtypes(include=[np.number])
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(numeric_X, y)
    ranked = sorted(zip(numeric_X.columns, model.feature_importances_), key=lambda x: -x[1])
    return [(name, round(float(score), 4)) for name, score in ranked]


def _mutual_info_ranking(X: pd.DataFrame, y: pd.Series) -> list[tuple[str, float]]:
    numeric_X = X.select_dtypes(include=[np.number])
    scores = mutual_info_classif(numeric_X, y, random_state=42)
    ranked = sorted(zip(numeric_X.columns, scores), key=lambda x: -x[1])
    return [(name, round(float(score), 4)) for name, score in ranked]


def _outlier_pct(X: pd.DataFrame) -> dict[str, float]:
    """IQR-rule outlier proportion per numeric column."""
    numeric_X = X.select_dtypes(include=[np.number])
    result = {}
    for col in numeric_X.columns:
        q1, q3 = numeric_X[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            result[col] = 0.0
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = ((numeric_X[col] < lower) | (numeric_X[col] > upper)).mean() * 100
        result[col] = round(float(outliers), 2)
    return result


def _identity_leak_columns(columns: list[str]) -> list[str]:
    flagged = []
    for col in columns:
        low = col.lower()
        if any(pattern in low for pattern in IDENTITY_LEAK_PATTERNS):
            flagged.append(col)
    return flagged


def _single_feature_leak_risk(
    X: pd.DataFrame, y: pd.Series, auc_threshold: float = 0.98
) -> list[tuple[str, float]]:
    """
    For each numeric feature alone, how well does it separate classes
    (one-vs-rest AUC, macro-averaged)? A near-1.0 AUC from a single column
    is suspicious — either a very strong real signal or a leak, worth a
    manual look either way.
    """
    numeric_X = X.select_dtypes(include=[np.number])
    y_encoded = LabelEncoder().fit_transform(y)
    n_classes = len(set(y_encoded))
    flagged = []

    for col in numeric_X.columns:
        values = numeric_X[[col]].to_numpy()
        try:
            if n_classes == 2:
                auc = roc_auc_score(y_encoded, values)
                auc = max(auc, 1 - auc)  # direction-agnostic
            else:
                # One-vs-rest per class, take the MAX rather than the mean.
                # A feature that perfectly separates even just one class from
                # all others (e.g. a leak that only distinguishes R2L) is
                # still a leak worth flagging — averaging across classes
                # would dilute that signal and let it slip through.
                aucs = []
                for c in set(y_encoded):
                    binary_y = (y_encoded == c).astype(int)
                    score = roc_auc_score(binary_y, values)
                    aucs.append(max(score, 1 - score))
                auc = float(np.max(aucs))
        except ValueError:
            continue  # constant column etc. — already caught by near-zero-variance check

        if auc >= auc_threshold:
            flagged.append((col, round(float(auc), 4)))

    return sorted(flagged, key=lambda x: -x[1])


def analyze(df: pd.DataFrame, feature_columns: list[str], label_column: str = "label") -> FeatureQualityReport:
    """
    df: the processed (cleaned, encoded) dataframe — typically
        app/data/processed/train.csv loaded back in.
    feature_columns: the exact selected feature list from
        app/ml/artifacts/feature_columns.json, so the report matches what
        the model actually trains on.
    """
    X = df[feature_columns]
    y = df[label_column]

    missing_pct = _missing_value_report(X)
    near_zero_var = _near_zero_variance(X)
    redundant = _redundant_pairs(X)
    importance = _importance_ranking(X, y)
    mutual_info = _mutual_info_ranking(X, y)
    outliers = _outlier_pct(X)
    class_balance = y.value_counts().to_dict()
    identity_leak_cols = _identity_leak_columns(feature_columns)
    single_feature_leak = _single_feature_leak_risk(X, y)
    exact_dupes = int(df.duplicated().sum())

    recommendations = []
    if near_zero_var:
        recommendations.append(
            f"Drop near-zero-variance features (carry almost no signal): {near_zero_var}"
        )
    if redundant:
        top = redundant[0]
        recommendations.append(
            f"{len(redundant)} highly correlated feature pair(s) found (e.g. "
            f"'{top[0]}' vs '{top[1]}' at r={top[2]}) — consider dropping one of each pair."
        )
    if identity_leak_cols:
        recommendations.append(
            f"Identity-like columns in the feature set: {identity_leak_cols}. These risk the "
            f"model memorizing hosts/sessions from the lab capture rather than learning traffic "
            f"behavior. Recommend excluding them from training features (fine to keep for "
            f"logging/display in the dashboard, just not as model input)."
        )
    if single_feature_leak:
        recommendations.append(
            f"{len(single_feature_leak)} feature(s) separate classes almost perfectly alone "
            f"(AUC >= 0.98): {single_feature_leak}. Verify these are genuine traffic signals, "
            f"not artifacts of how the dataset was generated."
        )
    if exact_dupes:
        recommendations.append(
            f"{exact_dupes} exact duplicate row(s) remain post-cleaning — investigate the "
            f"loading step if this number is large relative to dataset size."
        )
    if not recommendations:
        recommendations.append("No major feature-quality or leakage concerns detected.")

    return FeatureQualityReport(
        n_rows=len(df),
        n_features=len(feature_columns),
        missing_pct=missing_pct,
        near_zero_variance=near_zero_var,
        redundant_pairs=redundant,
        importance_ranking=importance,
        mutual_info_ranking=mutual_info,
        outlier_pct=outliers,
        class_balance=class_balance,
        identity_leak_columns=identity_leak_cols,
        single_feature_leak_risk=single_feature_leak,
        exact_duplicate_rows=exact_dupes,
        recommendations=recommendations,
    )