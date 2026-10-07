"""
Security-oriented feature analysis — Week 6.

Goes beyond Week 4's general feature-quality/leakage checks to ask a
narrower security question: if an attacker knows (or guesses) which
features the model relies on, how easily could they evade detection by
manipulating them?

Three checks:
1. SPOOFABILITY — rule-based: is a feature a raw per-packet field an
   attacker directly controls (packet size, TTL, flags, port numbers), or
   a statistical aggregate computed over many packets (mean/std/rate/IAT)
   that's much harder for a single attacker-crafted packet to fake?
2. PERTURBATION ROBUSTNESS — using the trained model: nudge each feature
   slightly and measure how often the prediction flips. A feature the
   model leans on heavily AND that flips easily under a small nudge is a
   practical evasion vector.
3. ZERO-DAY DISTRIBUTION SHIFT — compares each feature's distribution on
   the training pool vs. the held-out zero-day attack categories (from the
   pipeline's zero_day_test.csv). A feature whose distribution sits far
   from training on unseen attacks may not generalize — importance learned
   on known attacks doesn't guarantee it transfers.

These combine into a per-feature SECURITY RISK ranking: features that are
both important to the model AND easy for an attacker to manipulate are the
ones worth hardening or dropping first.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

ATTACKER_CONTROLLABLE_PATTERNS = [
    "flag", "ttl", "window", "port", "header", "payload", "protocol",
    "fin", "syn", "rst", "psh", "ack", "urg", "ece", "cwr", "len", "size",
]
AGGREGATE_PATTERNS = [
    "mean", "std", "avg", "variance", "var", "total", "rate", "iat",
    "duration", "count", "sum", "bytes/s", "packets/s",
]


def classify_spoofability(feature_name: str) -> str:
    """
    Rule-based, name-pattern classification. Not a substitute for a human
    security review, but a fast first pass across dozens of flow-statistics
    columns. Checked in this order: aggregate patterns first, since a name
    like "Flow Bytes/s" would otherwise also match "size"-adjacent terms.
    """
    name = feature_name.lower()
    if any(p in name for p in AGGREGATE_PATTERNS):
        return "aggregate (harder to spoof)"
    if any(p in name for p in ATTACKER_CONTROLLABLE_PATTERNS):
        return "attacker-controllable (per-packet field)"
    return "unclassified \u2014 manual review"


_SPOOFABILITY_WEIGHT = {
    "attacker-controllable (per-packet field)": 1.0,
    "unclassified \u2014 manual review": 0.5,
    "aggregate (harder to spoof)": 0.2,
}


def perturbation_robustness(
    model,
    X: pd.DataFrame,
    feature_columns: list[str],
    epsilon: float = 0.5,
    n_samples: int = 200,
    random_state: int = 42,
) -> dict[str, float]:
    """
    For each feature, perturb it by epsilon standard deviations (on a
    random sample of rows) and measure the % of predictions that flip.
    X is expected to already be in the model's input scale (e.g. the
    scaled train split from the pipeline).
    """
    n = min(n_samples, len(X))
    sample = X.sample(n=n, random_state=random_state).reset_index(drop=True)

    baseline = model.predict(sample[feature_columns].to_numpy())

    flip_rates: dict[str, float] = {}
    for col in feature_columns:
        perturbed = sample.copy()
        std = perturbed[col].std()
        if std == 0 or np.isnan(std):
            flip_rates[col] = 0.0
            continue
        perturbed[col] = perturbed[col] + epsilon * std
        new_preds = model.predict(perturbed[feature_columns].to_numpy())
        flip_rate = float(np.mean(new_preds != baseline)) * 100
        flip_rates[col] = round(flip_rate, 2)

    return flip_rates


def zero_day_distribution_shift(
    train_df: pd.DataFrame, zero_day_df: pd.DataFrame, feature_columns: list[str]
) -> dict[str, float]:
    """
    Standardized mean difference per feature between the training pool and
    the held-out zero-day attack set: |mean_zeroday - mean_train| / std_train.
    A value above ~1.0 means the feature sits over a full standard deviation
    away on average for unseen attacks — importance learned on known
    attacks may not transfer to that feature for genuinely new ones.
    """
    shifts: dict[str, float] = {}
    if len(zero_day_df) == 0:
        return shifts

    for col in feature_columns:
        if col not in zero_day_df.columns:
            continue
        train_mean, train_std = train_df[col].mean(), train_df[col].std()
        zd_mean = zero_day_df[col].mean()
        if train_std == 0 or np.isnan(train_std):
            shifts[col] = 0.0
            continue
        shifts[col] = round(float(abs(zd_mean - train_mean) / train_std), 3)
    return shifts


@dataclass
class SecurityFeatureRanking:
    feature: str
    importance: float                  # normalized 0-1, from importance ranking
    spoofability: str
    flip_rate_pct: float | None        # None if no model/perturbation data supplied
    distribution_shift: float | None   # None if no zero-day data supplied
    risk_score: float


def rank_security_risk(
    importance_ranking: list[tuple[str, float]],
    flip_rates: dict[str, float] | None = None,
    distribution_shifts: dict[str, float] | None = None,
) -> list[SecurityFeatureRanking]:
    """
    Combines importance (e.g. from a RandomForest importance ranking) with
    spoofability and, when available, perturbation flip rate, into a single
    risk ranking. Highest risk = important to the model AND easy to spoof
    (and/or flips the prediction easily under a small nudge).
    """
    if not importance_ranking:
        return []
    max_importance = max(score for _, score in importance_ranking) or 1.0

    results = []
    for feature, importance in importance_ranking:
        norm_importance = importance / max_importance
        spoof_label = classify_spoofability(feature)
        weight = _SPOOFABILITY_WEIGHT[spoof_label]

        flip_rate = flip_rates.get(feature) if flip_rates else None
        shift = distribution_shifts.get(feature) if distribution_shifts else None

        if flip_rate is not None:
            risk = 0.5 * norm_importance * weight + 0.5 * (flip_rate / 100)
        else:
            risk = norm_importance * weight

        results.append(SecurityFeatureRanking(
            feature=feature,
            importance=round(norm_importance, 4),
            spoofability=spoof_label,
            flip_rate_pct=flip_rate,
            distribution_shift=shift,
            risk_score=round(risk, 4),
        ))

    return sorted(results, key=lambda r: -r.risk_score)