"""PSI-based drift check.

Vertex Model Monitoring is the source of truth in production (see
monitoring/setup_monitoring.py). This is the same idea in plain Python so we
can sanity-check a batch locally or in a notebook during an incident without
waiting for the next scheduled monitoring run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn.features import add_derived_features

EPS = 1e-6


def psi(expected: list[float], actual: list[float]) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def feature_psi(df: pd.DataFrame, profile: dict) -> dict[str, float]:
    df = add_derived_features(df)
    out: dict[str, float] = {}

    for col, ref in profile.get("numeric", {}).items():
        edges = list(ref["edges"])
        # Open-ended outer bins so out-of-range serving values still get counted.
        edges[0], edges[-1] = -np.inf, np.inf
        actual = pd.cut(df[col], bins=edges).value_counts(normalize=True, sort=False).tolist()
        out[col] = psi(ref["freqs"], actual)

    for col, ref in profile.get("categorical", {}).items():
        cats = sorted(set(ref) | set(df[col].astype(str).unique()))
        actual_freq = df[col].astype(str).value_counts(normalize=True)
        out[col] = psi([ref.get(c, 0.0) for c in cats], [actual_freq.get(c, 0.0) for c in cats])

    return out


def drift_report(df: pd.DataFrame, profile: dict, threshold: float = 0.2) -> pd.DataFrame:
    scores = feature_psi(df, profile)
    report = pd.DataFrame({"feature": list(scores), "psi": list(scores.values())})
    report["drifted"] = report["psi"] > threshold
    return report.sort_values("psi", ascending=False).reset_index(drop=True)
