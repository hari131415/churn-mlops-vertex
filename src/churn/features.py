"""Feature engineering.

The derived columns here are mirrored in sql/feature_view.sql so BigQuery and
the online path produce the same values. If you change one, change the other;
tests/test_features.py checks the Python side against known rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from churn.config import FeatureConfig

ADDON_COLUMNS = [
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]

TENURE_BINS = [-1, 6, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-6m", "7-12m", "13-24m", "25-48m", "49m+"]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tenure = df["tenure"].astype(float)

    # For tenure 0 there's no history; fall back to the current monthly charge.
    df["avg_monthly_spend"] = np.where(
        tenure > 0, df["total_charges"] / tenure.replace(0, np.nan), df["monthly_charges"]
    )
    df["avg_monthly_spend"] = df["avg_monthly_spend"].astype(float)

    df["num_addon_services"] = (df[ADDON_COLUMNS] == "Yes").sum(axis=1).astype(int)
    df["tenure_bucket"] = pd.cut(tenure, bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)
    return df


def build_preprocessor(features: FeatureConfig) -> ColumnTransformer:
    # Tree model, so no scaling. Imputation is there for serving-time robustness;
    # training data has no nulls after cleaning.
    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, features.numeric),
            ("cat", categorical, features.categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
