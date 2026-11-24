"""Loading and cleaning the raw telco subscription data.

Source columns come from the public IBM Telco Customer Churn dataset. We rename
to snake_case once here, so nothing downstream has to know about the original
CamelCase headers.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

RAW_TO_CLEAN = {
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churned",
}

REQUIRED_COLUMNS = set(RAW_TO_CLEAN.values()) - {"churned"}


class DataValidationError(ValueError):
    """Raised when an input frame fails basic contract checks."""


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names and types. Safe to call on already-clean frames."""
    df = df.rename(columns=RAW_TO_CLEAN).copy()

    # TotalCharges is blank for brand-new customers (tenure == 0) in the source
    # data. They haven't been billed yet, so 0 is the honest value.
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce").fillna(0.0)
    df["monthly_charges"] = pd.to_numeric(df["monthly_charges"], errors="coerce")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").astype("Int64")

    # SeniorCitizen is 0/1 in the CSV but Yes/No everywhere else; keep it categorical.
    if pd.api.types.is_numeric_dtype(df["senior_citizen"]):
        df["senior_citizen"] = df["senior_citizen"].map({1: "Yes", 0: "No"})

    if "churned" in df.columns and not pd.api.types.is_numeric_dtype(df["churned"]):
        df["churned"] = (df["churned"].astype(str).str.strip().str.lower() == "yes").astype(int)

    return df


def validate(df: pd.DataFrame, require_label: bool = True) -> None:
    """Lightweight data contract. Fails loudly instead of training on garbage."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if require_label and "churned" not in df.columns:
        missing.add("churned")
    if missing:
        raise DataValidationError(f"missing columns: {sorted(missing)}")

    if df["customer_id"].duplicated().any():
        raise DataValidationError("duplicate customer_id values")

    if (df["tenure"] < 0).any():
        raise DataValidationError("negative tenure")

    if df["monthly_charges"].isna().any() or (df["monthly_charges"] < 0).any():
        raise DataValidationError("monthly_charges has nulls or negatives")

    if require_label:
        labels = set(df["churned"].unique())
        if not labels <= {0, 1}:
            raise DataValidationError(f"unexpected label values: {labels}")
        rate = df["churned"].mean()
        # The public dataset sits around 26%. Anything wildly off means the
        # upstream query or join is broken, not that churn tripled overnight.
        if not 0.05 <= rate <= 0.6:
            raise DataValidationError(f"churn rate {rate:.2%} outside expected range")


def load_csv(path: str | Path) -> pd.DataFrame:
    df = clean(pd.read_csv(path))
    validate(df)
    log.info("loaded %d rows from %s (churn rate %.3f)", len(df), path, df["churned"].mean())
    return df


def load_bigquery(project: str, table_or_query: str) -> pd.DataFrame:
    """Read from BigQuery. Accepts a fully-qualified table id or a SQL query."""
    from google.cloud import bigquery

    client = bigquery.Client(project=project)
    sql = table_or_query
    if not table_or_query.lstrip().lower().startswith("select"):
        sql = f"SELECT * FROM `{table_or_query}`"
    df = client.query(sql).to_dataframe()
    df = clean(df)
    validate(df)
    log.info("loaded %d rows from BigQuery", len(df))
    return df
