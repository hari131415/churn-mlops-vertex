import pandas as pd
import pytest

from churn.features import add_derived_features, build_preprocessor


def _row(**overrides):
    base = {
        "tenure": 10,
        "monthly_charges": 50.0,
        "total_charges": 480.0,
        "online_security": "Yes",
        "online_backup": "No",
        "device_protection": "Yes",
        "tech_support": "No internet service",
        "streaming_tv": "Yes",
        "streaming_movies": "No",
    }
    return {**base, **overrides}


def test_avg_monthly_spend():
    out = add_derived_features(pd.DataFrame([_row()]))
    assert out["avg_monthly_spend"].iloc[0] == pytest.approx(48.0)


def test_avg_monthly_spend_for_new_customer_uses_monthly_charge():
    out = add_derived_features(pd.DataFrame([_row(tenure=0, total_charges=0.0)]))
    assert out["avg_monthly_spend"].iloc[0] == pytest.approx(50.0)


def test_addon_count_ignores_no_internet_service():
    out = add_derived_features(pd.DataFrame([_row()]))
    assert out["num_addon_services"].iloc[0] == 3


@pytest.mark.parametrize(
    "tenure,bucket",
    [(0, "0-6m"), (6, "0-6m"), (7, "7-12m"), (24, "13-24m"), (48, "25-48m"), (72, "49m+")],
)
def test_tenure_buckets_match_sql(tenure, bucket):
    # Boundaries must line up with the CASE in sql/feature_view.sql
    out = add_derived_features(pd.DataFrame([_row(tenure=tenure)]))
    assert out["tenure_bucket"].iloc[0] == bucket


def test_does_not_mutate_input():
    df = pd.DataFrame([_row()])
    add_derived_features(df)
    assert "avg_monthly_spend" not in df.columns


def test_preprocessor_handles_unseen_category(clean_df, cfg):
    df = add_derived_features(clean_df)
    pre = build_preprocessor(cfg.features).fit(df[cfg.features.all])
    unseen = df[cfg.features.all].head(1).copy()
    unseen["payment_method"] = "Crypto"
    assert pre.transform(unseen).shape[0] == 1
