import pandas as pd
import pytest

from churn.data import DataValidationError, clean, validate


def test_clean_renames_and_types(raw_df):
    df = clean(raw_df)
    assert "customer_id" in df.columns and "customerID" not in df.columns
    assert set(df["churned"].unique()) <= {0, 1}
    assert set(df["senior_citizen"].unique()) <= {"Yes", "No"}
    assert pd.api.types.is_float_dtype(df["total_charges"])


def test_blank_total_charges_become_zero(raw_df):
    df = clean(raw_df)
    new_customers = df[df["tenure"] == 0]
    assert len(new_customers) > 0
    assert (new_customers["total_charges"] == 0).all()


def test_clean_is_idempotent(clean_df):
    again = clean(clean_df)
    pd.testing.assert_frame_equal(again, clean_df)


def test_validate_passes_on_good_data(clean_df):
    validate(clean_df)


def test_validate_rejects_missing_columns(clean_df):
    with pytest.raises(DataValidationError, match="missing columns"):
        validate(clean_df.drop(columns=["contract"]))


def test_validate_rejects_duplicate_ids(clean_df):
    dup = pd.concat([clean_df, clean_df.head(1)])
    with pytest.raises(DataValidationError, match="duplicate"):
        validate(dup)


def test_validate_rejects_negative_charges(clean_df):
    bad = clean_df.copy()
    bad.loc[bad.index[0], "monthly_charges"] = -5
    with pytest.raises(DataValidationError, match="monthly_charges"):
        validate(bad)


def test_validate_catches_broken_label_join(clean_df):
    # e.g. an upstream LEFT JOIN that nulls out most labels -> coerced to 0
    bad = clean_df.copy()
    bad["churned"] = 0
    with pytest.raises(DataValidationError, match="churn rate"):
        validate(bad)


def test_validate_without_label_for_scoring(clean_df):
    validate(clean_df.drop(columns=["churned"]), require_label=False)
