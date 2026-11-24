import pytest

from churn.drift import drift_report, psi
from churn.train import build_reference_profile


def test_psi_zero_for_identical():
    assert psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == pytest.approx(0.0)


def test_psi_grows_with_shift():
    small = psi([0.5, 0.5], [0.45, 0.55])
    large = psi([0.5, 0.5], [0.1, 0.9])
    assert 0 < small < large


def test_no_drift_on_same_distribution(clean_df, cfg):
    profile = build_reference_profile(clean_df.iloc[:750], cfg)
    report = drift_report(clean_df.iloc[750:], profile, threshold=0.2)
    assert not report["drifted"].any(), report.head()


def test_detects_contract_mix_shift(clean_df, cfg):
    profile = build_reference_profile(clean_df, cfg)
    shifted = clean_df.copy()
    shifted["contract"] = "Month-to-month"
    report = drift_report(shifted, profile, threshold=0.2).set_index("feature")
    assert report.loc["contract", "drifted"]


def test_detects_numeric_shift_out_of_range(clean_df, cfg):
    profile = build_reference_profile(clean_df, cfg)
    shifted = clean_df.copy()
    shifted["monthly_charges"] = shifted["monthly_charges"] * 3
    report = drift_report(shifted, profile, threshold=0.2).set_index("feature")
    assert report.loc["monthly_charges", "drifted"]
