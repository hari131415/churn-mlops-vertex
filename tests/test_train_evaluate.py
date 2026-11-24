import pytest

from churn.data import load_csv
from churn.evaluate import check_gate, compute_metrics, evaluate
from churn.train import load, split, train

from .conftest import REAL_DATA


def test_model_learns_signal(trained, cfg):
    model, _, _, test_df = trained
    metrics = evaluate(model, test_df, cfg)
    # Synthetic data has a strong signal; anything near 0.5 means a wiring bug.
    assert metrics["roc_auc"] > 0.75
    assert 0 < metrics["positive_rate"] < 1


def test_split_is_stratified(clean_df, cfg):
    train_df, test_df = split(clean_df, cfg)
    assert abs(train_df["churned"].mean() - test_df["churned"].mean()) < 0.02
    assert set(train_df["customer_id"]).isdisjoint(test_df["customer_id"])


def test_training_is_deterministic(clean_df, cfg):
    train_df, test_df = split(clean_df, cfg)
    a, _ = train(train_df, cfg)
    b, _ = train(train_df, cfg)
    from churn.features import add_derived_features

    X = add_derived_features(test_df)[cfg.features.all]
    assert (a.predict_proba(X) == b.predict_proba(X)).all()


def test_save_load_roundtrip(model_dir, trained, cfg):
    model, _, _, test_df = trained
    loaded, meta = load(model_dir)
    from churn.features import add_derived_features

    X = add_derived_features(test_df)[cfg.features.all]
    assert (loaded.predict_proba(X) == model.predict_proba(X)).all()
    assert meta.features == cfg.features.all
    assert set(meta.reference_profile["numeric"]) == set(cfg.features.numeric)


def test_compute_metrics_basic():
    import numpy as np

    m = compute_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]), 0.5)
    assert m["roc_auc"] == pytest.approx(0.75)
    assert m["recall_at_threshold"] == pytest.approx(0.5)
    assert m["precision_at_threshold"] == pytest.approx(1.0)


GOOD = {"roc_auc": 0.85, "pr_auc": 0.66, "recall_at_threshold": 0.75}


def test_gate_passes(cfg):
    assert check_gate(GOOD, cfg.evaluation).passed


def test_gate_fails_on_low_auc(cfg):
    result = check_gate({**GOOD, "roc_auc": 0.70}, cfg.evaluation)
    assert not result.passed
    assert "roc_auc" in result.summary()


def test_gate_blocks_regression_vs_champion(cfg):
    result = check_gate(GOOD, cfg.evaluation, champion_metrics={"roc_auc": 0.88})
    assert not result.passed
    assert "regressed" in result.summary()


def test_gate_allows_small_noise_vs_champion(cfg):
    assert check_gate(GOOD, cfg.evaluation, champion_metrics={"roc_auc": 0.855}).passed


@pytest.mark.slow
@pytest.mark.skipif(not REAL_DATA.exists(), reason="run scripts/download_data.sh first")
def test_real_data_clears_gate():
    """End-to-end on the public dataset with the production model config."""
    from churn.config import load_config

    prod_cfg = load_config()
    df = load_csv(REAL_DATA)
    train_df, test_df = split(df, prod_cfg)
    model, _ = train(train_df, prod_cfg)
    metrics = evaluate(model, test_df, prod_cfg)
    assert check_gate(metrics, prod_cfg.evaluation).passed, metrics
    # sanity: shouldn't be suspiciously good either (leakage check)
    assert metrics["roc_auc"] < 0.95
