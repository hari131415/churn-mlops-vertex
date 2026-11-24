"""Model training.

Produces a single sklearn Pipeline (preprocessing + XGBoost) so serving loads
one object and can't drift from how training transformed the data.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn import __version__
from churn.config import Config
from churn.features import add_derived_features, build_preprocessor

log = logging.getLogger(__name__)

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"


@dataclass
class ModelMetadata:
    model_version: str
    trained_at: str
    package_version: str
    features: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    decision_threshold: float
    train_rows: int
    best_iteration: int | None
    metrics: dict[str, float] = field(default_factory=dict)
    # Per-feature reference stats, used by the serving app's local drift check.
    reference_profile: dict = field(default_factory=dict)


def split(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(
        df,
        test_size=cfg.data["test_size"],
        random_state=cfg.data["random_state"],
        stratify=df[cfg.label],
    )


def _xgb_params(cfg: Config, pos_weight: float) -> dict:
    params = {k: v for k, v in cfg.model.items() if k != "early_stopping_rounds"}
    return {
        **params,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "scale_pos_weight": pos_weight,
        "random_state": cfg.data["random_state"],
        "n_jobs": -1,
    }


def train(train_df: pd.DataFrame, cfg: Config) -> tuple[Pipeline, int | None]:
    """Fit preprocessing + XGBoost. Holds out 15% of train for early stopping."""
    feats = cfg.features
    train_df = add_derived_features(train_df)

    fit_df, es_df = train_test_split(
        train_df,
        test_size=0.15,
        random_state=cfg.data["random_state"],
        stratify=train_df[cfg.label],
    )

    pre = build_preprocessor(feats)
    X_fit = pre.fit_transform(fit_df[feats.all])
    X_es = pre.transform(es_df[feats.all])
    y_fit, y_es = fit_df[cfg.label], es_df[cfg.label]

    pos_weight = float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))
    clf = XGBClassifier(
        **_xgb_params(cfg, pos_weight),
        early_stopping_rounds=cfg.model.get("early_stopping_rounds"),
    )
    clf.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
    best_iter = getattr(clf, "best_iteration", None)
    log.info("early stopping at iteration %s", best_iter)

    # Refit on the full training split with the tuned number of trees, without
    # early stopping, so we don't throw away 15% of the data.
    n_trees = (best_iter + 1) if best_iter is not None else cfg.model["n_estimators"]
    final_params = {**_xgb_params(cfg, pos_weight), "n_estimators": n_trees}
    model = Pipeline([("pre", build_preprocessor(feats)), ("clf", XGBClassifier(**final_params))])
    model.fit(train_df[feats.all], train_df[cfg.label])
    return model, best_iter


def build_reference_profile(df: pd.DataFrame, cfg: Config, bins: int = 10) -> dict:
    """Quantile bins for numeric features and category frequencies for the rest."""
    df = add_derived_features(df)
    profile: dict = {"numeric": {}, "categorical": {}}
    for col in cfg.features.numeric:
        edges = df[col].quantile([i / bins for i in range(bins + 1)]).unique().tolist()
        counts, _ = pd.cut(df[col], bins=edges, include_lowest=True, retbins=True)
        freqs = counts.value_counts(normalize=True, sort=False).tolist()
        profile["numeric"][col] = {"edges": edges, "freqs": freqs}
    for col in cfg.features.categorical:
        profile["categorical"][col] = df[col].astype(str).value_counts(normalize=True).to_dict()
    return profile


def save(model: Pipeline, metadata: ModelMetadata, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / MODEL_FILENAME)
    (out / METADATA_FILENAME).write_text(json.dumps(asdict(metadata), indent=2))
    log.info("saved model to %s", out)
    return out


def load(model_dir: str | Path) -> tuple[Pipeline, ModelMetadata]:
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / MODEL_FILENAME)
    meta = ModelMetadata(**json.loads((model_dir / METADATA_FILENAME).read_text()))
    return model, meta


def make_metadata(
    cfg: Config, train_df: pd.DataFrame, best_iter: int | None, metrics: dict[str, float]
) -> ModelMetadata:
    return ModelMetadata(
        model_version=time.strftime("%Y%m%d-%H%M%S"),
        trained_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        package_version=__version__,
        features=cfg.features.all,
        numeric_features=cfg.features.numeric,
        categorical_features=cfg.features.categorical,
        decision_threshold=cfg.evaluation.decision_threshold,
        train_rows=len(train_df),
        best_iteration=best_iter,
        metrics=metrics,
        reference_profile=build_reference_profile(train_df, cfg),
    )
