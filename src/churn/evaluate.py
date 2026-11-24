"""Offline evaluation and the promotion gate."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from churn.config import Config, EvalConfig
from churn.features import add_derived_features

log = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "precision_at_threshold": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_threshold": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_at_threshold": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(y_pred.mean()),
        "n": int(len(y_true)),
    }


def evaluate(model: Pipeline, test_df: pd.DataFrame, cfg: Config) -> dict[str, float]:
    df = add_derived_features(test_df)
    probs = model.predict_proba(df[cfg.features.all])[:, 1]
    metrics = compute_metrics(df[cfg.label].to_numpy(), probs, cfg.evaluation.decision_threshold)
    log.info("eval: %s", {k: round(v, 4) for k, v in metrics.items()})
    return metrics


@dataclass
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return "PASS" if self.passed else "FAIL: " + "; ".join(self.reasons)


def check_gate(
    metrics: dict[str, float],
    gate: EvalConfig,
    champion_metrics: dict[str, float] | None = None,
) -> GateResult:
    """Absolute thresholds first, then a no-regression check against the champion."""
    reasons = []
    if metrics["roc_auc"] < gate.min_roc_auc:
        reasons.append(f"roc_auc {metrics['roc_auc']:.4f} < {gate.min_roc_auc}")
    if metrics["pr_auc"] < gate.min_pr_auc:
        reasons.append(f"pr_auc {metrics['pr_auc']:.4f} < {gate.min_pr_auc}")
    if metrics["recall_at_threshold"] < gate.min_recall_at_threshold:
        reasons.append(
            f"recall {metrics['recall_at_threshold']:.4f} < {gate.min_recall_at_threshold}"
        )

    if champion_metrics and "roc_auc" in champion_metrics:
        drop = champion_metrics["roc_auc"] - metrics["roc_auc"]
        if drop > gate.max_auc_regression:
            reasons.append(
                f"roc_auc regressed {drop:.4f} vs champion "
                f"({champion_metrics['roc_auc']:.4f} -> {metrics['roc_auc']:.4f})"
            )

    return GateResult(passed=not reasons, reasons=reasons)
