"""Config loading.

The YAML file is the single source of truth. `${VAR}` placeholders are expanded
from the environment so the same file works locally, in CI and inside pipeline
containers.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


@dataclass(frozen=True)
class FeatureConfig:
    numeric: list[str]
    categorical: list[str]

    @property
    def all(self) -> list[str]:
        return [*self.numeric, *self.categorical]


@dataclass(frozen=True)
class EvalConfig:
    min_roc_auc: float
    min_pr_auc: float
    min_recall_at_threshold: float
    max_auc_regression: float
    decision_threshold: float


@dataclass(frozen=True)
class Config:
    gcp: dict[str, Any]
    data: dict[str, Any]
    features: FeatureConfig
    model: dict[str, Any]
    evaluation: EvalConfig
    monitoring: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.data["label_column"]

    @property
    def id_column(self) -> str:
        return self.data["id_column"]


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path or os.environ.get("CHURN_CONFIG", DEFAULT_CONFIG_PATH))
    with open(path) as f:
        raw = _expand_env(yaml.safe_load(f))

    return Config(
        gcp=raw["gcp"],
        data=raw["data"],
        features=FeatureConfig(**raw["features"]),
        model=raw["model"],
        evaluation=EvalConfig(**raw["evaluation"]),
        monitoring=raw.get("monitoring", {}),
    )


@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()
