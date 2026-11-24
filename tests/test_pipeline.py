import json

import pytest
import yaml

pytest.importorskip("kfp")


def test_pipeline_compiles(tmp_path):
    from pipelines.churn_pipeline import compile_pipeline

    path = compile_pipeline(str(tmp_path / "spec.yaml"))
    spec = yaml.safe_load((tmp_path / "spec.yaml").read_text())
    assert path.endswith("spec.yaml")

    tasks = spec["root"]["dag"]["tasks"]
    assert {"extract-data", "train-model", "evaluate-model"} <= tasks.keys()

    # register_model must only run inside the gate condition
    condition = next(t for name, t in tasks.items() if name.startswith("condition"))
    assert "pass" in json.dumps(condition["triggerPolicy"])


def test_monitoring_schema_uses_only_logged_fields():
    from churn.data import REQUIRED_COLUMNS
    from monitoring.setup_monitoring import monitored_features

    feats = set(monitored_features())
    assert "customer_id" not in feats
    assert feats <= REQUIRED_COLUMNS
    # derived features aren't in the prediction log, so they can't be monitored there
    assert "tenure_bucket" not in feats
