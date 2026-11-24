"""KFP components for the churn pipeline.

Each component is a thin shell around functions in the `churn` package. The
heavy lifting lives in src/ so it can be unit tested without KFP and reused by
the local CLI. Components run in the pipeline image (docker/pipeline.Dockerfile),
which has the package installed.
"""

import os

from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

PIPELINE_IMAGE = os.environ.get(
    "PIPELINE_IMAGE", "us-central1-docker.pkg.dev/PROJECT/churn/pipeline:latest"
)


@dsl.component(base_image=PIPELINE_IMAGE)
def extract_data(
    project: str,
    source_table: str,
    baseline_table: str,
    train_data: Output[Dataset],
    test_data: Output[Dataset],
):
    """Pull the raw table, validate it, split, and snapshot the training rows.

    The snapshot table is what Vertex Model Monitoring uses as its training
    baseline for skew detection, so it has to hold exactly the rows we trained on.
    """
    import logging

    from google.cloud import bigquery

    from churn.config import get_config
    from churn.data import load_bigquery
    from churn.features import add_derived_features
    from churn.train import split

    logging.basicConfig(level=logging.INFO)
    cfg = get_config()

    df = load_bigquery(project, source_table)
    train_df, test_df = split(df, cfg)
    train_df.to_csv(train_data.path, index=False)
    test_df.to_csv(test_data.path, index=False)

    baseline = add_derived_features(train_df)[[cfg.id_column, *cfg.features.all, cfg.label]]
    client = bigquery.Client(project=project)
    job = client.load_table_from_dataframe(
        baseline,
        baseline_table,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
    )
    job.result()

    train_data.metadata["rows"] = len(train_df)
    test_data.metadata["rows"] = len(test_df)
    train_data.metadata["churn_rate"] = float(train_df[cfg.label].mean())


@dsl.component(base_image=PIPELINE_IMAGE)
def train_model(train_data: Input[Dataset], model: Output[Model]):
    import json
    import logging

    import pandas as pd

    from churn.config import get_config
    from churn.train import make_metadata, save, train

    logging.basicConfig(level=logging.INFO)
    cfg = get_config()

    train_df = pd.read_csv(train_data.path)
    fitted, best_iter = train(train_df, cfg)

    # Metrics are filled in by the evaluate step; we write metadata now so the
    # artifact is self-contained even if evaluation is re-run separately.
    meta = make_metadata(cfg, train_df, best_iter, metrics={})
    save(fitted, meta, model.path)
    model.metadata["framework"] = "xgboost"
    model.metadata["best_iteration"] = best_iter
    model.metadata["params"] = json.dumps(cfg.model)


@dsl.component(base_image=PIPELINE_IMAGE)
def evaluate_model(
    project: str,
    region: str,
    model_display_name: str,
    model: Input[Model],
    test_data: Input[Dataset],
    metrics: Output[Metrics],
) -> str:
    """Score the holdout, compare to the champion, and return 'pass' or 'fail'."""
    import json
    import logging
    from dataclasses import asdict
    from pathlib import Path

    import pandas as pd

    from churn.config import get_config
    from churn.evaluate import check_gate, evaluate
    from churn.registry import get_champion_metrics
    from churn.train import METADATA_FILENAME, load

    logging.basicConfig(level=logging.INFO)
    cfg = get_config()

    fitted, meta = load(model.path)
    results = evaluate(fitted, pd.read_csv(test_data.path), cfg)
    for k, v in results.items():
        metrics.log_metric(k, v)

    champion = get_champion_metrics(project, region, model_display_name)
    gate = check_gate(results, cfg.evaluation, champion)
    metrics.log_metric("gate_passed", int(gate.passed))
    logging.info("gate: %s (champion=%s)", gate.summary(), champion)

    # Persist final metrics alongside the model so serving can expose them.
    meta.metrics = results
    Path(model.path, METADATA_FILENAME).write_text(json.dumps(asdict(meta), indent=2))
    return "pass" if gate.passed else "fail"


@dsl.component(base_image=PIPELINE_IMAGE)
def register_model(
    project: str,
    region: str,
    model_display_name: str,
    git_sha: str,
    model: Input[Model],
) -> str:
    import json
    import logging
    from pathlib import Path

    from churn.registry import upload_version
    from churn.train import METADATA_FILENAME

    logging.basicConfig(level=logging.INFO)
    meta = json.loads(Path(model.path, METADATA_FILENAME).read_text())
    return upload_version(
        project=project,
        region=region,
        display_name=model_display_name,
        artifact_uri=model.uri,
        metrics=meta["metrics"],
        git_sha=git_sha,
    )
