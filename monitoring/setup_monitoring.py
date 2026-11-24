"""Create (or recreate) the Vertex AI Model Monitor for the churn model.

The model is served from Cloud Run, not a Vertex endpoint, so we use Model
Monitoring v2 with BigQuery inputs:

  baseline : <dataset>.training_baseline   (written by the pipeline's extract step)
  target   : <dataset>.prediction_log      (written by the serving app)

Two objectives:
  - feature skew/drift on the raw request features vs. training
  - prediction output drift on churn_probability

    python monitoring/setup_monitoring.py --project my-proj
"""

from __future__ import annotations

import argparse
import logging

from churn.config import get_config
from churn.data import REQUIRED_COLUMNS

log = logging.getLogger(__name__)

NUMERIC_INPUTS = {"tenure", "monthly_charges", "total_charges"}


def monitored_features() -> list[str]:
    # Only raw request fields: those exist in both the baseline and the
    # prediction log. Derived features are deterministic functions of these.
    return sorted(REQUIRED_COLUMNS - {"customer_id"})


def build_schema(ml_monitoring):
    spec = ml_monitoring.spec
    fields = [
        spec.FieldSchema(name=f, data_type="float" if f in NUMERIC_INPUTS else "categorical")
        for f in monitored_features()
    ]
    return spec.ModelMonitoringSchema(
        feature_fields=fields,
        prediction_fields=[spec.FieldSchema(name="churn_probability", data_type="float")],
        ground_truth_fields=[spec.FieldSchema(name="churned", data_type="categorical")],
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--project")
    parser.add_argument("--model-version", default=None, help="defaults to the champion")
    parser.add_argument("--run-now", action="store_true", help="also trigger a one-off job")
    args = parser.parse_args()

    import vertexai
    from google.cloud import aiplatform
    from vertexai.resources.preview import ml_monitoring

    cfg = get_config()
    gcp, mon = cfg.gcp, cfg.monitoring
    project = args.project or gcp["project"]
    dataset = f"bq://{project}.{gcp['bq_dataset']}"
    vertexai.init(project=project, location=gcp["region"])

    parent = aiplatform.Model.list(filter=f'display_name="{gcp["model_display_name"]}"')[0]
    model = aiplatform.Model(model_name=f"{parent.resource_name}@champion")
    version = args.model_version or model.version_id

    spec = ml_monitoring.spec
    objective = spec.TabularObjective(
        feature_drift_spec=spec.DataDriftSpec(
            default_numeric_alert_threshold=mon["numeric_drift_threshold"],
            default_categorical_alert_threshold=mon["categorical_drift_threshold"],
        ),
        prediction_output_drift_spec=spec.DataDriftSpec(
            default_numeric_alert_threshold=mon["prediction_drift_threshold"],
        ),
    )
    notification = spec.NotificationSpec(
        user_emails=mon.get("alert_emails") or None,
        enable_cloud_logging=True,
    )
    output = spec.OutputSpec(gcs_base_dir=f"{gcp['pipeline_root']}/monitoring")

    monitor = ml_monitoring.ModelMonitor.create(
        display_name=f"{gcp['model_display_name']}-monitor",
        model_name=parent.resource_name,
        model_version_id=version,
        training_dataset=spec.MonitoringInput(table_uri=f"{dataset}.training_baseline"),
        model_monitoring_schema=build_schema(ml_monitoring),
        tabular_objective_spec=objective,
        output_spec=output,
        notification_spec=notification,
    )
    log.info("created monitor %s", monitor.resource_name)

    target = spec.MonitoringInput(
        table_uri=f"{dataset}.{gcp['prediction_log_table']}",
        timestamp_field="prediction_ts",
        window=mon["window"],
    )
    schedule = monitor.create_schedule(
        cron=mon["schedule_cron"],
        target_dataset=target,
        display_name=f"{gcp['model_display_name']}-daily",
    )
    log.info("scheduled %s (%s)", schedule.name, mon["schedule_cron"])

    if args.run_now:
        job = monitor.run(target_dataset=target, display_name="adhoc-check")
        log.info("started ad-hoc monitoring job %s", job.resource_name)


if __name__ == "__main__":
    main()
