"""Vertex AI pipeline definition.

Compile only:
    python -m pipelines.churn_pipeline --compile-only

Compile and submit:
    python -m pipelines.churn_pipeline --project my-proj --git-sha $(git rev-parse HEAD)
"""

import argparse
import logging

from kfp import compiler, dsl

from pipelines.components import evaluate_model, extract_data, register_model, train_model

PIPELINE_NAME = "telco-churn-training"
PIPELINE_SPEC = "churn_pipeline.yaml"


@dsl.pipeline(name=PIPELINE_NAME, description="Feature prep, train, evaluate, gated register.")
def churn_pipeline(
    project: str,
    region: str = "us-central1",
    source_table: str = "",
    baseline_table: str = "",
    model_display_name: str = "telco-churn-xgb",
    git_sha: str = "unknown",
):
    data = extract_data(
        project=project, source_table=source_table, baseline_table=baseline_table
    ).set_display_name("extract-and-split")

    trained = (
        train_model(train_data=data.outputs["train_data"])
        .set_display_name("train-xgboost")
        .set_cpu_limit("4")
        .set_memory_limit("8G")
    )

    evaluation = evaluate_model(
        project=project,
        region=region,
        model_display_name=model_display_name,
        model=trained.outputs["model"],
        test_data=data.outputs["test_data"],
    ).set_display_name("evaluate-and-gate")

    with dsl.If(evaluation.outputs["Output"] == "pass", name="gate-passed"):
        register_model(
            project=project,
            region=region,
            model_display_name=model_display_name,
            git_sha=git_sha,
            model=trained.outputs["model"],
        ).set_display_name("register-challenger")


def compile_pipeline(path: str = PIPELINE_SPEC) -> str:
    compiler.Compiler().compile(churn_pipeline, package_path=path)
    return path


def submit(args: argparse.Namespace) -> None:
    from google.cloud import aiplatform

    from churn.config import get_config

    cfg = get_config()
    gcp = cfg.gcp
    project = args.project or gcp["project"]
    dataset = f"{project}.{gcp['bq_dataset']}"

    aiplatform.init(project=project, location=gcp["region"], staging_bucket=gcp["pipeline_root"])
    job = aiplatform.PipelineJob(
        display_name=PIPELINE_NAME,
        template_path=compile_pipeline(),
        pipeline_root=gcp["pipeline_root"],
        parameter_values={
            "project": project,
            "region": gcp["region"],
            "source_table": f"{dataset}.{gcp['raw_table']}",
            "baseline_table": f"{dataset}.training_baseline",
            "model_display_name": gcp["model_display_name"],
            "git_sha": args.git_sha,
        },
        enable_caching=not args.no_cache,
    )
    job.submit(service_account=args.service_account)
    if args.wait:
        job.wait()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--project")
    p.add_argument("--git-sha", default="unknown")
    p.add_argument("--service-account")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--wait", action="store_true")
    args = p.parse_args()

    if args.compile_only:
        print(compile_pipeline())
    else:
        submit(args)


if __name__ == "__main__":
    main()
