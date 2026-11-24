"""Thin wrapper around Vertex AI Model Registry.

Versioning convention:
  - every model that passes the gate is uploaded as a new version of
    `model_display_name`, tagged with the alias `challenger`
  - the CD workflow moves the `champion` alias after the Cloud Run rollout
    is healthy, and Cloud Run always serves whatever `champion` points at
  - rollback = move `champion` back to the previous version and redeploy
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

CHAMPION = "champion"
CHALLENGER = "challenger"

# Prebuilt container is only used for metadata/registry purposes; we serve
# from our own image on Cloud Run.
SERVING_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.2-1:latest"


def _init(project: str, region: str):
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)
    return aiplatform


def find_parent_model(project: str, region: str, display_name: str) -> str | None:
    aiplatform = _init(project, region)
    models = aiplatform.Model.list(filter=f'display_name="{display_name}"', order_by="create_time")
    return models[0].resource_name if models else None


def get_champion_metrics(project: str, region: str, display_name: str) -> dict | None:
    """Metrics are stored as model labels at upload time; returns None if no champion yet."""
    aiplatform = _init(project, region)
    parent = find_parent_model(project, region, display_name)
    if parent is None:
        return None
    try:
        champ = aiplatform.Model(model_name=f"{parent}@{CHAMPION}")
    except Exception:  # noqa: BLE001 - not found surfaces as a few different errors
        log.info("no champion alias found for %s", display_name)
        return None
    labels = champ.labels or {}
    if "roc_auc" not in labels:
        return None
    # Labels can't hold dots, so we store e.g. roc_auc=0_8471.
    return {"roc_auc": float(labels["roc_auc"].replace("_", "."))}


def upload_version(
    project: str,
    region: str,
    display_name: str,
    artifact_uri: str,
    metrics: dict[str, float],
    git_sha: str = "unknown",
) -> str:
    aiplatform = _init(project, region)
    parent = find_parent_model(project, region, display_name)

    labels = {
        "roc_auc": f"{metrics['roc_auc']:.4f}".replace(".", "_"),
        "pr_auc": f"{metrics['pr_auc']:.4f}".replace(".", "_"),
        "git_sha": git_sha[:12].lower(),
    }
    model = aiplatform.Model.upload(
        display_name=display_name,
        parent_model=parent,
        artifact_uri=artifact_uri,
        serving_container_image_uri=SERVING_IMAGE,
        version_aliases=[CHALLENGER],
        is_default_version=parent is None,
        labels=labels,
        description=f"roc_auc={metrics['roc_auc']:.4f} pr_auc={metrics['pr_auc']:.4f}",
    )
    log.info("uploaded %s version %s", model.resource_name, model.version_id)
    return f"{model.resource_name}@{model.version_id}"


def promote(project: str, region: str, display_name: str, version_id: str) -> None:
    aiplatform = _init(project, region)
    parent = find_parent_model(project, region, display_name)
    registry = aiplatform.models.ModelRegistry(parent)
    registry.add_version_aliases([CHAMPION], version=version_id)
    log.info("promoted %s v%s to %s", display_name, version_id, CHAMPION)


def artifact_uri_for(project: str, region: str, display_name: str, alias: str = CHAMPION) -> str:
    aiplatform = _init(project, region)
    parent = find_parent_model(project, region, display_name)
    if parent is None:
        raise LookupError(f"no model named {display_name}")
    model = aiplatform.Model(model_name=f"{parent}@{alias}")
    return model.uri
