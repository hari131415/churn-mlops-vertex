"""Resolve and load the model the service should serve.

Resolution order:
  1. MODEL_DIR  - local directory (tests, docker run with a mounted volume)
  2. MODEL_URI  - gs:// prefix, downloaded once at startup (Cloud Run)
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from churn.train import METADATA_FILENAME, MODEL_FILENAME, ModelMetadata, load

log = logging.getLogger(__name__)


def _download_gcs(uri: str, dest: Path) -> Path:
    from google.cloud import storage

    bucket_name, _, prefix = uri.removeprefix("gs://").partition("/")
    bucket = storage.Client().bucket(bucket_name)
    for name in (MODEL_FILENAME, METADATA_FILENAME):
        blob = bucket.blob(f"{prefix.rstrip('/')}/{name}")
        blob.download_to_filename(dest / name)
    log.info("downloaded model from %s", uri)
    return dest


def resolve_model_dir() -> Path:
    if local := os.environ.get("MODEL_DIR"):
        return Path(local)
    if uri := os.environ.get("MODEL_URI"):
        return _download_gcs(uri, Path(tempfile.mkdtemp(prefix="model-")))
    raise RuntimeError("set MODEL_DIR or MODEL_URI")


def load_model():
    model, meta = load(resolve_model_dir())
    log.info("loaded model version %s", meta.model_version)
    return model, meta


__all__ = ["ModelMetadata", "load_model", "resolve_model_dir"]
