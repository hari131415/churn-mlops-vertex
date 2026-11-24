"""Print GitHub Actions outputs describing whether this commit produced a challenger.

The pipeline only registers a model when the gate passes, so "registered" is
true iff the latest challenger was built from $GITHUB_SHA.
"""

import os

from google.cloud import aiplatform

from churn.config import get_config
from churn.registry import CHALLENGER, find_parent_model


def main() -> None:
    cfg = get_config().gcp
    aiplatform.init(project=os.environ["GCP_PROJECT"], location=cfg["region"])
    parent = find_parent_model(os.environ["GCP_PROJECT"], cfg["region"], cfg["model_display_name"])
    sha = os.environ.get("GITHUB_SHA", "")[:12].lower()

    if parent is None:
        print("registered=false")
        return
    try:
        model = aiplatform.Model(model_name=f"{parent}@{CHALLENGER}")
    except Exception:  # noqa: BLE001
        print("registered=false")
        return

    built_here = (model.labels or {}).get("git_sha") == sha
    print(f"registered={'true' if built_here else 'false'}")
    print(f"version={model.version_id}")


if __name__ == "__main__":
    main()
