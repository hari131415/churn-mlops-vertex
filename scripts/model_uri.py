"""Print the GCS artifact URI for a model alias (champion by default)."""

import argparse
import os

from churn.config import get_config
from churn.registry import CHAMPION, artifact_uri_for

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--alias", default=CHAMPION)
    args = p.parse_args()
    cfg = get_config().gcp
    print(
        artifact_uri_for(
            os.environ["GCP_PROJECT"], cfg["region"], cfg["model_display_name"], args.alias
        )
    )
