"""Move the `champion` alias to a model version. Also used for rollback.

GCP_PROJECT=my-proj python scripts/promote.py --version 7
"""

import argparse
import os

from churn.config import get_config
from churn.registry import promote

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--version", required=True, help="Model Registry version id, e.g. 7")
    args = p.parse_args()
    cfg = get_config().gcp
    promote(os.environ["GCP_PROJECT"], cfg["region"], cfg["model_display_name"], args.version)
