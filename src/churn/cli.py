"""Local entrypoint: train, evaluate and gate in one go.

    churn-train --data data/telco_churn.csv --out artifacts/model

Same code path the Vertex pipeline components call, just without KFP.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from churn.config import load_config
from churn.data import load_csv
from churn.evaluate import check_gate, evaluate
from churn.train import make_metadata, save, split, train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="path to the raw telco CSV")
    parser.add_argument("--out", default="artifacts/model")
    parser.add_argument("--config", default=None)
    parser.add_argument("--champion-metrics", default=None, help="json file from a prior run")
    parser.add_argument("--no-gate", action="store_true", help="save even if the gate fails")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(args.config)

    df = load_csv(args.data)
    train_df, test_df = split(df, cfg)
    model, best_iter = train(train_df, cfg)
    metrics = evaluate(model, test_df, cfg)

    champion = None
    if args.champion_metrics:
        with open(args.champion_metrics) as f:
            champion = json.load(f)
    gate = check_gate(metrics, cfg.evaluation, champion)
    print(json.dumps({"metrics": metrics, "gate": gate.summary()}, indent=2))

    if not gate.passed and not args.no_gate:
        return 1
    save(model, make_metadata(cfg, train_df, best_iter, metrics), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
