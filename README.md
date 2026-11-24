# churn-mlops-vertex

End-to-end churn prediction on Google Cloud: a Vertex AI pipeline that trains and
gates an XGBoost model on telecom subscription data in BigQuery, a FastAPI service
on Cloud Run that serves it, and Vertex AI Model Monitoring watching for drift and
training/serving skew.

Data is the public [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d)
dataset (7,043 customers, ~26.5% churn).

```
          BigQuery                     Vertex AI Pipelines                       Model Registry
  ┌─────────────────────┐   ┌───────────────────────────────────────────┐   ┌──────────────────┐
  │ telco_customers_raw │──▶│ extract ─▶ train ─▶ evaluate ─▶ [gate] ──▶│──▶│ v1 v2 v3 ...     │
  │ training_baseline ◀─┼───│    │                              register │   │ @challenger      │
  └─────────────────────┘   └───────────────────────────────────────────┘   │ @champion        │
            ▲                                                               └────────┬─────────┘
            │                                                                        │
  ┌─────────┴──────────┐        ┌──────────────────────────┐                         │
  │   prediction_log   │◀───────│ Cloud Run: FastAPI       │◀── MODEL_URI (champion) ┘
  └─────────┬──────────┘        │ /predict /health         │
            │                   └──────────────────────────┘
            ▼
  Vertex AI Model Monitoring (daily) ──▶ alerts ──▶ docs/runbook.md
```

## What's in here

| Path | What it does |
|---|---|
| `src/churn/` | The actual ML code: cleaning, data contract checks, features, training, evaluation gate, registry helpers |
| `src/churn/serve/` | FastAPI app, request schema, async BigQuery prediction logging |
| `pipelines/` | KFP v2 components and the Vertex pipeline definition |
| `monitoring/` | Creates the Model Monitor + daily schedule |
| `sql/` | Table DDL, feature view (mirrors the Python features), incident queries |
| `docker/` | Serving and pipeline images |
| `.github/workflows/` | `ci.yml` on every PR, `cd.yml` for train → gate → deploy |
| `docs/runbook.md` | What to do when an alert fires, including rollback |

## Design notes

**One code path.** Pipeline components are thin wrappers around `churn.*`
functions. The same code runs in the local CLI, in pytest and on Vertex, so a bug
can be reproduced without spinning up a pipeline.

**One serialized object.** Preprocessing and the XGBoost model are saved as a
single sklearn `Pipeline`. Serving can't apply a different encoding than training did.

**Promotion is gated twice.** The pipeline only registers a model that clears the
absolute thresholds in `configs/config.yaml` *and* doesn't regress ROC-AUC by more
than 0.01 against the current champion. Then CD deploys it to a Cloud Run tag with
0% traffic, smoke-tests it, shifts traffic, and only then moves the `champion` alias.

**Monitoring without a Vertex endpoint.** Serving is on Cloud Run because it's
cheaper for this traffic profile and scales to zero in dev. Vertex Model Monitoring
v2 accepts BigQuery inputs, so the service writes each scored request to
`prediction_log` (batched, off the request path), and the monitor compares that
against the exact rows the model was trained on (`training_baseline`, written by the
pipeline's extract step).

**Threshold choice.** `scale_pos_weight` pushes probabilities up, so 0.5 isn't a
natural cut-off. 0.55 gave the best F1 on the holdout; recall is what the retention
team cares about, so the gate also enforces a recall floor.

## Results

Holdout set (20%, stratified, 1,409 customers), production config:

| Metric | Value |
|---|---|
| ROC-AUC | 0.847 |
| PR-AUC | 0.657 |
| Recall @ 0.55 | 0.75 |
| Precision @ 0.55 | 0.56 |
| Brier | 0.165 |

Roughly in line with published results on this dataset. Anything above ~0.90 ROC-AUC
on it usually means leakage, and there's a test that fails if that happens.

## Running locally

Python 3.11+.

```bash
make install          # pip install -e ".[dev,serve,gcp]"
make data             # downloads the CSV into data/
make train            # trains, evaluates, applies the gate, writes artifacts/model
make serve            # uvicorn on :8080 using artifacts/model
make smoke            # hits the local server with scripts/sample_request.json
make test             # unit + API tests, pipeline compile check
```

```bash
curl -s localhost:8080/predict -H 'Content-Type: application/json' \
  -d @scripts/sample_request.json | jq
```

## Deploying to GCP

One-time setup (assumes `gcloud` is authenticated and the project has Vertex AI,
BigQuery, Cloud Run and Artifact Registry APIs enabled):

```bash
export GCP_PROJECT=your-project
gsutil mb -l us-central1 gs://${GCP_PROJECT}-churn
gcloud artifacts repositories create churn --repository-format=docker --location=us-central1
./scripts/load_to_bigquery.sh
```

GitHub secrets used by `cd.yml`:

| Secret | |
|---|---|
| `GCP_PROJECT` | project id |
| `WIF_PROVIDER` | Workload Identity Federation provider (no JSON keys) |
| `DEPLOY_SA` | SA used by Actions: Artifact Registry writer, Vertex user, Run admin |
| `PIPELINE_SA` | SA the pipeline runs as: BigQuery, GCS, Vertex |
| `SERVING_SA` | SA for Cloud Run: GCS read on the model bucket, BigQuery insert on `prediction_log` |

Then run the **cd** workflow from the Actions tab. After the first champion exists:

```bash
python monitoring/setup_monitoring.py --project $GCP_PROJECT
```

To submit the pipeline by hand instead of through Actions:

```bash
PIPELINE_IMAGE=us-central1-docker.pkg.dev/$GCP_PROJECT/churn/pipeline:<tag> \
  python -m pipelines.churn_pipeline --project $GCP_PROJECT --git-sha $(git rev-parse HEAD) --wait
```

## Config

Everything tunable is in `configs/config.yaml`: feature lists, XGBoost params,
gate thresholds and drift thresholds. `${VAR}` placeholders are expanded from the
environment, and `CHURN_CONFIG` points at a different file if needed.

## Things I'd do next

- Pull ground-truth labels back into `prediction_log` once the churn window closes, so
  monitoring can track actual performance and not just drift
- Calibrate probabilities (isotonic on a held-out slice). The risk bands would
  then mean something in absolute terms
- Add per-segment metrics to the gate (contract type, senior citizens) so a model
  that's better on average but worse for one group doesn't slip through
- Move the derived features to a Vertex Feature Store view if more models start using them
