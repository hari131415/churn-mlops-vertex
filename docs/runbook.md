# Churn service runbook

Owner: ML platform. Alerts land via Vertex Model Monitoring email/Cloud Logging and
Cloud Run's default uptime/error alerts.

## Quick reference

| | |
|---|---|
| Service | Cloud Run `churn-service`, `us-central1` |
| Model | Vertex Model Registry `telco-churn-xgb`, alias `champion` |
| Prediction log | `churn.prediction_log` (partitioned by day) |
| Training baseline | `churn.training_baseline` (overwritten each training run) |
| Monitor | `telco-churn-xgb-monitor`, daily 06:00 UTC, 24h window |
| Incident SQL | `sql/incident_queries.sql` |

## 1. Feature drift / skew alert

Monitoring compares the last 24h of requests to the training rows.

1. **Which features, how much?** Open the monitoring job in the Vertex console
   (Model Registry → telco-churn-xgb → Monitor). Note the feature(s) and the distance.
2. **Is it real or a pipeline issue?** Run query 3 in `incident_queries.sql`. Nulls,
   zeros or impossible combinations usually mean an upstream caller changed, not
   the customer base. Check with the calling team before touching the model.
3. **Did scores move?** Run query 1. If `mean_score` and `flagged_rate` are stable,
   the drift isn't affecting decisions yet; log it and keep watching.
4. **Look at the mix.** Query 2 with the flagged feature. A real business shift
   (e.g. a promo moving people to month-to-month) is a retraining trigger, not a bug.
5. **Act:**
   - Bad input data → get the caller fixed. Consider tightening `serve/schemas.py`
     so the bad values get a 422 instead of a score.
   - Real shift, scores still sane → trigger the `cd` workflow to retrain on fresh data.
   - Real shift, scores clearly wrong → roll back (section 3) if a recent deploy
     is involved, otherwise retrain.

## 2. Prediction output drift alert

The score distribution moved even if no single feature crossed its threshold.

1. Query 1: compare by `model_version`. If the change lines up with a new version
   going live, that's the cause; roll back and look at the pipeline run that
   produced it.
2. If the version didn't change, treat it like section 1 and look for which
   features moved together.

## 3. Rollback

Cloud Run keeps prior revisions, so rollback is fast and doesn't need a rebuild.

```bash
# see revisions and which one has traffic
gcloud run revisions list --service churn-service --region us-central1

# send all traffic back to the previous revision
gcloud run services update-traffic churn-service --region us-central1 \
  --to-revisions <previous-revision>=100

# point the champion alias back so the next deploy doesn't undo it
GCP_PROJECT=... python scripts/promote.py --version <previous-version-id>
```

Then verify:

```bash
./scripts/smoke_test.sh $(gcloud run services describe churn-service \
  --region us-central1 --format 'value(status.url)')
```

and check `/metadata` reports the expected `model_version`.

## 4. Latency / 5xx

- `X-Process-Time-Ms` is on every response and request latency is logged as a
  structured field. Filter Cloud Logging on `jsonPayload.latency_ms > 300`.
- Large batches are the usual cause of slow requests: the API allows up to 1000
  instances per call. If one caller is sending huge batches, ask them to chunk.
- Cold starts: `--min-instances 1` is set in `cd.yml`. If someone reduced it to 0,
  that's why p95 jumped.
- 5xx with "prediction failed" in logs almost always means the artifact and the
  code are out of sync (e.g. a feature was added in code but the champion predates it).
  Roll back the revision.

## 5. Training pipeline failed

- `extract-and-split` fails with `DataValidationError`: the data contract caught
  something. The message says which check. Don't bypass it; fix the source table.
- `evaluate-and-gate` returned `fail`: that's the gate working. Metrics are in the
  pipeline run's Metrics artifact. Nothing is registered or deployed, and the
  current champion keeps serving.
- Infra errors (quota, permissions): rerun with `--no-cache` after fixing.
