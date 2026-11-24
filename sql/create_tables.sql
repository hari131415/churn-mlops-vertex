-- One-time setup. Run with:
--   bq query --use_legacy_sql=false --project_id=$GCP_PROJECT < sql/create_tables.sql
-- after loading the raw CSV with scripts/load_to_bigquery.sh.

CREATE SCHEMA IF NOT EXISTS churn OPTIONS (location = "us-central1");

-- Serving writes one row per scored customer. Partitioned by day so the
-- monitoring jobs and ad-hoc incident queries only scan the window they need.
CREATE TABLE IF NOT EXISTS churn.prediction_log (
  customer_id        STRING NOT NULL,
  gender             STRING,
  senior_citizen     STRING,
  partner            STRING,
  dependents         STRING,
  tenure             INT64,
  phone_service      STRING,
  multiple_lines     STRING,
  internet_service   STRING,
  online_security    STRING,
  online_backup      STRING,
  device_protection  STRING,
  tech_support       STRING,
  streaming_tv       STRING,
  streaming_movies   STRING,
  contract           STRING,
  paperless_billing  STRING,
  payment_method     STRING,
  monthly_charges    FLOAT64,
  total_charges      FLOAT64,
  churn_probability  FLOAT64 NOT NULL,
  model_version      STRING NOT NULL,
  prediction_ts      TIMESTAMP NOT NULL
)
PARTITION BY DATE(prediction_ts)
OPTIONS (partition_expiration_days = 400);
