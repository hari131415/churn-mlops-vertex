#!/usr/bin/env bash
# Load the raw CSV into BigQuery and create the supporting tables/views.
#   GCP_PROJECT=my-proj ./scripts/load_to_bigquery.sh
set -euo pipefail

: "${GCP_PROJECT:?set GCP_PROJECT}"
DATASET="${BQ_DATASET:-churn}"
REGION="${REGION:-us-central1}"
CSV="${1:-data/telco_churn.csv}"

./scripts/download_data.sh "$CSV"

bq --project_id="$GCP_PROJECT" --location="$REGION" mk -f --dataset "$GCP_PROJECT:$DATASET" >/dev/null

# TotalCharges has blanks for new customers, so load it as STRING and cast in SQL.
bq --project_id="$GCP_PROJECT" load --replace --skip_leading_rows=1 --source_format=CSV \
  "$DATASET.telco_customers_raw" "$CSV" \
  customerID:STRING,gender:STRING,SeniorCitizen:INT64,Partner:STRING,Dependents:STRING,tenure:INT64,PhoneService:STRING,MultipleLines:STRING,InternetService:STRING,OnlineSecurity:STRING,OnlineBackup:STRING,DeviceProtection:STRING,TechSupport:STRING,StreamingTV:STRING,StreamingMovies:STRING,Contract:STRING,PaperlessBilling:STRING,PaymentMethod:STRING,MonthlyCharges:FLOAT64,TotalCharges:STRING,Churn:STRING

bq --project_id="$GCP_PROJECT" query --use_legacy_sql=false < sql/create_tables.sql
bq --project_id="$GCP_PROJECT" query --use_legacy_sql=false < sql/feature_view.sql
echo "done: $GCP_PROJECT.$DATASET"
