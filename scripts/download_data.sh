#!/usr/bin/env bash
# Public IBM Telco Customer Churn dataset (7,043 customers).
set -euo pipefail

URL="https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
OUT="${1:-data/telco_churn.csv}"

mkdir -p "$(dirname "$OUT")"
if [[ -s "$OUT" ]]; then
  echo "already have $OUT"
  exit 0
fi
curl -fsSL "$URL" -o "$OUT"
echo "wrote $(wc -l < "$OUT") lines to $OUT"
