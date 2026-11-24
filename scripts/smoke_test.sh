#!/usr/bin/env bash
# Basic post-deploy check: health is ok and a known request scores sensibly.
#   ./scripts/smoke_test.sh https://candidate---churn-service-xxxx.a.run.app
set -euo pipefail

URL="${1:?usage: smoke_test.sh <base-url>}"
AUTH=()
if command -v gcloud >/dev/null && [[ "$URL" == https://* ]]; then
  AUTH=(-H "Authorization: Bearer $(gcloud auth print-identity-token)")
fi

echo "GET $URL/health"
curl -fsS "${AUTH[@]}" "$URL/health" | tee /dev/stderr | grep -q '"status":"ok"'

echo "POST $URL/predict"
resp=$(curl -fsS "${AUTH[@]}" -H "Content-Type: application/json" \
  -d @scripts/sample_request.json "$URL/predict")
echo "$resp"

# The month-to-month fiber customer in the sample should score higher than
# the two-year DSL one. If not, something is badly wrong with the artifact.
python3 - "$resp" <<'EOF'
import json, sys
preds = {p["customer_id"]: p["churn_probability"] for p in json.loads(sys.argv[1])["predictions"]}
assert preds["smoke-high"] > preds["smoke-low"], preds
print("ok")
EOF
