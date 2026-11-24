-- Queries referenced from docs/runbook.md. Replace the dates as needed.

-- 1. Score distribution by day and model version. A step change on a deploy
--    day points at the model; a step change on a non-deploy day points at data.
SELECT
  DATE(prediction_ts)                         AS day,
  model_version,
  COUNT(*)                                    AS n,
  ROUND(AVG(churn_probability), 4)            AS mean_score,
  ROUND(COUNTIF(churn_probability >= 0.55) / COUNT(*), 4) AS flagged_rate,
  APPROX_QUANTILES(churn_probability, 4)      AS quartiles
FROM churn.prediction_log
WHERE prediction_ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 14 DAY)
GROUP BY day, model_version
ORDER BY day DESC, model_version;

-- 2. Categorical mix today vs. training baseline. Swap `contract` for the
--    feature Vertex Model Monitoring flagged.

WITH base AS (
  SELECT contract AS v, COUNT(*) / SUM(COUNT(*)) OVER () AS p
  FROM churn.training_baseline GROUP BY v
),
live AS (
  SELECT contract AS v, COUNT(*) / SUM(COUNT(*)) OVER () AS p
  FROM churn.prediction_log
  WHERE DATE(prediction_ts) = CURRENT_DATE()
  GROUP BY v
)
SELECT
  COALESCE(base.v, live.v) AS value,
  ROUND(base.p, 4) AS baseline_share,
  ROUND(live.p, 4) AS live_share,
  ROUND(IFNULL(live.p, 0) - IFNULL(base.p, 0), 4) AS delta
FROM base FULL OUTER JOIN live USING (v)
ORDER BY ABS(delta) DESC;

-- 3. Null / out-of-range checks on the last day of traffic. These should all be 0.
SELECT
  COUNTIF(total_charges = 0 AND tenure > 1)          AS zero_total_charges,
  COUNTIF(monthly_charges > 200)                      AS extreme_monthly,
  COUNTIF(internet_service = 'No' AND streaming_tv = 'Yes') AS inconsistent_addons
FROM churn.prediction_log
WHERE prediction_ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY);
