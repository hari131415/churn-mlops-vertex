-- Cleaned + derived features over the raw table. Mirrors churn.data.clean and
-- churn.features.add_derived_features; keep them in sync.
-- Handy for analysis and for scoring in bulk from BigQuery.

CREATE OR REPLACE VIEW churn.telco_customers_features AS
WITH cleaned AS (
  SELECT
    customerID                                   AS customer_id,
    gender,
    IF(SeniorCitizen = 1, 'Yes', 'No')           AS senior_citizen,
    Partner                                      AS partner,
    Dependents                                   AS dependents,
    tenure,
    PhoneService                                 AS phone_service,
    MultipleLines                                AS multiple_lines,
    InternetService                              AS internet_service,
    OnlineSecurity                               AS online_security,
    OnlineBackup                                 AS online_backup,
    DeviceProtection                             AS device_protection,
    TechSupport                                  AS tech_support,
    StreamingTV                                  AS streaming_tv,
    StreamingMovies                              AS streaming_movies,
    Contract                                     AS contract,
    PaperlessBilling                             AS paperless_billing,
    PaymentMethod                                AS payment_method,
    MonthlyCharges                               AS monthly_charges,
    IFNULL(SAFE_CAST(TotalCharges AS FLOAT64), 0) AS total_charges,
    IF(Churn = 'Yes', 1, 0)                      AS churned
  FROM churn.telco_customers_raw
)
SELECT
  *,
  IF(tenure > 0, total_charges / tenure, monthly_charges) AS avg_monthly_spend,
  (
    IF(online_security = 'Yes', 1, 0) + IF(online_backup = 'Yes', 1, 0)
    + IF(device_protection = 'Yes', 1, 0) + IF(tech_support = 'Yes', 1, 0)
    + IF(streaming_tv = 'Yes', 1, 0) + IF(streaming_movies = 'Yes', 1, 0)
  ) AS num_addon_services,
  CASE
    WHEN tenure <= 6  THEN '0-6m'
    WHEN tenure <= 12 THEN '7-12m'
    WHEN tenure <= 24 THEN '13-24m'
    WHEN tenure <= 48 THEN '25-48m'
    ELSE '49m+'
  END AS tenure_bucket
FROM cleaned;
