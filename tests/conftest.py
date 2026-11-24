from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn.config import load_config
from churn.data import clean
from churn.train import make_metadata, save, split, train

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_DATA = REPO_ROOT / "data" / "telco_churn.csv"


def make_raw_frame(n: int = 1500, seed: int = 7) -> pd.DataFrame:
    """Synthetic rows in the raw CSV format with a believable churn signal.

    Month-to-month contracts, fiber, electronic check and short tenure push
    churn up, which is roughly what the real data looks like.
    """
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    payment = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        n,
    )
    tenure = rng.integers(0, 73, n)
    monthly = np.round(
        np.where(internet == "Fiber optic", 85, np.where(internet == "DSL", 55, 22))
        + rng.normal(0, 8, n),
        2,
    ).clip(18)
    total = np.where(tenure == 0, " ", np.round(monthly * tenure * rng.uniform(0.95, 1.05, n), 2))

    def addon():
        vals = rng.choice(["Yes", "No"], n)
        return np.where(internet == "No", "No internet service", vals)

    logit = (
        -1.4
        + 2.0 * (contract == "Month-to-month")
        + 1.0 * (internet == "Fiber optic")
        + 0.8 * (payment == "Electronic check")
        - 0.05 * tenure
        + rng.normal(0, 0.3, n)
    )
    churn = rng.uniform(size=n) < 1 / (1 + np.exp(-logit))
    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])

    return pd.DataFrame(
        {
            "customerID": [f"{i:04d}-TEST" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": np.where(
                phone == "No", "No phone service", rng.choice(["Yes", "No"], n)
            ),
            "InternetService": internet,
            "OnlineSecurity": addon(),
            "OnlineBackup": addon(),
            "DeviceProtection": addon(),
            "TechSupport": addon(),
            "StreamingTV": addon(),
            "StreamingMovies": addon(),
            "Contract": contract,
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": np.where(churn, "Yes", "No"),
        }
    )


@pytest.fixture(scope="session")
def cfg():
    base = load_config(REPO_ROOT / "configs" / "config.yaml")
    # Smaller model so the suite stays fast; the gate is tested separately.
    return replace(base, model={**base.model, "n_estimators": 120})


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return make_raw_frame()


@pytest.fixture(scope="session")
def clean_df(raw_df) -> pd.DataFrame:
    return clean(raw_df)


@pytest.fixture(scope="session")
def trained(clean_df, cfg):
    train_df, test_df = split(clean_df, cfg)
    model, best_iter = train(train_df, cfg)
    return model, best_iter, train_df, test_df


@pytest.fixture(scope="session")
def model_dir(trained, cfg, tmp_path_factory) -> Path:
    model, best_iter, train_df, _ = trained
    out = tmp_path_factory.mktemp("model")
    save(model, make_metadata(cfg, train_df, best_iter, {"roc_auc": 0.8}), out)
    return out


@pytest.fixture
def sample_customer() -> dict:
    return {
        "customer_id": "c-1",
        "gender": "Female",
        "senior_citizen": "No",
        "partner": "Yes",
        "dependents": "No",
        "tenure": 5,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 89.1,
        "total_charges": 440.0,
    }
