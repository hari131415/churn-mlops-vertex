from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

YesNo = Literal["Yes", "No"]
YesNoNoService = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    customer_id: str = Field(..., min_length=1, max_length=64)
    gender: Literal["Male", "Female"]
    senior_citizen: YesNo
    partner: YesNo
    dependents: YesNo
    tenure: int = Field(..., ge=0, le=600)
    phone_service: YesNo
    multiple_lines: Literal["Yes", "No", "No phone service"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: YesNoNoService
    online_backup: YesNoNoService
    device_protection: YesNoNoService
    tech_support: YesNoNoService
    streaming_tv: YesNoNoService
    streaming_movies: YesNoNoService
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: YesNo
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    monthly_charges: float = Field(..., ge=0, le=1000)
    total_charges: float = Field(..., ge=0)

    @model_validator(mode="after")
    def _charges_consistent(self) -> Customer:
        # Catches swapped fields from callers, which is the most common bad request.
        if (
            self.tenure > 0
            and self.total_charges > 0
            and self.total_charges < self.monthly_charges * 0.5
        ):
            raise ValueError("total_charges is implausibly low for the given tenure")
        return self


class PredictRequest(BaseModel):
    instances: list[Customer] = Field(..., min_length=1, max_length=1000)


class Prediction(BaseModel):
    customer_id: str
    churn_probability: float
    will_churn: bool
    risk_band: Literal["low", "medium", "high"]


class PredictResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: Literal["ok", "loading"]
    model_version: str | None = None
