"""
FastAPI inference endpoint.

POST /predict with a JSON body describing one store-day, return the LightGBM
point forecast for `Sales`. The entity-embedding model can be swapped in via
`MODEL_CHOICE=emb` env var.
"""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rossmann_forecast.config import Settings
from rossmann_forecast.features.engineer import (
    CATEGORICAL_COLS,
    CONTINUOUS_COLS,
    _encode_category,
    _parse_promo_interval,
)


class StoreDay(BaseModel):
    Store: int = Field(..., ge=1, le=1115)
    Date: date
    Promo: int = Field(..., ge=0, le=1)
    StateHoliday: str = "0"
    SchoolHoliday: int = Field(0, ge=0, le=1)
    StoreType: Literal["a", "b", "c", "d"] = "a"
    Assortment: Literal["a", "b", "c"] = "a"
    CompetitionDistance: float = 5000.0
    CompetitionOpenSinceMonth: int | None = None
    CompetitionOpenSinceYear: int | None = None
    Promo2: int = Field(0, ge=0, le=1)
    PromoInterval: str | None = None


class Prediction(BaseModel):
    sales: float
    model_used: str


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def _model_choice() -> str:
    return os.getenv("MODEL_CHOICE", "gbm").lower()


@lru_cache(maxsize=1)
def _gbm_booster():
    path = _settings().artifacts_root / "lightgbm.joblib"
    if not path.exists():
        raise RuntimeError("lightgbm.joblib not found. Run `forecast train-gbm` first.")
    return joblib.load(path)


def _row_to_features(row: StoreDay) -> pd.DataFrame:
    d = pd.DataFrame([{
        "Store": row.Store,
        "Date": pd.Timestamp(row.Date),
        "DayOfWeek": pd.Timestamp(row.Date).dayofweek + 1,
        "Promo": row.Promo,
        "StateHoliday": row.StateHoliday,
        "SchoolHoliday": row.SchoolHoliday,
        "StoreType": row.StoreType,
        "Assortment": row.Assortment,
        "CompetitionDistance": row.CompetitionDistance,
        "CompetitionOpenSinceMonth": row.CompetitionOpenSinceMonth,
        "CompetitionOpenSinceYear": row.CompetitionOpenSinceYear,
        "Promo2": row.Promo2,
        "PromoInterval": row.PromoInterval or "None",
    }])

    d["Day"] = d["Date"].dt.day.astype("int8")
    d["Month"] = d["Date"].dt.month.astype("int8")
    d["Year"] = d["Date"].dt.year.astype("int16")
    d["InPromoIntervalMonth"] = (
        _parse_promo_interval(d).where(d["Promo2"] == 1, 0).astype("int8")
    )

    comp_year = d["CompetitionOpenSinceYear"].astype("float32")
    comp_month = d["CompetitionOpenSinceMonth"].astype("float32")
    current_months = d["Year"].astype("float32") * 12 + d["Month"].astype("float32")
    open_months = comp_year * 12 + comp_month
    d["CompetitionOpenMonths"] = (current_months - open_months).clip(lower=0, upper=120).fillna(0)

    for col in CATEGORICAL_COLS:
        src = d["PromoInterval"] if col == "PromoIntervalCode" else d[col]
        codes, _ = _encode_category(src)
        d[col] = codes

    return d[CATEGORICAL_COLS + CONTINUOUS_COLS]


app = FastAPI(title="Rossmann Forecast API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": _model_choice()}


@app.post("/predict", response_model=Prediction)
def predict(row: StoreDay) -> Prediction:
    choice = _model_choice()
    if choice != "gbm":
        raise HTTPException(status_code=501, detail=f"MODEL_CHOICE={choice!r} not wired yet.")
    booster = _gbm_booster()
    features = _row_to_features(row)
    y_pred_log = booster.predict(features, num_iteration=booster.best_iteration)
    y_pred = float(np.expm1(y_pred_log)[0])
    return Prediction(sales=max(0.0, y_pred), model_used="lightgbm")
