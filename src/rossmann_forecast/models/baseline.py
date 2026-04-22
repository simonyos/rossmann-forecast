"""
Naive baselines. The point is to anchor the story: if a learned model cannot beat
these, there is a feature-engineering or target-leakage problem upstream.

- `seasonal_naive`: for each (Store, DayOfWeek), predict the mean of the last N
   observed Sales values before the validation cutoff.
- `median_per_store`: the trivial per-store median.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from rossmann_forecast.config import Settings
from rossmann_forecast.features.engineer import TARGET_COL, load_bundle
from rossmann_forecast.metrics import mae, rmse, rmspe


@dataclass
class BaselineResult:
    name: str
    rmspe: float
    rmse: float
    mae: float
    train_seconds: float
    artifact_path: Path


def _seasonal_naive_predict(train: pd.DataFrame, valid: pd.DataFrame, last_n: int) -> np.ndarray:
    recent = (
        train.sort_values("Date")
        .groupby(["Store", "DayOfWeek"])
        .tail(last_n)
        .groupby(["Store", "DayOfWeek"])[TARGET_COL]
        .mean()
        .rename("pred")
        .reset_index()
    )
    joined = valid.merge(recent, on=["Store", "DayOfWeek"], how="left")
    fallback = float(train[TARGET_COL].median())
    return joined["pred"].fillna(fallback).to_numpy(dtype=np.float32)


def _median_per_store_predict(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    medians = train.groupby("Store")[TARGET_COL].median().rename("pred")
    joined = valid.merge(medians, on="Store", how="left")
    fallback = float(train[TARGET_COL].median())
    return joined["pred"].fillna(fallback).to_numpy(dtype=np.float32)


def _score(
    name: str, y_true: np.ndarray, y_pred: np.ndarray, train_seconds: float, artifacts_dir: Path
) -> BaselineResult:
    result = BaselineResult(
        name=name,
        rmspe=rmspe(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
        mae=mae(y_true, y_pred),
        train_seconds=train_seconds,
        artifact_path=artifacts_dir / f"{name}_predictions.npy",
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    np.save(result.artifact_path, y_pred)
    return result


def run(settings: Settings) -> list[BaselineResult]:
    bundle = load_bundle(settings)
    y_true = bundle.valid[TARGET_COL].to_numpy(dtype=np.float32)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("rossmann-forecast/baselines")

    results: list[BaselineResult] = []
    for name, predictor in (
        ("seasonal_naive_8", lambda: _seasonal_naive_predict(bundle.train, bundle.valid, 8)),
        ("seasonal_naive_4", lambda: _seasonal_naive_predict(bundle.train, bundle.valid, 4)),
        ("median_per_store", lambda: _median_per_store_predict(bundle.train, bundle.valid)),
    ):
        with mlflow.start_run(run_name=name):
            t0 = time.perf_counter()
            y_pred = predictor()
            dt = time.perf_counter() - t0
            r = _score(name, y_true, y_pred, dt, settings.artifacts_root)
            mlflow.log_metrics(
                {"rmspe": r.rmspe, "rmse": r.rmse, "mae": r.mae, "train_seconds": dt}
            )
            results.append(r)

    summary_path = settings.artifacts_root / "baseline_summary.json"
    summary_path.write_text(
        json.dumps(
            [
                {
                    "name": r.name,
                    "rmspe": r.rmspe,
                    "rmse": r.rmse,
                    "mae": r.mae,
                    "train_seconds": r.train_seconds,
                }
                for r in results
            ],
            indent=2,
        )
    )
    return results
