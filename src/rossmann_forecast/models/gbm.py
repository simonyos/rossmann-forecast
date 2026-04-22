"""
LightGBM baseline. The winning Rossmann Kaggle solutions were GBM variants, so this is
the natural strong baseline to put next to the entity-embedding MLP.

Targets log1p(Sales) to match the RMSPE objective more closely (RMSPE on Sales is
equivalent to RMSE on log Sales up to a second-order term). Final predictions are
exp-backed to the Sales scale.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np

from rossmann_forecast.config import Settings
from rossmann_forecast.features.engineer import (
    CATEGORICAL_COLS,
    CONTINUOUS_COLS,
    TARGET_COL,
    load_bundle,
)
from rossmann_forecast.metrics import mae, rmse, rmspe


@dataclass
class GBMResult:
    rmspe: float
    rmse: float
    mae: float
    best_iteration: int
    train_seconds: float
    model_path: Path
    predictions_path: Path


def run(settings: Settings) -> GBMResult:
    bundle = load_bundle(settings)
    feature_cols = CATEGORICAL_COLS + CONTINUOUS_COLS

    X_train = bundle.train[feature_cols]
    y_train = np.log1p(bundle.train[TARGET_COL].to_numpy(dtype=np.float32))
    X_valid = bundle.valid[feature_cols]
    y_valid = bundle.valid[TARGET_COL].to_numpy(dtype=np.float32)

    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_COLS)
    dvalid = lgb.Dataset(
        X_valid,
        label=np.log1p(y_valid),
        categorical_feature=CATEGORICAL_COLS,
        reference=dtrain,
    )

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": settings.seed,
        "verbosity": -1,
    }

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("rossmann-forecast/gbm")

    with mlflow.start_run(run_name="lightgbm"):
        mlflow.log_params(params)

        t0 = time.perf_counter()
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=3000,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=100)],
        )
        dt = time.perf_counter() - t0

        y_pred_log = booster.predict(X_valid, num_iteration=booster.best_iteration)
        y_pred = np.expm1(y_pred_log).astype(np.float32)

        metrics = {
            "rmspe": rmspe(y_valid, y_pred),
            "rmse": rmse(y_valid, y_pred),
            "mae": mae(y_valid, y_pred),
            "best_iteration": int(booster.best_iteration),
            "train_seconds": dt,
        }
        mlflow.log_metrics(metrics)

        settings.artifacts_root.mkdir(parents=True, exist_ok=True)
        model_path = settings.artifacts_root / "lightgbm.joblib"
        pred_path = settings.artifacts_root / "lightgbm_predictions.npy"
        joblib.dump(booster, model_path)
        np.save(pred_path, y_pred)
        mlflow.log_artifact(str(model_path))

    (settings.artifacts_root / "gbm_summary.json").write_text(json.dumps(metrics, indent=2))

    return GBMResult(
        rmspe=metrics["rmspe"],
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        best_iteration=metrics["best_iteration"],
        train_seconds=dt,
        model_path=model_path,
        predictions_path=pred_path,
    )
