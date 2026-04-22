"""Competition metric (RMSPE) and common regression metrics."""

from __future__ import annotations

import numpy as np


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-Mean-Square-Percentage-Error, Kaggle Rossmann metric.

    Rows with y_true == 0 are ignored (consistent with the competition rules,
    but also because the ratio is undefined).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = y_true > 0
    if not mask.any():
        return float("nan")
    r = (y_true[mask] - y_pred[mask]) / y_true[mask]
    return float(np.sqrt(np.mean(r * r)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))
