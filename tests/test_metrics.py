import numpy as np

from rossmann_forecast.metrics import mae, rmse, rmspe


def test_rmspe_perfect_predictions_is_zero():
    y = np.array([100.0, 200.0, 300.0])
    assert rmspe(y, y) == 0.0


def test_rmspe_ignores_zero_targets():
    y = np.array([0.0, 100.0])
    yhat = np.array([5000.0, 100.0])  # the zero target would blow up if not masked
    assert rmspe(y, yhat) == 0.0


def test_rmspe_one_percent_error():
    y = np.array([100.0, 100.0])
    yhat = np.array([101.0, 99.0])
    assert abs(rmspe(y, yhat) - 0.01) < 1e-9


def test_rmse_and_mae_symmetry():
    y = np.array([0.0, 10.0])
    yhat = np.array([1.0, 11.0])
    assert abs(rmse(y, yhat) - 1.0) < 1e-9
    assert abs(mae(y, yhat) - 1.0) < 1e-9
