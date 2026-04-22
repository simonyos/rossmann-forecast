import pandas as pd

from rossmann_forecast.features.engineer import _parse_promo_interval


def test_in_promo_interval_flag():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2014-03-15", "2014-06-01", "2014-07-04"]),
        "PromoInterval": ["Jan,Apr,Jul,Oct", "Feb,May,Aug,Nov", "Mar,Jun,Sept,Dec"],
    })
    out = _parse_promo_interval(df).tolist()
    # row 0: Mar in {Jan,Apr,Jul,Oct} → 0
    # row 1: Jun in {Feb,May,Aug,Nov} → 0
    # row 2: Jul in {Mar,Jun,Sept,Dec} → 0
    assert out == [0, 0, 0]


def test_in_promo_interval_hit():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2014-04-15", "2014-10-01"]),
        "PromoInterval": ["Jan,Apr,Jul,Oct", "Jan,Apr,Jul,Oct"],
    })
    out = _parse_promo_interval(df).tolist()
    assert out == [1, 1]


def test_in_promo_interval_none():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2014-04-15"]),
        "PromoInterval": [None],
    })
    assert _parse_promo_interval(df).tolist() == [0]
