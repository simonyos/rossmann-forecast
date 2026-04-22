"""
Feature engineering for Rossmann.

Mirrors the feature set used by Guo & Berkhahn (2016), arXiv:1604.06737, Table 2,
augmented with a few continuous features that also benefit the gradient-boosted
baseline: CompetitionDistance, a months-since-competition-opened scalar, and
a binary in-PromoInterval-this-month flag.

All categorical columns are pre-indexed to small integer codes so torch embedding
tables and LightGBM can consume them directly.

Chronological split follows the Kaggle convention: the last 6 weeks of the training
period (2015-06-19 → 2015-07-31 inclusive) are held out as the public validation
set; everything before that is training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rossmann_forecast.config import Settings
from rossmann_forecast.data.load import (
    drop_unscored_rows,
    load_store,
    load_train,
)

CATEGORICAL_COLS = [
    "Store",
    "DayOfWeek",
    "Day",
    "Month",
    "Year",
    "StateHoliday",
    "SchoolHoliday",
    "Promo",
    "StoreType",
    "Assortment",
    "Promo2",
    "PromoIntervalCode",
    "InPromoIntervalMonth",
]

CONTINUOUS_COLS = [
    "CompetitionDistance",
    "CompetitionOpenMonths",
]

TARGET_COL = "Sales"
VALIDATION_START = pd.Timestamp("2015-06-19")


@dataclass
class FeatureBundle:
    train: pd.DataFrame
    valid: pd.DataFrame
    categorical: list[str]
    continuous: list[str]
    cardinalities: dict[str, int]

    def save(self, parquet_dir: Path) -> None:
        parquet_dir.mkdir(parents=True, exist_ok=True)
        self.train.to_parquet(parquet_dir / "train.parquet", index=False)
        self.valid.to_parquet(parquet_dir / "valid.parquet", index=False)


def _parse_promo_interval(
    df: pd.DataFrame, date_col: str = "Date", interval_col: str = "PromoInterval"
) -> pd.Series:
    month_abbr_to_num = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }

    def months_in(s: str | None) -> frozenset[int]:
        if s is None or pd.isna(s) or not s:
            return frozenset()
        return frozenset(month_abbr_to_num[m] for m in s.split(",") if m in month_abbr_to_num)

    month_sets = df[interval_col].map(months_in)
    this_month = df[date_col].dt.month
    return pd.Series(
        [m in s for m, s in zip(this_month, month_sets, strict=False)], index=df.index
    ).astype("int8")


def _encode_category(series: pd.Series, known: list | None = None) -> tuple[pd.Series, list]:
    vals = series.astype("string").fillna("__nan__")
    uniq = sorted(vals.unique().tolist()) if known is None else list(known)
    mapping = {v: i for i, v in enumerate(uniq)}
    default = mapping.get("__nan__", 0)
    codes = vals.map(lambda v: mapping.get(v, default)).astype("int32")
    return codes, uniq


def build(settings: Settings) -> FeatureBundle:
    train = load_train(settings.raw_dir)
    store = load_store(settings.raw_dir)
    train = drop_unscored_rows(train)

    df = train.merge(store, on="Store", how="left")

    df["Day"] = df["Date"].dt.day.astype("int8")
    df["Month"] = df["Date"].dt.month.astype("int8")
    df["Year"] = df["Date"].dt.year.astype("int16")

    df["PromoInterval"] = df["PromoInterval"].fillna("None")
    df["InPromoIntervalMonth"] = (
        _parse_promo_interval(df).where(df["Promo2"] == 1, 0).astype("int8")
    )

    # Continuous: months-since-competition-opened, clipped to [0, 120].
    comp_year = df["CompetitionOpenSinceYear"].astype("float32")
    comp_month = df["CompetitionOpenSinceMonth"].astype("float32")
    current_months = df["Year"].astype("float32") * 12 + df["Month"].astype("float32")
    open_months = comp_year * 12 + comp_month
    comp_open = (current_months - open_months).clip(lower=0, upper=120)
    df["CompetitionOpenMonths"] = comp_open.fillna(0).astype("float32")

    median_distance = float(df["CompetitionDistance"].median())
    df["CompetitionDistance"] = (
        df["CompetitionDistance"].fillna(median_distance).astype("float32")
    )

    cardinalities: dict[str, int] = {}
    for col in CATEGORICAL_COLS:
        if col == "PromoIntervalCode":
            codes, uniq = _encode_category(df["PromoInterval"])
        else:
            codes, uniq = _encode_category(df[col])
        df[col] = codes
        cardinalities[col] = len(uniq)

    cols = ["Date", TARGET_COL, *CATEGORICAL_COLS, *CONTINUOUS_COLS]
    df = df[cols].copy()

    valid_mask = df["Date"] >= VALIDATION_START
    train_df = df.loc[~valid_mask].reset_index(drop=True)
    valid_df = df.loc[valid_mask].reset_index(drop=True)

    return FeatureBundle(
        train=train_df,
        valid=valid_df,
        categorical=list(CATEGORICAL_COLS),
        continuous=list(CONTINUOUS_COLS),
        cardinalities=cardinalities,
    )


def load_bundle(settings: Settings) -> FeatureBundle:
    train = pd.read_parquet(settings.train_parquet)
    valid = pd.read_parquet(settings.valid_parquet)
    cardinalities = {c: int(max(train[c].max(), valid[c].max()) + 1) for c in CATEGORICAL_COLS}
    return FeatureBundle(
        train=train,
        valid=valid,
        categorical=list(CATEGORICAL_COLS),
        continuous=list(CONTINUOUS_COLS),
        cardinalities=cardinalities,
    )


def to_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_cat = df[CATEGORICAL_COLS].to_numpy(dtype=np.int64)
    X_cont = df[CONTINUOUS_COLS].to_numpy(dtype=np.float32)
    y = df[TARGET_COL].to_numpy(dtype=np.float32)
    return X_cat, X_cont, y
