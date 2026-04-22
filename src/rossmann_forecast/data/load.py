"""
Raw-CSV → typed pandas DataFrames.

Rossmann quirks handled here:
  - StateHoliday column is mixed int/str; cast to str.
  - Sales == 0 rows are dropped for training (competition rule: not scored).
  - Closed-store rows (Open == 0) have Sales == 0; dropped together with the above.
  - Date parsed to pandas datetime.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TRAIN_DTYPES = {
    "Store": "int32",
    "DayOfWeek": "int8",
    "Sales": "int32",
    "Customers": "int32",
    "Open": "int8",
    "Promo": "int8",
    "StateHoliday": "string",
    "SchoolHoliday": "int8",
}

STORE_DTYPES = {
    "Store": "int32",
    "StoreType": "string",
    "Assortment": "string",
    "CompetitionDistance": "float32",
    "CompetitionOpenSinceMonth": "Int8",
    "CompetitionOpenSinceYear": "Int16",
    "Promo2": "int8",
    "Promo2SinceWeek": "Int8",
    "Promo2SinceYear": "Int16",
    "PromoInterval": "string",
}


def load_train(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "train.csv", dtype=TRAIN_DTYPES, parse_dates=["Date"])
    df["StateHoliday"] = df["StateHoliday"].astype("string").fillna("0")
    return df


def load_test(raw_dir: Path) -> pd.DataFrame:
    dtypes = {k: v for k, v in TRAIN_DTYPES.items() if k not in ("Sales", "Customers")}
    dtypes["Id"] = "int32"
    df = pd.read_csv(raw_dir / "test.csv", dtype=dtypes, parse_dates=["Date"])
    df["Open"] = df["Open"].fillna(1).astype("int8")
    df["StateHoliday"] = df["StateHoliday"].astype("string").fillna("0")
    return df


def load_store(raw_dir: Path) -> pd.DataFrame:
    return pd.read_csv(raw_dir / "store.csv", dtype=STORE_DTYPES)


def drop_unscored_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Kaggle metric ignores rows with Sales == 0; closed-store rows also dropped."""
    mask = (df["Open"] == 1) & (df["Sales"] > 0)
    return df.loc[mask].reset_index(drop=True)
