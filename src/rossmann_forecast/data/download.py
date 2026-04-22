"""
Kaggle Rossmann Store Sales downloader.

Requires a Kaggle API token at ~/.kaggle/kaggle.json. Competition rules require
you to accept the Rossmann competition terms on the Kaggle website before the
API will serve the data:

  https://www.kaggle.com/competitions/rossmann-store-sales

Produces, under data/raw/:
  train.csv (~1.0M rows)
  test.csv  (~41k rows, no Sales column)
  store.csv (1,115 rows)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from rossmann_forecast.config import Settings

COMPETITION = "rossmann-store-sales"
EXPECTED_FILES = ("train.csv", "test.csv", "store.csv")


def _kaggle_cli_available() -> bool:
    return shutil.which("kaggle") is not None


def download(settings: Settings, force: bool = False) -> dict[str, Path]:
    settings.ensure_dirs()
    raw = settings.raw_dir
    existing = {name: raw / name for name in EXPECTED_FILES if (raw / name).is_file()}
    if len(existing) == len(EXPECTED_FILES) and not force:
        return existing

    if not _kaggle_cli_available():
        raise RuntimeError(
            "Kaggle CLI not found. `pip install kaggle` and place your API token at "
            "~/.kaggle/kaggle.json (chmod 600)."
        )

    cmd = ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(raw)]
    env = os.environ.copy()
    subprocess.run(cmd, check=True, env=env)

    zip_path = raw / f"{COMPETITION}.zip"
    if zip_path.is_file():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw)
        zip_path.unlink()

    for inner_zip in raw.glob("*.zip"):
        with zipfile.ZipFile(inner_zip) as zf:
            zf.extractall(raw)
        inner_zip.unlink()

    produced = {name: raw / name for name in EXPECTED_FILES if (raw / name).is_file()}
    missing = set(EXPECTED_FILES) - set(produced)
    if missing:
        raise RuntimeError(f"Download complete but missing expected files: {sorted(missing)}")
    return produced
