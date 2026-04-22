"""Build the comparison table and diagnostic figures across all trained models."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rossmann_forecast.config import Settings
from rossmann_forecast.features.engineer import TARGET_COL, load_bundle


def _load_predictions(artifacts_dir: Path) -> list[dict]:
    rows: list[dict] = []

    baseline_path = artifacts_dir / "baseline_summary.json"
    if baseline_path.exists():
        for entry in json.loads(baseline_path.read_text()):
            pred_path = artifacts_dir / f"{entry['name']}_predictions.npy"
            rows.append({**entry, "predictions_path": str(pred_path)})

    gbm_path = artifacts_dir / "gbm_summary.json"
    if gbm_path.exists():
        metrics = json.loads(gbm_path.read_text())
        rows.append({
            "name": "lightgbm",
            **metrics,
            "predictions_path": str(artifacts_dir / "lightgbm_predictions.npy"),
        })

    emb_path = artifacts_dir / "embeddings_summary.json"
    if emb_path.exists():
        metrics = json.loads(emb_path.read_text())
        rows.append({
            "name": "entity_embeddings",
            **metrics,
            "predictions_path": str(artifacts_dir / "entity_embeddings_predictions.npy"),
        })

    return rows


def _bar_chart(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ordered = df.sort_values("rmspe")
    ax.barh(ordered["name"], ordered["rmspe"])
    ax.invert_yaxis()
    ax.set_xlabel("Validation RMSPE (lower is better)")
    for i, v in enumerate(ordered["rmspe"]):
        ax.text(v + 0.002, i, f"{v:.4f}", va="center")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _residual_hist(
    y_true: np.ndarray, rows: list[dict], out: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for row in rows:
        path = Path(row["predictions_path"])
        if not path.exists():
            continue
        y_pred = np.load(path)
        residual = (y_true - y_pred) / np.maximum(y_true, 1.0)
        ax.hist(residual.clip(-1, 1), bins=60, histtype="step", label=row["name"], linewidth=1.2)
    ax.set_xlabel("(y - yhat) / y (clipped to [-1, 1])")
    ax.set_ylabel("Count")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run(settings: Settings) -> Path:
    figs_dir = Path("reports/figures")
    rows = _load_predictions(settings.artifacts_root)
    if not rows:
        raise FileNotFoundError("No trained models found under artifacts/. Run training first.")

    df = pd.DataFrame(rows)
    _bar_chart(df, figs_dir / "rmspe_comparison.png")

    bundle = load_bundle(settings)
    y_true = bundle.valid[TARGET_COL].to_numpy(dtype=np.float32)
    _residual_hist(y_true, rows, figs_dir / "residual_distribution.png")

    ranked = df.sort_values("rmspe")[["name", "rmspe", "rmse", "mae", "train_seconds"]]
    md = ["# Results summary", "", ranked.to_markdown(index=False, floatfmt=".4f")]
    out_md = Path("reports/summary.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md))
    return out_md
