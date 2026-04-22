"""Typer CLI entry points."""

from __future__ import annotations

import typer
from rich import print as rprint

from rossmann_forecast.config import Settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Rossmann forecast CLI.")


@app.command("download")
def download() -> None:
    """Download the Kaggle Rossmann Store Sales dataset via the Kaggle CLI."""
    from rossmann_forecast.data.download import download as _dl

    out = _dl(Settings())
    rprint({k: str(v) for k, v in out.items()})


@app.command("build-features")
def build_features() -> None:
    """Engineer features, chronological split, persist train/valid parquet."""
    from rossmann_forecast.features.engineer import build

    settings = Settings()
    bundle = build(settings)
    bundle.save(settings.processed_dir)
    rprint({
        "train_rows": len(bundle.train),
        "valid_rows": len(bundle.valid),
        "cardinalities": bundle.cardinalities,
    })


@app.command("train-baseline")
def train_baseline() -> None:
    """Train seasonal-naive and per-store-median baselines."""
    from rossmann_forecast.models.baseline import run

    for r in run(Settings()):
        rprint(f"{r.name}: rmspe={r.rmspe:.4f} rmse={r.rmse:.1f} mae={r.mae:.1f}")


@app.command("train-gbm")
def train_gbm() -> None:
    """Train LightGBM on engineered features with early stopping."""
    from rossmann_forecast.models.gbm import run

    r = run(Settings())
    rprint({"rmspe": r.rmspe, "rmse": r.rmse, "mae": r.mae, "best_iteration": r.best_iteration})


@app.command("train-emb")
def train_emb() -> None:
    """Train the Guo & Berkhahn entity-embeddings MLP."""
    from rossmann_forecast.models.entity_embeddings import run

    r = run(Settings())
    rprint({"rmspe": r.rmspe, "rmse": r.rmse, "mae": r.mae, "best_epoch": r.best_epoch})


@app.command("evaluate")
def evaluate() -> None:
    """Build comparison table, bar chart, residual histogram."""
    from rossmann_forecast.evaluate import run

    out = run(Settings())
    rprint(f"Wrote [bold]{out}[/bold]")


if __name__ == "__main__":
    app()
