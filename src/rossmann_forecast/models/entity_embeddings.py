"""
Entity-embeddings MLP after Guo & Berkhahn (2016), arXiv:1604.06737.

Paper recipe:
  - One embedding table per categorical input, dim = min(50, cardinality // 2 + 1).
    (This repository uses the same rule but caps at 50; Rossmann's largest column is
    Store with cardinality 1,115, so the Store embedding is the 50-dim cap.)
  - Continuous features concatenated with the embedded vectors.
  - Two hidden layers of 1000 and 500 ReLU units.
  - Adam optimizer; MSE on log1p(Sales) target; early stopping on validation RMSPE.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rossmann_forecast.config import Settings
from rossmann_forecast.features.engineer import load_bundle, to_matrix
from rossmann_forecast.metrics import mae, rmse, rmspe


@dataclass
class EmbeddingConfig:
    batch_size: int = 1024
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    hidden: tuple[int, ...] = (1000, 500)
    dropout: float = 0.1
    patience: int = 3


@dataclass
class EmbeddingResult:
    rmspe: float
    rmse: float
    mae: float
    best_epoch: int
    train_seconds: float
    model_path: Path
    predictions_path: Path


def _embedding_dim(cardinality: int) -> int:
    return min(50, cardinality // 2 + 1)


class RossmannMLP(nn.Module):
    def __init__(
        self,
        cardinalities: list[int],
        n_continuous: int,
        hidden: tuple[int, ...] = (1000, 500),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(c, _embedding_dim(c)) for c in cardinalities]
        )
        in_dim = sum(_embedding_dim(c) for c in cardinalities) + n_continuous
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([*embedded, x_cont], dim=1)
        return self.mlp(x).squeeze(-1)


def _pick_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_loader(
    X_cat: np.ndarray, X_cont: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X_cat),
        torch.from_numpy(X_cont),
        torch.from_numpy(y),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def run(settings: Settings, cfg: EmbeddingConfig | None = None) -> EmbeddingResult:
    cfg = cfg or EmbeddingConfig()
    bundle = load_bundle(settings)

    X_train_cat, X_train_cont, y_train = to_matrix(bundle.train)
    X_valid_cat, X_valid_cont, y_valid = to_matrix(bundle.valid)

    # Target transform: log1p, centered & scaled (standardization on log scale).
    y_train_log = np.log1p(y_train)
    mu, sigma = float(y_train_log.mean()), float(y_train_log.std() or 1.0)
    y_train_n = ((y_train_log - mu) / sigma).astype(np.float32)

    # Continuous standardization (fit on train only).
    cont_mu = X_train_cont.mean(axis=0)
    cont_sd = X_train_cont.std(axis=0) + 1e-6
    X_train_cont = (X_train_cont - cont_mu) / cont_sd
    X_valid_cont = (X_valid_cont - cont_mu) / cont_sd

    device = _pick_device(settings.device)
    torch.manual_seed(settings.seed)
    cardinalities = [bundle.cardinalities[c] for c in bundle.categorical]
    model = RossmannMLP(
        cardinalities=cardinalities,
        n_continuous=len(bundle.continuous),
        hidden=cfg.hidden,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    train_loader = _build_loader(
        X_train_cat, X_train_cont.astype(np.float32), y_train_n, cfg.batch_size, shuffle=True
    )

    valid_cat_t = torch.from_numpy(X_valid_cat).to(device)
    valid_cont_t = torch.from_numpy(X_valid_cont.astype(np.float32)).to(device)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("rossmann-forecast/entity_embeddings")
    settings.artifacts_root.mkdir(parents=True, exist_ok=True)
    model_path = settings.artifacts_root / "entity_embeddings.pt"
    pred_path = settings.artifacts_root / "entity_embeddings_predictions.npy"

    best_rmspe, best_epoch, patience_left = float("inf"), -1, cfg.patience
    t0 = time.perf_counter()

    with mlflow.start_run(run_name="entity_embeddings"):
        mlflow.log_params({
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "hidden": str(cfg.hidden),
            "dropout": cfg.dropout,
            "n_categorical": len(cardinalities),
            "embedding_dim_sum": sum(_embedding_dim(c) for c in cardinalities),
        })

        for epoch in range(cfg.epochs):
            model.train()
            running = 0.0
            n = 0
            for xc, xn, yy in train_loader:
                xc, xn, yy = xc.to(device), xn.to(device), yy.to(device)
                optimizer.zero_grad()
                pred = model(xc, xn)
                loss = loss_fn(pred, yy)
                loss.backward()
                optimizer.step()
                running += loss.item() * yy.size(0)
                n += yy.size(0)
            train_loss = running / max(n, 1)

            model.eval()
            with torch.no_grad():
                pred_n = model(valid_cat_t, valid_cont_t).cpu().numpy()
            pred_log = pred_n * sigma + mu
            y_pred = np.expm1(pred_log).clip(min=0.0)
            r = rmspe(y_valid, y_pred)
            mlflow.log_metrics({"train_loss": train_loss, "valid_rmspe": r}, step=epoch)

            if r < best_rmspe:
                best_rmspe, best_epoch, patience_left = r, epoch, cfg.patience
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "cardinalities": cardinalities,
                        "n_continuous": len(bundle.continuous),
                        "hidden": list(cfg.hidden),
                        "dropout": cfg.dropout,
                        "target_mu": mu,
                        "target_sigma": sigma,
                        "cont_mu": cont_mu.tolist(),
                        "cont_sd": cont_sd.tolist(),
                    },
                    model_path,
                )
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        with torch.no_grad():
            pred_n = model(valid_cat_t, valid_cont_t).cpu().numpy()
        pred_log = pred_n * sigma + mu
        y_pred = np.expm1(pred_log).clip(min=0.0)
        np.save(pred_path, y_pred)

        metrics = {
            "rmspe": rmspe(y_valid, y_pred),
            "rmse": rmse(y_valid, y_pred),
            "mae": mae(y_valid, y_pred),
            "best_epoch": best_epoch,
            "train_seconds": time.perf_counter() - t0,
        }
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(model_path))

    (settings.artifacts_root / "embeddings_summary.json").write_text(json.dumps(metrics, indent=2))

    return EmbeddingResult(
        rmspe=metrics["rmspe"],
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        best_epoch=metrics["best_epoch"],
        train_seconds=metrics["train_seconds"],
        model_path=model_path,
        predictions_path=pred_path,
    )
