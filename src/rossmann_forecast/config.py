import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_root: Path = field(default_factory=lambda: Path(os.getenv("DATA_ROOT", "./data")))
    artifacts_root: Path = field(
        default_factory=lambda: Path(os.getenv("ARTIFACTS_ROOT", "./artifacts"))
    )
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    device: str = os.getenv("DEVICE", "auto")
    seed: int = int(os.getenv("SEED", "42"))

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_root / "processed"

    @property
    def features_path(self) -> Path:
        return self.processed_dir / "features.parquet"

    @property
    def train_parquet(self) -> Path:
        return self.processed_dir / "train.parquet"

    @property
    def valid_parquet(self) -> Path:
        return self.processed_dir / "valid.parquet"

    def ensure_dirs(self) -> None:
        for p in (self.raw_dir, self.processed_dir, self.artifacts_root):
            p.mkdir(parents=True, exist_ok=True)
