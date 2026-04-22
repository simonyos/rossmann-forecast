from pathlib import Path

from rossmann_forecast.config import Settings


def test_derived_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ARTIFACTS_ROOT", str(tmp_path / "art"))
    s = Settings()
    assert s.raw_dir == Path(tmp_path / "data" / "raw")
    assert s.processed_dir == Path(tmp_path / "data" / "processed")
    assert s.features_path == Path(tmp_path / "data" / "processed" / "features.parquet")


def test_ensure_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ARTIFACTS_ROOT", str(tmp_path / "art"))
    s = Settings()
    s.ensure_dirs()
    assert s.raw_dir.is_dir()
    assert s.processed_dir.is_dir()
    assert s.artifacts_root.is_dir()
