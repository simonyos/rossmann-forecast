def test_cli_imports():
    from rossmann_forecast.cli import app

    assert app is not None
    assert {c.name for c in app.registered_commands} >= {
        "download",
        "build-features",
        "train-baseline",
        "train-gbm",
        "train-emb",
        "evaluate",
    }
