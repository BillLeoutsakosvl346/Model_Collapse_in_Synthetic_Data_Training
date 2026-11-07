"""Offline tests have no access to live credentials, sockets or operational state."""
import socket
import sys

import pytest


@pytest.fixture(autouse=True)
def offline_boundary(monkeypatch, tmp_path):
    from model_collapse import common, paths

    def denied(*args, **kwargs):
        raise AssertionError("Offline test attempted an external connection")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("MODEL_COLLAPSE_STATE_DIR", str(tmp_path / "state"))
    for key in ("RUNPOD_API_KEY", "RUNPOD_S3_ACCESS_KEY", "RUNPOD_S3_SECRET_KEY",
                "RUNPOD_S3_ACCESS_KEY_ID", "RUNPOD_S3_SECRET_ACCESS_KEY",
                "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "HF_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HF_TOKEN", "offline-test-placeholder")
    # Also cover aliases imported during test collection. Local tests may replace
    # these with explicit fakes after this fixture establishes the boundary.
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__name__", "")
        if name.startswith("model_collapse"):
            if hasattr(module, "STATE"):
                monkeypatch.setattr(module, "STATE", tmp_path / "state")
            if hasattr(module, "load_secrets"):
                monkeypatch.setattr(module, "load_secrets", lambda: None)
    monkeypatch.setattr(paths, "STATE", tmp_path / "state")
    monkeypatch.setattr(common, "load_secrets", lambda: None)
