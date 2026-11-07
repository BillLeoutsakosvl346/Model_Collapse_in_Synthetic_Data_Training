import pytest

from model_collapse import remote
from scripts.operations import launch_completion_mixed80 as launcher


@pytest.mark.parametrize("failure", [False, True])
def test_startup_observation_waits_for_generation_and_preserves_launcher_guards(monkeypatch, failure):
    calls = []

    def verify(run_id, *, max_wait_seconds=None):
        calls.append((run_id, max_wait_seconds))
        return {"progress_after_disconnect": True}

    def launch(**kwargs):
        assert kwargs["generations"] == 10
        assert kwargs["config_path"].name == "full.yaml"
        assert remote.verify_detached("new-run", max_wait_seconds=600)["progress_after_disconnect"]
        if failure:
            raise RuntimeError("standard admission rejected launch")
        return {"run_id": "new-run"}

    monkeypatch.setattr(remote, "verify_detached", verify)
    monkeypatch.setattr(remote, "launch", launch)
    monkeypatch.setattr(remote, "ledger", lambda: {"launches": []})
    if failure:
        with pytest.raises(RuntimeError, match="standard admission"):
            launcher.launch()
    else:
        assert launcher.launch()["run_id"] == "new-run"
    assert calls == [("new-run", 900)]
    assert remote.verify_detached is verify


def test_existing_full_launch_is_recovered_instead_of_duplicated(monkeypatch):
    monkeypatch.setattr(remote, "ledger", lambda: {"launches": [
        {"allocation": "fineweb_completion64_mixtures_mixed80", "kind": "full", "run_id": "existing"}]})
    monkeypatch.setattr(remote, "launch", lambda **kwargs: pytest.fail("Duplicate launch reached provisioning"))
    with pytest.raises(RuntimeError, match="status or resume for existing"):
        launcher.launch()
