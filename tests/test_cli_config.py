"""Public configuration selection stays explicit without contacting providers."""
import subprocess
import sys

import pytest

from model_collapse import __main__ as cli, remote
from model_collapse.common import PROJECT, config


def test_explicit_config_and_legacy_mode_are_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["model_collapse", "launch", "--mode", "continued", "--config", "unused.yaml"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("command", ["smoke", "launch"])
def test_explicit_config_dispatches_without_provider_calls(monkeypatch, command):
    calls = []
    monkeypatch.setattr(remote, "launch", lambda **kwargs: calls.append(kwargs) or {})
    monkeypatch.setattr(sys, "argv", ["model_collapse", command, "--config", "profile.yaml"])
    cli.main()
    assert calls[0]["config_path"] == "profile.yaml"
    assert calls[0]["smoke"] == (command == "smoke")


def test_legacy_default_remains_original_pretrained_reset(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "launch", lambda **kwargs: calls.append(kwargs) or {})
    monkeypatch.setattr(sys, "argv", ["model_collapse", "launch"])
    cli.main()
    assert calls == [{"generations": 10, "smoke": False, "mode": "fresh_base"}]


def test_new_profiles_share_exact_origin_and_historical_profiles_keep_their_meaning():
    base = PROJECT / "experiments/opt125m_wikitext2"
    full, smoke = (config(base / "reset_to_real_s0" / f"{kind}.yaml") for kind in ("full", "smoke"))
    assert full["initialization"] == smoke["initialization"] == "reset_s0"
    assert full["branch_source"] == smoke["branch_source"]
    assert full["start_checkpoint"] == smoke["start_checkpoint"]
    assert full["branch_source"]["first_new_cycle"] == 2
    assert full["generations"] == 10 and smoke["generations"] == 3
    assert smoke["train_blocks"] == 256 and smoke["validation_blocks"] == smoke["test_blocks"] == 64
    for condition, legacy_suffix in [("pretrained_reset", ""), ("continued_from_real_s0", "-continued")]:
        for kind in ("full", "smoke"):
            historical = config(PROJECT / f"configs/{kind}{legacy_suffix}.yaml")
            named = config(base / condition / f"{kind}.yaml")
            assert all(named[k] == v for k, v in historical.items())


def test_watchdog_import_does_not_require_site_packages():
    result = subprocess.run([sys.executable, "-S", "-c", "import model_collapse.supervisor"], cwd=PROJECT,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("smoke,kind", [(True, "full"), (False, "smoke")])
def test_launch_kind_mismatch_is_rejected_before_network(monkeypatch, smoke, kind):
    monkeypatch.setattr(remote, "Runpod", lambda: pytest.fail("A mismatched command must not contact providers"))
    with pytest.raises(ValueError, match="configuration kind"):
        remote._launch(smoke=smoke, config_path=PROJECT / f"configs/{kind}.yaml")


def test_source_resolution_loads_environment_before_authentication(monkeypatch):
    from model_collapse.workflow import branch
    loaded = []
    monkeypatch.setattr(remote, "load_secrets", lambda: loaded.append(True))
    def resolve(c):
        assert loaded, "Private Hub source resolution needs .env credentials first"
        return c
    monkeypatch.setattr(branch, "resolve_branch_source", resolve)
    class Boundary(Exception):
        pass
    monkeypatch.setattr(remote, "Runpod", lambda: (_ for _ in ()).throw(Boundary()))
    with pytest.raises(Boundary):
        remote._launch(smoke=True, config_path=PROJECT / "experiments/opt125m_wikitext2/reset_to_real_s0/smoke.yaml")
