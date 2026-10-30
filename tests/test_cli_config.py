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


def test_full_reset_launch_reaches_provisioning_boundary_without_spending(tmp_path, monkeypatch):
    from model_collapse.common import atomic_json
    from model_collapse.workflow import branch
    from model_collapse import pipeline
    calls = []
    class Boundary(Exception):
        pass
    class API:
        def request(self, method, path, body=None):
            calls.append((method, path))
            assert method == "GET", "A dry-run admission test must never provision"
            return ({"pods": []} if path == "/pods" else
                    {"templates": [{"id": "runpod-torch-v280", "image": "recorded-image"}]})
        def candidates(self):
            return [{"gpu": "NVIDIA RTX PRO 4000 Blackwell", "hourly_usd": .57, "data_center": "EU-RO-1"}]
    monkeypatch.setattr(remote, "STATE", tmp_path)
    monkeypatch.setattr(remote, "Runpod", API)
    monkeypatch.setattr(remote, "preflight", lambda **kwargs: {"account": {"clientBalance": 23}})
    monkeypatch.setattr(remote, "ledger", lambda: {"launches": []})
    monkeypatch.setattr(remote, "source_archive", lambda: ("unused", "tested-runtime"))
    monkeypatch.setattr(branch, "resolve_branch_source", lambda c: c)
    monkeypatch.setattr(remote, "verify_start_checkpoint", lambda c: None)
    monkeypatch.setattr(remote, "register_key", lambda api: (_ for _ in ()).throw(Boundary()))
    original_forecast = pipeline.forecast
    def forecast(*args, **kwargs):
        estimate = original_forecast(*args, **kwargs)
        assert estimate["initialization"] == "reset_s0"
        assert estimate["new_training_stages"] == 9
        assert estimate["new_full_generation_stages"] == 8
        return estimate
    monkeypatch.setattr(pipeline, "forecast", forecast)
    from model_collapse.workflow.profiles import protocol_fingerprint
    atomic_json(tmp_path / "validation_reset_s0.json", {"source_code_sha256": "tested-runtime",
                "protocol_fingerprint": protocol_fingerprint(config(PROJECT / "experiments/opt125m_wikitext2/reset_to_real_s0/full.yaml"))})
    atomic_json(tmp_path / "benchmark.json", {"training_target_tokens_per_second": 30000,
                "generation_new_tokens_per_second": 1300,
                "full_split_blocks": {"train": 37946, "validation": 3922, "test": 4494}})
    with pytest.raises(Boundary):
        remote.launch(generations=10, config_path=PROJECT / "experiments/opt125m_wikitext2/reset_to_real_s0/full.yaml")
    assert calls and all(method == "GET" for method, _ in calls)


@pytest.mark.parametrize("value", [0, -1, float("nan"), True])
def test_invalid_session_caps_are_rejected(value):
    from model_collapse.common import validate_config
    with pytest.raises(ValueError, match="positive finite"):
        validate_config({"max_session_usd": value})


def test_smoke_session_cap_reaches_provisioning_as_configured(tmp_path, monkeypatch, capsys):
    import json
    from model_collapse.workflow import branch
    class Boundary(Exception):
        pass
    class API:
        def candidates(self):
            return [{"gpu": "NVIDIA RTX PRO 4000 Blackwell", "hourly_usd": .57, "data_center": "EU-RO-1"}]
        def request(self, method, path, body=None):
            if method == "POST":
                assert path == "/network-volumes"
                raise Boundary()
            return ({"pods": []} if path == "/pods" else
                    {"templates": [{"id": "runpod-torch-v280", "image": "recorded-image"}]})
    monkeypatch.setattr(remote, "PROJECT", tmp_path)
    monkeypatch.setattr(remote, "Runpod", API)
    monkeypatch.setattr(remote, "STATE", tmp_path / "state")
    monkeypatch.setattr(remote, "preflight", lambda **kwargs: {"account": {"clientBalance": 23}})
    monkeypatch.setattr(remote, "ledger", lambda: {"launches": []})
    monkeypatch.setattr(remote, "source_archive", lambda: ("unused", "runtime"))
    monkeypatch.setattr(branch, "resolve_branch_source", lambda c: c)
    monkeypatch.setattr(remote, "verify_start_checkpoint", lambda c: None)
    monkeypatch.setattr(remote, "register_key", lambda api: "unused")
    with pytest.raises(Boundary):
        remote.launch(smoke=True, config_path=PROJECT / "experiments/opt125m_wikitext2/reset_to_real_s0/smoke.yaml")
    output = json.loads(capsys.readouterr().out)
    assert output["session_allowance_usd"] == .60
