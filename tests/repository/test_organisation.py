"""Contract and failure-path regressions for the repository organisation."""
import hashlib
import importlib
import json
import os
import socket
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from model_collapse.common import PROJECT, atomic_json, config, read_json
from scripts.maintenance.check_repository import heading_anchors, markdown_errors, private_path, publication_errors


@pytest.mark.parametrize("legacy,canonical", [
    ("engine", "science.engine"), ("data", "science.wikitext"),
    ("pipeline", "workflow.wikitext"), ("remote", "cloud.execution"),
    ("supervisor", "cloud.supervisor"), ("runpod_api", "cloud.runpod"),
    ("hub", "artifacts.hub"),
])
def test_legacy_modules_share_implementation_globals(legacy, canonical, monkeypatch):
    old = importlib.import_module("model_collapse." + legacy)
    new = importlib.import_module("model_collapse." + canonical)
    assert old is new
    monkeypatch.setattr(old, "compatibility_probe", object(), raising=False)
    assert new.compatibility_probe is old.compatibility_probe


@pytest.mark.parametrize("failure", ["heartbeat", "event", "receipt", "malformed_control",
                                     "malformed_guard", "missing_guard_deadline", "nan_guard", "bool_guard"])
def test_watchdog_always_stops_despite_failed_local_evidence(tmp_path, monkeypatch, failure):
    from model_collapse.cloud import supervisor
    atomic_json(tmp_path / "resources.json", {"created_at": 0, "session_allowance_usd": .01,
                "guard_hourly_usd": 1, "pod_id": "owned"})
    if failure == "malformed_control":
        (tmp_path / "stop_request.json").write_text("{broken")
    if failure.endswith("guard") or failure == "missing_guard_deadline":
        content = {"malformed_guard": "{broken", "missing_guard_deadline": "{}",
                   "nan_guard": '{"deadline": NaN}', "bool_guard": '{"deadline": true}'}[failure]
        (tmp_path / "validation_guard_deadline.json").write_text(content)
    write = supervisor.atomic_json
    def failing_write(path, value):
        if failure == "heartbeat" or (failure == "receipt" and path.name == "watchdog_exit.json"):
            raise OSError("disk full")
        return write(path, value)
    def failing_event(*args, **kwargs):
        raise OSError("log unavailable")
    stopped = []
    monkeypatch.setattr(supervisor, "atomic_json", failing_write)
    if failure == "event":
        monkeypatch.setattr(supervisor, "event", failing_event)
    monkeypatch.setattr(supervisor, "_publish_shutdown_intent", lambda *args: None)
    monkeypatch.setattr(supervisor, "Runpod", lambda: SimpleNamespace(stop=lambda pod: stopped.append(pod)))
    supervisor.watchdog(tmp_path)
    assert stopped == ["owned"]


@pytest.mark.parametrize("state", ["CREATED", "PENDING", "STARTING", "UNKNOWN"])
def test_nonterminal_status_never_settles_a_launch(tmp_path, monkeypatch, state):
    from model_collapse import remote
    owner = {"run_id": "legacy", "pod_id": "pod"}
    monkeypatch.setattr(remote, "PROJECT", tmp_path)
    monkeypatch.setattr(remote, "local_resources", lambda _: owner)
    monkeypatch.setattr(remote, "Runpod", lambda: SimpleNamespace(pod=lambda _: {"status": state, "cost": .5}))
    monkeypatch.setattr(remote, "update_ledger", lambda *a, **k: pytest.fail("Nonterminal state settled compute"))
    assert remote.status("legacy")["compute_running"]


def test_reused_pod_resolves_latest_exact_volume_session():
    from model_collapse.cloud.ownership import latest_session
    owner = {"run_id": "run", "volume_id": "pod", "pod_id": "pod", "launch_id": "new"}
    rows = [{**owner, "launch_id": "old", "created_at": 1}, {**owner, "created_at": 2},
            {**owner, "volume_id": "foreign", "created_at": 3}]
    assert latest_session(rows, owner, "pod")["launch_id"] == "new"


@pytest.mark.parametrize("profile", ["configs/smoke.yaml", "comparisons/fineweb_completion64_mixed85/smoke.yaml",
                                    "experiments/opt125m_fineweb_edu/fresh_real_one_epoch/smoke.yaml"])
@pytest.mark.parametrize("key,value", [("learning_rate", float("inf")), ("weight_decay", float("nan")),
                                     ("seed", 43), ("seed", True)])
def test_shared_config_validation_rejects_invalid_protocol_numbers(profile, key, value):
    from model_collapse.common import validate_config
    c = config(PROJECT / profile)
    with pytest.raises(ValueError):
        validate_config({**c, key: value})


def test_shared_hash_helper_preserves_legacy_int32_bytes():
    from model_collapse.science.identity import token_hash
    from model_collapse.completion_mixing import token_hash as completion_hash
    from model_collapse.workflow.lineage import token_hash as block_hash
    values = np.array([[0, 1, 65535], [4, 5, 6]], dtype=np.int64)
    expected = hashlib.sha256(values.astype(np.int32).tobytes()).hexdigest()
    assert token_hash(values) == completion_hash(values) == block_hash(values) == expected


def test_offline_test_boundary_denies_sockets_and_disables_credentials():
    from model_collapse.common import load_secrets
    assert load_secrets() is None
    assert not any(key in os.environ for key in ("RUNPOD_API_KEY", "RUNPOD_S3_ACCESS_KEY_ID",
                                                "RUNPOD_S3_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY"))
    with socket.socket() as sock:
        with pytest.raises(AssertionError, match="external connection"):
            sock.connect(("192.0.2.1", 443))


def test_cli_reporting_does_not_import_training_or_provider_clients():
    script = """import sys
from model_collapse import __main__ as cli
sys.argv = ['model_collapse', 'report', 'nonexistent-offline-fixture']
try:
    cli.main()
except (RuntimeError, ValueError, FileNotFoundError, KeyError):
    pass
assert 'torch' not in sys.modules
assert 'model_collapse.cloud.execution' not in sys.modules
assert 'paramiko' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=PROJECT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("manifest", [{}, {"files": {}}, {"files": []}, {"artifacts": {}}])
def test_empty_publication_inventories_are_rejected(tmp_path, manifest):
    root = tmp_path / "reports/run"
    root.mkdir(parents=True)
    path = root / "publication_manifest.json"
    path.write_text(json.dumps(manifest))
    assert publication_errors(tmp_path, [path])


def test_historical_artifacts_inventory_is_actually_checked(tmp_path):
    root = tmp_path / "reports/run"
    root.mkdir(parents=True)
    data = root / "data.json"
    data.write_bytes(b"{}")
    manifest = root / "publication_manifest.json"
    manifest.write_text(json.dumps({"artifacts": {"data.json": hashlib.sha256(b"{}").hexdigest()}}))
    assert not publication_errors(tmp_path, [manifest])
    data.write_bytes(b"changed")
    assert publication_errors(tmp_path, [manifest])


@pytest.mark.parametrize("name", [".env.production", ".ssh/id_ed25519", ".runpod/key", "model.safetensors",
                                 "Seddik_2024.pdf", "docs/project/proposal.docx", "literature/papers/paper.pdf"])
def test_private_path_policy_covers_existing_ignore_contract(name):
    assert private_path(name)


@pytest.mark.parametrize("name", [".env.example", "reports/run/probe.npy", "reports/run/figure.pdf"])
def test_scientific_publications_are_allowed(name):
    assert not private_path(name)


def test_markdown_anchor_check_handles_duplicates_and_local_fragments(tmp_path):
    doc = tmp_path / "README.md"
    doc.write_text("# Hello, world!\n# Hello, world!\n[one](#hello-world)\n[two](#hello-world-1)\n[bad](#absent)\n")
    assert "hello-world-1" in heading_anchors(doc.read_text())
    errors = markdown_errors(tmp_path, [doc])
    assert len(errors) == 1 and "#absent" in errors[0]


def test_archived_operational_chronology_retains_exact_bytes():
    snapshots = read_json(PROJECT / "docs/history/2026-11-07_cleanup-snapshots.json")
    for meta in snapshots.values():
        assert hashlib.sha256((PROJECT / meta["snapshot"]).read_bytes()).hexdigest() == meta["sha256"]


def test_unknown_workflow_is_rejected_before_legacy_fallback():
    from model_collapse.common import validate_config
    with pytest.raises(ValueError, match="Unknown workflow"):
        validate_config({**config(PROJECT / "configs/smoke.yaml"), "workflow": "typo"})


def test_setup_allowance_uses_recorded_limit_and_preserves_default():
    from model_collapse import remote
    assert remote.setup_budget({}) == remote.SETUP_BUDGET
    assert remote.setup_budget({"setup_budget_usd": 3.5}) == 3.5
    with pytest.raises(RuntimeError, match="positive finite"):
        remote.setup_budget({"setup_budget_usd": float("nan")})
