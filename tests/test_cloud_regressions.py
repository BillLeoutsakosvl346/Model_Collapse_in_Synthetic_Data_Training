"""Cloud safety regressions run with local fixtures; no paid API calls."""
import json
import tarfile
from types import SimpleNamespace

import pytest

from model_collapse import remote
from model_collapse.common import atomic_json, read_json


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "PROJECT", tmp_path)
    monkeypatch.setattr(remote, "STATE", tmp_path / ".state")
    monkeypatch.setattr(remote, "now", lambda: 36000.)
    return tmp_path


def resource(**changes):
    return {"run_id": "run", "launch_id": "one", "kind": "smoke", "pod_id": "pod-one",
            "volume_id": "volume", "created_at": 0., "guard_hourly_usd": 1.2,
            "compute_hourly_usd": 1., "volume_hourly_usd": .1, **changes}


def test_compute_settles_once_storage_closes_at_deletion(sandbox):
    r = resource()
    remote.update_ledger(r, state="running")
    remote.update_ledger(r, state="exited", stopped_at=3600.)
    assert remote.accrued(remote.ledger()) == pytest.approx(2.)  # 1 compute + 10h storage
    remote.update_ledger(r, state="removed", stopped_at=36000., volume_deleted_at=7200., estimated_usd=99.)
    settled = remote.ledger()
    assert settled["launches"][0]["stopped_at"] == 3600.
    assert settled["launches"][0]["compute_stopped_at"] == 3600.
    assert remote.accrued(settled) == pytest.approx(1.2)


def test_resuming_a_volume_does_not_double_charge_storage(sandbox):
    remote.update_ledger(resource(), stopped_at=3600.)
    remote.update_ledger(resource(launch_id="two", pod_id="pod-two", created_at=7200.), stopped_at=10800.)
    assert len(remote.ledger()["volumes"]) == 1
    assert remote.accrued(remote.ledger()) == pytest.approx(3.)  # two compute hours + ten storage hours


def test_legacy_retained_volume_intervals_are_coalesced(sandbox):
    entry = {"kind": "full", "run_id": "run", "volume_id": "same", "created_at": 0,
             "guard_hourly_usd": 1, "estimated_usd": 1, "stopped_at": 3600,
             "volume_retained_since": 3600, "volume_hourly_usd": .1}
    d = {"launches": [entry, {**entry, "volume_retained_since": 7200}]}
    assert remote.accrued(d) == pytest.approx(2.9)


def test_settled_legacy_compute_not_reestimated_during_cleanup(sandbox):
    r = resource()
    legacy = {k: v for k, v in r.items() if k not in ("compute_hourly_usd", "volume_hourly_usd")}
    atomic_json(remote.STATE / "ledger.json", {"launches": [{**legacy, "state": "exited", "stopped_at": 3600,
                    "estimated_usd": 1.2, "volume_retained_since": 3600., "volume_hourly_usd": .1}]})
    remote.update_ledger(r, state="removed", stopped_at=36000., estimated_usd=12,
                         volume_deleted_at=7200.)
    d = remote.ledger()
    assert d["launches"][0]["estimated_usd"] == 1.2
    assert remote.accrued(d) == pytest.approx(1.3)


def test_registry_ignores_historical_fetched_control(sandbox):
    current, historical = resource(), resource(launch_id="old", pod_id="pod-old", created_at=-1.)
    atomic_json(remote.STATE / "ledger.json", {"launches": [historical, current]})
    remote.register_resources(current)
    atomic_json(sandbox / "runs/run/resources.json", historical)
    assert remote.local_resources("run")["pod_id"] == "pod-one"


@pytest.mark.parametrize("field", ["run_id", "launch_id", "pod_id", "volume_id"])
def test_resource_ownership_requires_complete_tuple(sandbox, field):
    expected = resource()
    atomic_json(remote.STATE / "ledger.json", {"launches": [expected]})
    atomic_json(remote.STATE / "registry/run/resources.json", {**expected, field: "foreign"})
    with pytest.raises(RuntimeError, match="resource registry"):
        remote.local_resources("run")


def test_legacy_registry_migration_rejects_old_launch_control(sandbox):
    old, new = resource(created_at=0), resource(launch_id="new", pod_id="new", created_at=1)
    atomic_json(remote.STATE / "ledger.json", {"launches": [old, new]})
    atomic_json(sandbox / "runs/run/resources.json", old)
    with pytest.raises(RuntimeError, match="resource registry"):
        remote.local_resources("run")


def test_completed_resume_rejected_before_credentials_or_provider(sandbox, monkeypatch):
    atomic_json(sandbox / "runs/run/completion.json", {"complete": True})
    monkeypatch.setattr(remote, "load_secrets", lambda: pytest.fail("Must not read credentials"))
    monkeypatch.setattr(remote, "Runpod", lambda: pytest.fail("Must not inspect or provision cloud"))
    with pytest.raises(RuntimeError, match="Completed runs cannot resume"):
        remote.launch(resume_id="run")


def test_source_bundle_includes_subpackages_but_config_not_runtime_identity(sandbox):
    (sandbox / "model_collapse/cloud").mkdir(parents=True)
    (sandbox / "model_collapse/__init__.py").write_text("", encoding="utf-8")
    runtime = sandbox / "model_collapse/cloud/runtime.py"
    runtime.write_text("VALUE = 1\n", encoding="utf-8")
    (sandbox / "pyproject.toml").write_text("[project]\nname='tiny'\n", encoding="utf-8")
    (sandbox / "configs").mkdir()
    config_file = sandbox / "configs/full.yaml"
    config_file.write_text("generations: 10\n", encoding="utf-8")
    archive, first = remote.source_archive()
    with tarfile.open(archive) as bundle:
        names = bundle.getnames()
        manifest = json.load(bundle.extractfile("source_bundle.json"))
    assert "model_collapse/cloud/runtime.py" in names
    assert "pyproject.toml" in names
    assert not any(name.startswith("configs/") for name in names)
    assert manifest["runtime_sha256"] == first
    assert set(manifest["files"]) == set(names) - {"source_bundle.json"}
    archive_bytes = archive.read_bytes()
    config_file.write_text("generations: 5\n", encoding="utf-8")
    assert remote.source_archive()[1] == first
    assert archive.read_bytes() == archive_bytes
    runtime.write_text("VALUE = 2\n", encoding="utf-8")
    assert remote.source_archive()[1] != first


def test_cost_update_rejects_same_launch_id_with_foreign_volume(sandbox):
    remote.update_ledger(resource())
    with pytest.raises(RuntimeError, match="ownership tuple"):
        remote.update_ledger(resource(volume_id="foreign"), stopped_at=3600.)
