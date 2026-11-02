"""Durable Pod-disk placement, ownership, restart and deletion stay offline."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from model_collapse import remote
from model_collapse.common import PROJECT, atomic_json, config, read_json, validate_config
from model_collapse.runpod_api import Runpod
from model_collapse.workflow import initial_data
from model_collapse.workflow.profiles import protocol_fingerprint


def test_pod_disk_discovery_does_not_require_network_storage():
    api = object.__new__(Runpod)
    def request(method, path):
        assert method == "GET"
        if "datacenters" in path:
            return {"dataCenters": [{"id": "DC", "networkVolumeTypes": []}]}
        return {"gpus": [{"id": "NVIDIA L4", "manufacturer": "NVIDIA", "memory": 24,
                           "secure": True, "availability": "LOW", "price": {"secure": .59},
                           "dataCenters": [{"id": "DC", "availability": "LOW"}]}]}
    api.request = request
    assert api.candidates() == []
    assert api.candidates(network_storage=False)[0]["data_center"] == "DC"


def test_storage_is_part_of_acceptance_and_workflow_scope():
    root = PROJECT / "experiments/opt125m_wikitext2/continued_real_only_final5"
    full, smoke = (config(root / f"{kind}-pod-storage.yaml") for kind in ("full", "smoke"))
    assert protocol_fingerprint(full) == protocol_fingerprint(smoke)
    assert protocol_fingerprint(full) != protocol_fingerprint(config(root / "full.yaml"))
    with pytest.raises(ValueError, match="Unknown storage"):
        validate_config({**smoke, "storage_backend": "ephemeral"})
    with pytest.raises(ValueError, match="standalone"):
        validate_config({**smoke, "workflow": "paired"})


class Deployed(Exception):
    pass


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    data = {"total_budget_usd": 50, "launches": []}
    calls, updates = [], []
    pod = {}
    class API:
        def request(self, method, path, body=None):
            calls.append((method, path, copy.deepcopy(body)))
            assert not path.startswith("/network-volumes"), "Pod disks must never touch independent volumes"
            if method == "GET" and path == "/pods":
                return {"pods": [pod] if pod else []}
            if "catalog/templates" in path:
                return {"templates": [{"id": "runpod-torch-v280", "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"}]}
            if method == "POST" and path == "/pods":
                pod.update(id="owned-pod", status="RUNNING", name=body["name"], cost=.59,
                           gpu={"id": "NVIDIA L4"}, dataCenterId="DC", mounts=body["mounts"], ssh={"direct": {"host": "example"}})
                return copy.deepcopy(pod)
            if method == "POST" and path == "/pods/owned-pod/action":
                assert body == {"action": "start"}
                pod["status"] = "RUNNING"
                return copy.deepcopy(pod)
            if method == "DELETE" and path == "/pods/owned-pod":
                pod.clear()
                return None
            pytest.fail(f"Unexpected provider operation: {method} {path}")
        def pod(self, name):
            assert name == "owned-pod"
            return copy.deepcopy(pod)
        def stop(self, name):
            assert name == "owned-pod"
            pod["status"] = "EXITED"
        def account(self):
            return {"clientBalance": 20}
        def billing(self):
            return {}

    def ledger_update(resources, **changes):
        updates.append((copy.deepcopy(resources), changes))
        if not any(x["launch_id"] == resources["launch_id"] for x in data["launches"]):
            data["launches"].append(copy.deepcopy(resources))
        return data
    monkeypatch.setattr(remote, "PROJECT", tmp_path)
    monkeypatch.setattr(remote, "STATE", tmp_path / ".state")
    monkeypatch.setattr(remote, "now", lambda: 100)
    monkeypatch.setattr(remote, "Runpod", API)
    monkeypatch.setattr(remote, "load_secrets", lambda: None)
    monkeypatch.setattr(remote, "ledger", lambda: data)
    monkeypatch.setattr(remote, "accrued", lambda *args: 0)
    monkeypatch.setattr(remote, "campaign_accrued", lambda *args: 0)
    monkeypatch.setattr(remote, "preflight", lambda **kw: {"account": {"clientBalance": 20}, "hugging_face": {"repos": {}},
        "gpu_candidates": [{"gpu": "NVIDIA L4", "data_center": "DC", "hourly_usd": .59}]})
    monkeypatch.setattr(remote, "source_archive", lambda: (Path("unused"), "runtime"))
    monkeypatch.setattr(remote, "verify_start_checkpoint", lambda c: None)
    monkeypatch.setattr(remote, "register_key", lambda api: Path("unused"))
    monkeypatch.setattr(initial_data, "resolve_initial_dataset", lambda c: c)
    monkeypatch.setattr(remote, "reject_completed_cloud_resume", lambda *args: None)
    monkeypatch.setattr(remote, "ssh_connect", lambda *args: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(remote, "update_ledger", ledger_update)
    monkeypatch.setattr(remote, "command", lambda *a, **k: (_ for _ in ()).throw(Deployed()))
    return tmp_path, calls, updates, pod, API


def test_pod_disk_creation_and_resume_keep_exact_physical_owner(deployment):
    root, calls, updates, pod, _ = deployment
    with pytest.raises(Deployed):
        remote.launch(smoke=True, gpu="NVIDIA L4",
                      config_path=PROJECT / "experiments/opt125m_wikitext2/continued_real_only_final5/smoke-pod-storage.yaml")
    resource = updates[0][0]
    assert resource["pod_id"] == resource["volume_id"] == "owned-pod"
    assert resource["storage_backend"] == "pod_persistent"
    assert resource["network_volume_gb"] == 0 and resource["persistent_volume_gb"] == 30
    assert resource["volume_hourly_usd"] == pytest.approx(30 * .2 / 730)
    request = next(body for method, path, body in calls if method == "POST" and path == "/pods")
    assert request["mounts"] == {"persistent": {"path": "/workspace", "size": 30}}
    assert pod["status"] == "EXITED"
    with pytest.raises(Deployed):
        remote.launch(resume_id=resource["run_id"])
    assert len([1 for method, path, _ in calls if method == "POST" and path == "/pods"]) == 1
    assert any(path == "/pods/owned-pod/action" for _, path, _ in calls)
    assert updates[-1][0]["volume_id"] == "owned-pod"


def test_recovered_pod_disk_cleanup_deletes_only_the_owned_pod(deployment, monkeypatch):
    root, calls, updates, pod, _ = deployment
    with pytest.raises(Deployed):
        remote.launch(smoke=True, gpu="NVIDIA L4",
                      config_path=PROJECT / "experiments/opt125m_wikitext2/continued_real_only_final5/smoke-pod-storage.yaml")
    resource = updates[0][0]
    local = root / "runs" / resource["run_id"]
    atomic_json(local / "artifact_index.json", {"files": {}})
    atomic_json(local / "recovery_verified.json", {"metadata_only": False})
    monkeypatch.setattr(remote, "_provider_shutdown", lambda *a: (101, {}))
    result = remote.cleanup(resource["run_id"])
    assert result["pod_removed"] and result["volume_removed"]
    assert pod == {}
    assert [path for method, path, _ in calls if method == "DELETE"] == ["/pods/owned-pod"]


def test_recovery_missing_blocks_pod_deletion(deployment):
    _, calls, updates, _, _ = deployment
    with pytest.raises(Deployed):
        remote.launch(smoke=True, gpu="NVIDIA L4",
                      config_path=PROJECT / "experiments/opt125m_wikitext2/continued_real_only_final5/smoke-pod-storage.yaml")
    with pytest.raises(RuntimeError, match="Fetch and checksum"):
        remote.cleanup(updates[0][0]["run_id"])
    assert not any(method == "DELETE" for method, _, _ in calls)
