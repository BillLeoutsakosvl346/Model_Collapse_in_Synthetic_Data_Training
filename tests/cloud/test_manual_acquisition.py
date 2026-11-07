"""Historical manual tools retain failure shutdown before their watchdog starts."""
from types import SimpleNamespace

import pytest


def test_cross_corpus_registration_failure_stops_created_pod(tmp_path, monkeypatch):
    from scripts.operations import cross_corpus_evaluation as tool
    from model_collapse import remote
    pod = {}
    def request(method, path, body=None):
        if path == "/pods" and method == "GET":
            return {"pods": [pod] if pod else []}
        if path == "/pods" and method == "POST":
            pod.update(id="owned", status="RUNNING")
            return pod
        if path.startswith("/catalog/templates"):
            return {"templates": [{"id": "runpod-torch-v280", "image": tool.IMAGE}]}
        if path.startswith("/network-volumes"):
            return {"id": "volume", "size": 10, "dataCenter": "DC"}
        pytest.fail(f"Unexpected fake-provider call: {method} {path}")
    def stop(pod_id):
        assert pod_id == "owned"
        pod["status"] = "EXITED"
    api = SimpleNamespace(request=request, stop=stop, pod=lambda _: pod)
    monkeypatch.setattr(tool, "PROJECT", tmp_path)
    monkeypatch.setattr(tool, "prepare_inputs", lambda *a: {"evaluator_sha256": "source"})
    monkeypatch.setattr(remote, "Runpod", lambda: api)
    monkeypatch.setattr(remote, "preflight", lambda **kw: {"account": {"clientBalance": 30},
        "hugging_face": {"repos": {}}, "gpu_candidates": [{"gpu": tool.GPU, "hourly_usd": .5, "data_center": "DC"}]})
    monkeypatch.setattr(remote, "ledger", lambda: {"launches": [], "total_budget_usd": 50})
    monkeypatch.setattr(remote, "source_archive", lambda: ("unused", "runtime"))
    monkeypatch.setattr(remote, "register_key", lambda _: "unused")
    monkeypatch.setattr(remote, "register_resources", lambda _: (_ for _ in ()).throw(OSError("registry failed")))
    monkeypatch.setattr(remote, "update_ledger", lambda *a, **kw: {})
    monkeypatch.setattr(remote, "_provider_shutdown", lambda *a: (100, {}))
    with pytest.raises(OSError, match="registry failed"):
        tool.execute(tmp_path / "unused-bank")
    assert pod["status"] == "EXITED"


@pytest.mark.parametrize("failure", ["resources", "ledger"])
def test_shutdown_probe_bookkeeping_failure_stops_and_removes_pod(tmp_path, monkeypatch, failure):
    from scripts.operations import check_shutdown_timeout as tool
    pod, stopped = {}, []
    def request(method, path, body=None):
        if path == "/pods" and method == "GET":
            return {"pods": [pod] if pod else []}
        if path == "/pods" and method == "POST":
            pod.update(id="owned", status="RUNNING")
            return pod
        if path == "/pods/owned" and method == "DELETE":
            pod.clear()
            return
        if path.startswith("/catalog/templates"):
            return {"templates": [{"id": "runpod-torch-v280", "image": "test"}]}
        pytest.fail(f"Unexpected fake-provider call: {method} {path}")
    api = SimpleNamespace(request=request, stop=lambda name: stopped.append(name),
        account=lambda: {"isAutoPayEnabled": False, "enableBackupPayments": False, "clientBalance": 30},
        candidates=lambda: [{"gpu": "fake", "hourly_usd": .5, "data_center": "DC"}])
    monkeypatch.setattr(tool, "PROJECT", tmp_path)
    monkeypatch.setattr(tool, "Runpod", lambda: api)
    monkeypatch.setattr(tool, "ledger", lambda: {"launches": []})
    monkeypatch.setattr(tool, "accrued", lambda *a: 0)
    monkeypatch.setattr(tool, "register_key", lambda _: "unused")
    monkeypatch.setattr(tool, "source_archive", lambda: ("unused", "source"))
    write = tool.atomic_json
    def failing_write(path, value):
        if failure == "resources" and path.name == "resources.json":
            raise OSError("bookkeeping failed")
        return write(path, value)
    def update(*args, **kwargs):
        if failure == "ledger" and kwargs.get("state") == "provisioning":
            raise OSError("bookkeeping failed")
    monkeypatch.setattr(tool, "atomic_json", failing_write)
    monkeypatch.setattr(tool, "update_ledger", update)
    with pytest.raises(OSError, match="bookkeeping failed"):
        tool.probe()
    assert stopped == ["owned"] and not pod
