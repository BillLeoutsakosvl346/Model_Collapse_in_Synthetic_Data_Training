from types import SimpleNamespace

import pytest

from model_collapse.common import atomic_json, read_json
from model_collapse import remote, supervisor


def test_transfer_retries_transient_failure_but_never_permission_denial(monkeypatch):
    from model_collapse import hub
    from requests.exceptions import HTTPError
    monkeypatch.setattr(hub.time, "sleep", lambda _: None)
    calls = []
    def transient():
        calls.append(1)
        if len(calls) == 1:
            raise HTTPError(response=SimpleNamespace(status_code=503))
        return "verified"
    assert hub.retry_transfer(transient) == "verified"
    assert len(calls) == 2
    def denied():
        calls.append(1)
        raise HTTPError(response=SimpleNamespace(status_code=403))
    with pytest.raises(HTTPError):
        hub.retry_transfer(denied)
    assert len(calls) == 3


def test_expired_independent_watchdog_calls_provider_stop(tmp_path, monkeypatch):
    stopped = []
    atomic_json(tmp_path / "resources.json", {"created_at": 0, "session_allowance_usd": .01,
                "guard_hourly_usd": 1, "pod_id": "owned", "repos": {"dataset": "test/private"}})
    monkeypatch.setattr(supervisor, "Runpod", lambda: SimpleNamespace(stop=lambda pod: stopped.append(pod)))
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kwargs: SimpleNamespace(upload_file=lambda **kwargs: None))
    supervisor.watchdog(tmp_path)
    assert stopped == ["owned"]
    assert read_json(tmp_path / "watchdog_exit.json")["reason"] == "budget_exhausted"


def test_worker_failure_requests_shutdown(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(poll=lambda: 1, returncode=1))
    with pytest.raises(RuntimeError, match="Worker failed"):
        supervisor.supervise(tmp_path)
    assert read_json(tmp_path / "stop_request.json")["reason"] == "failure"
    assert read_json(tmp_path / "status.json")["state"] == "failed"


def test_exhausted_task_budget_prevents_provisioning(monkeypatch):
    calls = []
    class API:
        def request(self, method, path, body=None):
            calls.append(method)
            assert method == "GET"
            return {"pods": []}
        def candidates(self):
            return [{"gpu": "test", "data_center": "test", "hourly_usd": .2}]
    monkeypatch.setattr(remote, "Runpod", API)
    monkeypatch.setattr(remote, "preflight", lambda **kw: {"account": {"clientBalance": 30}})
    monkeypatch.setattr(remote, "source_archive", lambda: ("unused", "hash"))
    monkeypatch.setattr(remote, "ledger", lambda: {"launches": [{"kind": "smoke", "estimated_usd": 25, "created_at": 0, "guard_hourly_usd": .2}]})
    with pytest.raises(RuntimeError, match="Insufficient budget"):
        remote.launch()
    assert "POST" not in calls


def test_cleanup_only_removes_registry_owned_resources(tmp_path, monkeypatch):
    root = tmp_path / "runs/test"
    atomic_json(root / "recovery_verified.json", {"verified_files": 1})
    atomic_json(root / "artifact_index.json", {"files": {}})
    records = [{"id": "old", "status": "EXITED"}, {"id": "new", "status": "EXITED"}, {"id": "foreign", "status": "RUNNING"}]
    deleted = []
    class API:
        def request(self, method, path, body=None):
            if method == "DELETE":
                deleted.append(path)
                records[:] = [p for p in records if path != "/pods/" + p["id"]]
                return None
            return {"pods": records[:]} if path == "/pods" else {"networkVolumes": []}
        def account(self):
            return {"clientBalance": 30}
        def billing(self):
            return {}
    monkeypatch.setattr(remote, "PROJECT", tmp_path)
    monkeypatch.setattr(remote, "Runpod", API)
    monkeypatch.setattr(remote, "local_resources", lambda _: {"run_id": "test", "pod_id": "new", "volume_id": "vol",
                        "starting_balance_usd": 30, "created_at": 0, "guard_hourly_usd": .2})
    monkeypatch.setattr(remote, "ledger", lambda: {"launches": [{"run_id": "test", "pod_id": p} for p in ("old", "new")]})
    monkeypatch.setattr(remote, "update_ledger", lambda *a, **kw: {})
    remote.cleanup("test")
    assert set(deleted) == {"/pods/old", "/pods/new", "/network-volumes/vol"}
    assert records == [{"id": "foreign", "status": "RUNNING"}]


def test_concurrent_launch_is_rejected_before_provisioning(tmp_path, monkeypatch):
    from filelock import FileLock, Timeout
    monkeypatch.setattr(remote, "STATE", tmp_path)
    monkeypatch.setattr(remote, "_launch", lambda *args: pytest.fail("Provisioning must not run while another launcher holds the lock"))
    with FileLock(str(tmp_path / "launch.lock")):
        with pytest.raises(Timeout):
            remote.launch()
