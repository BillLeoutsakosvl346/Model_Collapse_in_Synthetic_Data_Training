import copy
import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import pytest

from model_collapse import remote
from model_collapse.common import PROJECT, atomic_json, config, read_json, sha256
from model_collapse.mixing import certificate_name
from model_collapse.workflow import mixed_campaign as campaign
from model_collapse.workflow.profiles import CAMPAIGN_CHECKS, mixture_checks, protocol_fingerprint


class PaidBoundary(Exception):
    pass


def certificate(c):
    proof = {f"{arm}_{kind}_sha256": ("a" if kind == "tokens" else "b") * 64
             for arm in ("continued", "reset") for kind in ("tokens", "selection")}
    return {"workflow": "mixed_campaign", "source_code_sha256": "runtime",
            "protocol_fingerprint": protocol_fingerprint(c), "checks": {k: True for k in CAMPAIGN_CHECKS},
            "unique_training_stages": 6, "executed_epochs": 30, "shared20_mixture": proof,
            "child_certificates": {name: {"source_code_sha256": "runtime", "protocol_fingerprint": protocol_fingerprint(job),
                "checks": {k: True for k in mixture_checks(job, include_operational=False)}} for name, job in c["jobs"].items()}}


@pytest.fixture
def admission(tmp_path, monkeypatch):
    c = config(PROJECT / "comparisons/mixed_fraction_final5/full.yaml")
    data = {"total_budget_usd": 45., "launches": []}
    class API:
        def candidates(self):
            return [{"gpu": c["required_gpu"], "data_center": "EU-RO-1", "hourly_usd": .57}]
        def request(self, method, path, body=None):
            assert method == "GET", "Admission cannot provision a paid resource"
            return {"pods": []} if path == "/pods" else {"templates": [{"id": "runpod-torch-v280", "image": c["required_image"]}]}
    monkeypatch.setattr(remote, "STATE", tmp_path / "state")
    monkeypatch.setattr(remote, "PROJECT", tmp_path)
    monkeypatch.setattr(remote, "Runpod", API)
    monkeypatch.setattr(remote, "load_secrets", lambda: None)
    monkeypatch.setattr(remote, "ledger", lambda: data)
    monkeypatch.setattr(remote, "accrued", lambda d, kind=None: .5 if kind == "smoke" else 25.)
    monkeypatch.setattr(remote, "campaign_accrued", lambda *a: 0.)
    monkeypatch.setattr(remote, "preflight", lambda **k: {"account": {"clientBalance": 100}, "hugging_face": {"repos": {"model": "test/m", "dataset": "test/d"}}})
    monkeypatch.setattr(remote, "source_archive", lambda: ("unused", "runtime"))
    monkeypatch.setattr(remote, "verify_start_checkpoint", lambda *a: None)
    monkeypatch.setattr(campaign, "resolve_campaign_inputs", lambda c: c)
    monkeypatch.setattr(remote, "register_key", lambda *a: (_ for _ in ()).throw(PaidBoundary()))
    atomic_json(remote.STATE / certificate_name(c, "runtime"), certificate(c))
    atomic_json(remote.STATE / "benchmark.json", {"training_target_tokens_per_second": 34000,
                "generation_new_tokens_per_second": 1400,
                "full_split_blocks": {"train": 37946, "validation": 3922, "test": 4494}})
    return c, data, API


def test_campaign_uses_explicit_operational_ceiling_without_changing_legacy_default(admission):
    _, _, _ = admission
    assert remote.TOTAL_BUDGET == 25.
    with pytest.raises(PaidBoundary):
        remote.launch(config_path=PROJECT / "comparisons/mixed_fraction_final5/full.yaml")


def test_current_ceiling_blocks_full_campaign_before_provision(admission):
    _, data, _ = admission
    data["total_budget_usd"] = 30.
    with pytest.raises(RuntimeError, match="deadline|estimate"):
        remote.launch(config_path=PROJECT / "comparisons/mixed_fraction_final5/full.yaml")


def test_insufficient_padded_deadline_blocks_even_when_raw_estimate_fits(admission, monkeypatch):
    from model_collapse.analysis import forecasts
    _, data, _ = admission
    data["total_budget_usd"] = 30.
    monkeypatch.setattr(forecasts, "forecast_campaign", lambda *a: {
        "estimated_hours": 5., "estimated_compute_and_running_storage_usd": 2.9})
    with pytest.raises(RuntimeError, match="padded watchdog deadline"):
        remote.launch(config_path=PROJECT / "comparisons/mixed_fraction_final5/full.yaml")


def test_historical_storage_is_reserved_before_admission(admission, monkeypatch):
    from model_collapse.analysis import forecasts
    monkeypatch.setattr(forecasts, "forecast_campaign", lambda *a: {
        "estimated_hours": 20., "estimated_compute_and_running_storage_usd": 10.})
    monkeypatch.setattr(remote, "historical_running_storage", lambda *a, **k: 7.)
    with pytest.raises(RuntimeError, match="historical storage"):
        remote.launch(config_path=PROJECT / "comparisons/mixed_fraction_final5/full.yaml")


def test_campaign_cannot_reuse_legacy_acceptance(admission):
    c, _, _ = admission
    (remote.STATE / certificate_name(c, "runtime")).unlink()
    atomic_json(remote.STATE / "validation_continued_mixed.json", certificate(c))
    with pytest.raises(RuntimeError, match="Validate the complete smoke"):
        remote.launch(config_path=PROJECT / "comparisons/mixed_fraction_final5/full.yaml")


def test_campaign_attempt_cap_includes_all_validation_retries(admission, monkeypatch):
    monkeypatch.setattr(remote, "campaign_accrued", lambda *a: .74)
    with pytest.raises(RuntimeError, match="Insufficient budget"):
        remote.launch(smoke=True, config_path=PROJECT / "comparisons/mixed_fraction_final5/smoke.yaml")


def test_campaign_keeps_single_active_pod_rule(admission, monkeypatch):
    _, _, API = admission
    original = API.request
    def active(self, method, path, body=None):
        if path == "/pods": return {"pods": [{"status": "RUNNING"}]}
        return original(self, method, path, body)
    monkeypatch.setattr(API, "request", active)
    with pytest.raises(RuntimeError, match="existing Pod"):
        remote.launch(config_path=PROJECT / "comparisons/mixed_fraction_final5/full.yaml")


def test_campaign_volume_creation_is_exactly_sixty_gb(admission, monkeypatch):
    _, _, API = admission
    original, writes = API.request, []
    def boundary(self, method, path, body=None):
        if method == "POST" and path == "/network-volumes":
            writes.append(body)
            raise PaidBoundary()
        return original(self, method, path, body)
    monkeypatch.setattr(API, "request", boundary)
    monkeypatch.setattr(remote, "register_key", lambda *a: Path("unused"))
    with pytest.raises(PaidBoundary):
        remote.launch(smoke=True, config_path=PROJECT / "comparisons/mixed_fraction_final5/smoke.yaml")
    assert len(writes) == 1 and writes[0]["size"] == 60 and writes[0]["type"] == "STANDARD"


@pytest.mark.parametrize("value", [None, True, -1, float("nan"), float("inf")])
def test_campaign_budget_requires_explicit_finite_authorization(value):
    with pytest.raises(RuntimeError, match="explicit authorized"):
        remote.campaign_budget({"total_budget_usd": value})


def test_historical_storage_forecast_excludes_current_volume_without_double_charging(monkeypatch):
    monkeypatch.setattr(remote, "now", lambda: 100.)
    data = {"launches": [], "volumes": {
        "historical": {"charge_started_at": 0., "hourly_usd": .01},
        "current": {"charge_started_at": 0., "hourly_usd": .006},
        "deleted": {"charge_started_at": 0., "hourly_usd": 1., "deleted_at": 50.},
    }}
    assert remote.historical_running_storage(data, 10, exclude_volume="current") == pytest.approx(.1)


@pytest.mark.parametrize("value", ["2026-10-31T22:55:32Z", "Exited by user: Sat Oct 31 2026 22:55:32 GMT+0000 (Coordinated Universal Time)"])
def test_provider_shutdown_time_uses_provider_clock(value, monkeypatch):
    expected = dt.datetime(2026, 10, 31, 22, 55, 32, tzinfo=dt.timezone.utc).timestamp()
    monkeypatch.setattr(remote, "now", lambda: expected + 3600)
    pod = {"id": "pod", "status": "EXITED", "lastStatusChange": value}
    timestamp, proof = remote._provider_shutdown(None, pod, {"pod_id": "pod", "created_at": expected - 1000})
    assert timestamp == expected and proof["observed_at"] == expected + 3600


def test_provider_clock_fallback_reads_graphql_and_rejects_wrong_owner(monkeypatch):
    monkeypatch.setattr(remote, "now", lambda: 5000.)
    class API:
        def request(self, method, path, body, *, graphql):
            assert method == "POST" and graphql and "lastStatusChange" in body["query"]
            return {"data": {"pod": {"id": "pod", "desiredStatus": "EXITED", "lastStatusChange": "1970-01-01T00:50:00Z"}}}
    timestamp, proof = remote._provider_shutdown(API(), {"id": "pod", "status": "EXITED"}, {"pod_id": "pod", "created_at": 2000.})
    assert timestamp == 3000.
    with pytest.raises(RuntimeError, match="verified shutdown timestamp"):
        remote._provider_shutdown(API(), {"id": "wrong", "status": "EXITED", "stoppedAt": "1970-01-01T00:50:00Z"}, {"pod_id": "pod", "created_at": 2000.})


def test_stopped_campaign_status_closes_at_provider_time_not_observation(tmp_path, monkeypatch):
    import huggingface_hub
    resources = {"run_id": "campaign", "pod_id": "pod", "created_at": 2000., "workflow": "mixed_campaign",
                 "repos": {"dataset": "test/data"}, "volume_hourly_usd": .006}
    class API:
        def pod(self, _):
            return {"id": "pod", "status": "EXITED", "cost": .57, "stoppedAt": "1970-01-01T00:50:00Z"}
    updates = []
    monkeypatch.setattr(remote, "Runpod", API)
    monkeypatch.setattr(remote, "PROJECT", tmp_path)
    monkeypatch.setattr(remote, "local_resources", lambda _: resources)
    monkeypatch.setattr(remote, "update_ledger", lambda *a, **kw: updates.append(kw))
    monkeypatch.setattr(remote, "now", lambda: 5000.)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    result = remote.status("campaign")
    assert updates[0]["stopped_at"] == updates[0]["volume_retained_since"] == 3000.
    assert result["provider_shutdown"]["observed_at"] == 5000.
    assert read_json(tmp_path / "runs/campaign/provider_shutdown.json")["provider_shutdown_epoch"] == 3000.


def test_late_campaign_resume_is_admitted_for_verified_remaining_job(admission, monkeypatch):
    c, data, _ = admission
    data["total_budget_usd"] = 35.
    run_id = "unfinished-campaign"
    atomic_json(remote.PROJECT / "runs" / run_id / "config.json", c)
    previous = {"run_id": run_id, "repos": {"model": "test/m", "dataset": "test/d"},
                "code_sha256": "runtime", "data_center": "EU-RO-1", "volume_id": "volume"}
    monkeypatch.setattr(remote, "local_resources", lambda _: previous)
    monkeypatch.setattr(remote, "reject_completed_cloud_resume", lambda *a: None)
    monkeypatch.setattr(campaign, "verified_resume_jobs", lambda *a: {
        "sealed_completed_jobs": ["reset50", "continued20"],
        "parent_artifact_index_sha256": "a" * 64})
    with pytest.raises(PaidBoundary):
        remote.launch(resume_id=run_id)
    # The original ten-cycle job configurations are never shortened to gain admission.
    saved = read_json(remote.PROJECT / "runs" / run_id / "config.json")
    assert all(job["generations"] == 10 for job in saved["jobs"].values())


@pytest.mark.parametrize("new_epochs,finite,failed,accept", [(0, False, False, False), (1, True, False, True), (1, True, True, False)])
def test_detached_campaign_requires_finite_new_training_not_imported_history(tmp_path, monkeypatch, new_epochs, finite, failed, accept):
    clock = [1000.]
    r = {"run_id": "campaign", "launch_id": "launch", "pod_id": "pod", "volume_id": "volume",
         "created_at": 1000., "session_allowance_usd": .65, "guard_hourly_usd": .66,
         "remote_root": "/workspace/runs/campaign", "workflow": "mixed_campaign"}
    atomic_json(tmp_path / "runs/campaign/config.json", {"workflow": "mixed_campaign"})
    def observe(_):
        t = clock[0]
        return {"status": {"state": "failed" if failed else "running", "phase": "reset50", "active_child": "campaign-reset-mixed50"},
                "child_status": {"state": "running", "generation": 2, "stage": "training"},
                "event_bytes": int(t), "epochs": 5 + new_epochs, "new_epochs": new_epochs,
                "finite_new_training": finite, "supervisor": {"time": t}, "watchdog": {"time": t}}
    class Client:
        def open_sftp(self): return object()
        def close(self): pass
    monkeypatch.setattr(remote, "PROJECT", tmp_path)
    monkeypatch.setattr(remote, "local_resources", lambda _: r)
    monkeypatch.setattr(remote, "_detached_observation", observe)
    monkeypatch.setattr(remote, "now", lambda: clock[0])
    monkeypatch.setattr(remote.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(remote, "Runpod", lambda: SimpleNamespace(pod=lambda _: {}))
    monkeypatch.setattr(remote, "ssh_connect", lambda *a: Client())
    monkeypatch.setattr(remote, "upload_json", lambda *a: None)
    monkeypatch.setattr(remote, "command", lambda *a, **k: "")
    if accept:
        receipt = remote.verify_detached("campaign", max_wait_seconds=7)
        assert receipt["new_training_after_import_verified"]
        assert receipt["observations"][1]["new_epochs"] == 1
    else:
        with pytest.raises(RuntimeError, match="No real|failed or paused"):
            remote.verify_detached("campaign", max_wait_seconds=7)
        assert not (tmp_path / "runs/campaign/detached_verification.json").exists()


def test_backup_only_resume_can_handoff_verified_completion_without_new_epochs(tmp_path, monkeypatch):
    clock = 5000.
    r = {"run_id": "campaign", "launch_id": "launch", "pod_id": "pod", "volume_id": "volume",
         "created_at": 2000., "session_allowance_usd": 1., "guard_hourly_usd": .66,
         "volume_hourly_usd": .006, "workflow": "mixed_campaign"}
    root = tmp_path / "runs/campaign"
    atomic_json(root / "config.json", {"workflow": "mixed_campaign"})
    atomic_json(root / "continued20_seal.json", {"source": "immutable"})
    atomic_json(root / "completion.json", {"complete": True, "workflow": "mixed_campaign",
        "source20_seal_sha256": sha256(root / "continued20_seal.json")})
    atomic_json(root / "artifact_index.json", {"files": {"completion.json": {"sha256": "verified by pinned fetch"}}})
    class API:
        def pod(self, _):
            return {"id": "pod", "status": "EXITED", "stoppedAt": "1970-01-01T01:20:00Z"}
    updates = []
    monkeypatch.setattr(remote, "PROJECT", tmp_path)
    monkeypatch.setattr(remote, "now", lambda: clock)
    monkeypatch.setattr(remote, "local_resources", lambda _: r)
    monkeypatch.setattr(remote, "Runpod", API)
    monkeypatch.setattr(campaign, "verified_resume_jobs", lambda *a: {"sealed_completed_jobs": list(campaign.JOB_ORDER)})
    monkeypatch.setattr(remote, "update_ledger", lambda *a, **kw: updates.append(kw))
    receipt = remote.verify_campaign_terminal("campaign", max_wait_seconds=10)
    assert receipt["state"] == "complete"
    assert receipt["provider_shutdown"]["provider_shutdown_epoch"] == updates[0]["stopped_at"] == 4800.
    assert not (root / "detached_verification.json").exists()


def test_terminal_repair_rejects_missing_completed_backup(tmp_path, monkeypatch):
    r = {"run_id": "campaign", "pod_id": "pod", "created_at": 2000., "session_allowance_usd": 1., "guard_hourly_usd": .66}
    root = tmp_path / "runs/campaign"
    atomic_json(root / "config.json", {"workflow": "mixed_campaign"})
    atomic_json(root / "artifact_index.json", {"files": {}})
    monkeypatch.setattr(remote, "PROJECT", tmp_path)
    monkeypatch.setattr(remote, "now", lambda: 5000.)
    monkeypatch.setattr(remote, "local_resources", lambda _: r)
    monkeypatch.setattr(remote, "Runpod", lambda: SimpleNamespace(pod=lambda _: {"status": "EXITED"}))
    monkeypatch.setattr(campaign, "verified_resume_jobs", lambda *a: {"sealed_completed_jobs": list(campaign.JOB_ORDER)})
    with pytest.raises(RuntimeError, match="completed private backup"):
        remote.verify_campaign_terminal("campaign", max_wait_seconds=10)


def test_transient_observation_read_is_retried_but_permanent_failure_is_bounded(monkeypatch):
    clock, calls = [1000.], []
    monkeypatch.setattr(remote, "now", lambda: clock[0])
    monkeypatch.setattr(remote.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    def transient():
        calls.append(1)
        if len(calls) < 3: raise OSError("temporary SSL EOF")
        return {"new_epochs": 1, "finite_new_training": True}
    assert remote._transport_read(transient, 1020.)["new_epochs"] == 1
    assert len(calls) == 3
    calls.clear()
    def permanent():
        calls.append(1)
        raise EOFError("connection permanently unavailable")
    with pytest.raises(EOFError): remote._transport_read(permanent, 1040.)
    assert len(calls) == 4


@pytest.mark.parametrize("error", [remote.paramiko.AuthenticationException("denied"),
    remote.paramiko.BadHostKeyException("host", "actual", "expected"), RuntimeError("Runpod GET /pods: HTTP 401")])
def test_authentication_and_changed_host_key_reads_are_never_retried(monkeypatch, error):
    calls = []
    monkeypatch.setattr(remote, "now", lambda: 1000.)
    def fail():
        calls.append(1)
        raise error
    with pytest.raises(type(error)): remote._transport_read(fail, 2000.)
    assert len(calls) == 1
