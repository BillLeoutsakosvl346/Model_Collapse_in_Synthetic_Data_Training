import copy
import sys
from types import SimpleNamespace

import pytest

from model_collapse.common import PROJECT, atomic_json, config, digest, event, read_json, sha256
from model_collapse.workflow import mixed_campaign as campaign
from model_collapse.workflow.profiles import CAMPAIGN_CHECKS, mixture_checks, protocol_fingerprint


@pytest.fixture
def session(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    c = config(PROJECT / "comparisons/mixed_fraction_final5/smoke.yaml")
    resources = {"run_id": root.name, "launch_id": "launch", "pod_id": "pod", "volume_id": "volume",
                 "created_at": 123, "session_allowance_usd": .65, "guard_hourly_usd": .66,
                 "code_sha256": "runtime", "workflow": "mixed_campaign",
                 "repos": {"model": "test/models", "dataset": "test/data"}}
    atomic_json(root / "config.json", c)
    atomic_json(root / "resources.json", resources)
    monkeypatch.setattr(campaign.shutil, "disk_usage", lambda *a: SimpleNamespace(free=60_000_000_000))
    return root, c, resources


def shared_proof():
    return {"continued_tokens_sha256": "a" * 64, "reset_tokens_sha256": "a" * 64,
            "continued_selection_sha256": "b" * 64, "reset_selection_sha256": "b" * 64}


def install_workers(session, monkeypatch):
    from model_collapse import hub, supervisor
    root, c, resources = session
    calls = []

    def worker(child, **kwargs):
        calls.append(child.name)
        job = read_json(child / "config.json")
        atomic_json(child / "status.json", {"state": "complete"})
        atomic_json(child / "completion.json", {"complete": True})
        for g in (1, 2, 3):
            atomic_json(child / f"seed_42/gen_{g:02d}/training.json", {"selected_epoch": 5, "executed_epochs": 5})
        atomic_json(child / "remote_validation.json", {
            "source_code_sha256": "runtime", "protocol_fingerprint": protocol_fingerprint(job),
            "checks": {k: True for k in mixture_checks(job, include_operational=False)}})
        atomic_json(child / "evidence.json", {"job": child.name})
        atomic_json(child / "artifact_index.json", {"files": {"evidence.json": sha256(child / "evidence.json")}})
        snapshot = {"run_id": child.name, "repos": resources["repos"],
                    "revisions": {"model": "a" * 40, "dataset": "b" * 40},
                    "artifact_index_sha256": sha256(child / "artifact_index.json")}
        atomic_json(child / "backup_snapshot.json", snapshot)
        campaign.record_child_snapshot(child)

    def seal(root, c):
        reset = copy.deepcopy(c["jobs"]["reset20"])
        reset["branch_source"]["run_id"] = campaign.job_roots(root)["continued20"].name
        value = {"snapshot": read_json(root / "child_snapshots.json")["continued20"],
                 "branch_source": reset["branch_source"]}
        campaign._write_fixed(root / "continued20_seal.json", value)
        return reset, value

    monkeypatch.setattr(supervisor, "run_worker", worker)
    monkeypatch.setattr(hub, "backup", lambda *a, **k: {})
    monkeypatch.setattr(campaign, "verify_completed_child", lambda child, c: read_json(child / "backup_snapshot.json"))
    monkeypatch.setattr(campaign, "seal_continued20", seal)
    monkeypatch.setattr(campaign, "_verify_shared20", lambda *a: shared_proof())
    monkeypatch.setattr(campaign, "report_campaign", lambda *a, **k: {})
    return calls


def test_profiles_volume_and_fixed_endpoints_match():
    smoke, full = [config(PROJECT / f"comparisons/mixed_fraction_final5/{kind}.yaml") for kind in ("smoke", "full")]
    assert protocol_fingerprint(smoke) == protocol_fingerprint(full)
    assert smoke["network_volume_gb"] == full["network_volume_gb"] == 60
    assert [smoke["jobs"][name]["generations"] for name in campaign.JOB_ORDER] == [3, 2, 3]
    assert [full["jobs"][name]["generations"] for name in campaign.JOB_ORDER] == [10, 10, 10]


@pytest.mark.parametrize("change", ["fraction", "mode", "epochs", "endpoint", "gpu", "volume", "source", "settings"])
def test_invalid_campaign_contract_rejected(session, change):
    _, c, _ = session
    if change == "fraction": c["jobs"]["reset20"]["synthetic_fraction"] = .5
    if change == "mode": c["jobs"]["reset50"]["initialization"] = "continued"
    if change == "epochs": c["jobs"]["continued20"]["epochs"] = 1
    if change == "endpoint": c["jobs"]["continued20"]["generations"] = 3
    if change == "gpu": c["jobs"]["reset20"]["required_gpu"] = "other"
    if change == "volume": c["network_volume_gb"] = 30
    if change == "source": c["jobs"]["reset20"]["branch_source"]["run_id"] = "historical"
    if change == "settings": c["jobs"]["reset20"]["seed"] = 123
    with pytest.raises((ValueError, RuntimeError)):
        campaign.validate_campaign_config(c)


def test_three_jobs_and_both_handoffs_resume_without_retraining(session, monkeypatch):
    from model_collapse.engine import ValidationInterruption
    root, c, resources = session
    calls = install_workers(session, monkeypatch)
    with pytest.raises(ValidationInterruption, match="handoff seal"):
        campaign.run_campaign(root, validation=True)
    original = sha256(campaign.job_roots(root)["reset50"] / "resources.json")
    event(root, "validation_worker_resumed", interruption=1)
    with pytest.raises(ValidationInterruption, match="handoff seal"):
        campaign.run_campaign(root, validation=True)
    assert sha256(campaign.job_roots(root)["reset50"] / "resources.json") == original
    event(root, "validation_worker_resumed", interruption=2)
    campaign.run_campaign(root, validation=True)
    assert calls == [campaign.job_roots(root)[name].name for name in campaign.JOB_ORDER]
    assert read_json(root / "completion.json")["unique_training_stages"] == 6
    assert read_json(root / "completion.json")["executed_epochs"] == 30
    assert read_json(root / "remote_validation.json")["shared20_mixture"] == shared_proof()
    assert read_json(root / "status.json")["state"] == "complete"


def test_budget_pause_cannot_advance_to_next_job(session, monkeypatch):
    from model_collapse import supervisor
    root, c, resources = session
    calls = install_workers(session, monkeypatch)
    def pause(child, **kwargs):
        calls.append(child.name)
        atomic_json(child / "status.json", {"state": "budget_paused"})
    monkeypatch.setattr(supervisor, "run_worker", pause)
    campaign.run_campaign(root)
    assert calls == [campaign.job_roots(root)["reset50"].name]
    assert read_json(root / "status.json")["state"] == "budget_paused"
    assert not (campaign.job_roots(root)["continued20"] / "config.json").exists()


def test_changed_completed_evidence_blocks_campaign(session, monkeypatch):
    from model_collapse.engine import ValidationInterruption
    root, c, _ = session
    calls = install_workers(session, monkeypatch)
    with pytest.raises(ValidationInterruption): campaign.run_campaign(root, validation=True)
    child = campaign.job_roots(root)["reset50"]
    atomic_json(child / "evidence.json", {"corrupt": True})
    event(root, "validation_worker_resumed", interruption=1)
    with pytest.raises(RuntimeError, match="checksum|changed|corrupt"):
        campaign.run_campaign(root, validation=True)
    assert len(calls) == 1


def test_low_disk_refuses_before_child_creation(session, monkeypatch):
    root, _, _ = session
    monkeypatch.setattr(campaign.shutil, "disk_usage", lambda *a: SimpleNamespace(free=39_999_999_999))
    with pytest.raises(RuntimeError, match="40 GB"):
        campaign.run_campaign(root)
    assert not (root / "children.json").exists()


@pytest.mark.parametrize("key", ["run_id", "launch_id", "pod_id", "volume_id", "owner_campaign_id", "logical_run_id"])
def test_complete_owner_tuple_required(session, key):
    root, c, resources = session
    child = campaign.job_roots(root)["reset50"]
    campaign._session(root, child, c["jobs"]["reset50"], owner_key="owner_campaign_id")
    changed = read_json(child / "resources.json")
    changed[key] = "wrong"
    with pytest.raises(RuntimeError, match="ownership|association"):
        campaign._check_child_owner(root, child, changed)


def test_prior_provider_session_requires_explicit_registration_and_completion(session):
    root, c, resources = session
    child = campaign.job_roots(root)["reset50"]
    campaign._session(root, child, c["jobs"]["reset50"], owner_key="owner_campaign_id")
    old = read_json(child / "resources.json")
    previous = {k: old[k] for k in ("run_id", "launch_id", "pod_id", "volume_id")}
    resources.update(launch_id="newlaunch", pod_id="newpod", provider_sessions=[previous])
    atomic_json(root / "resources.json", resources)
    with pytest.raises(RuntimeError, match="association"):
        campaign._check_child_owner(root, child)
    atomic_json(child / "completion.json", {"complete": True})
    campaign._check_child_owner(root, child)
    resources["provider_sessions"] = []
    atomic_json(root / "resources.json", resources)
    with pytest.raises(RuntimeError, match="association"):
        campaign._check_child_owner(root, child)


def test_sealed_child_backup_cannot_be_replaced(session, monkeypatch):
    from model_collapse.engine import ValidationInterruption
    root, _, _ = session
    install_workers(session, monkeypatch)
    with pytest.raises(ValidationInterruption): campaign.run_campaign(root, validation=True)
    child = campaign.job_roots(root)["reset50"]
    snapshot = read_json(child / "backup_snapshot.json")
    snapshot["revisions"]["dataset"] = "c" * 40
    atomic_json(child / "backup_snapshot.json", snapshot)
    with pytest.raises(RuntimeError, match="sealed"):
        campaign.record_child_snapshot(child)


def test_all_three_recoveries_required_before_cleanup(session):
    root, _, _ = session
    atomic_json(root / "recovery_verified.json", {"children": {"reset50": {}}})
    with pytest.raises(RuntimeError, match="All three"):
        campaign.verify_campaign_recovery(root)


def test_worker_dispatches_campaign_and_ordinary_children(session, monkeypatch):
    from model_collapse import __main__, pipeline
    root, c, _ = session
    calls = []
    monkeypatch.setattr(campaign, "run_campaign", lambda *a: calls.append("parent"))
    monkeypatch.setattr(pipeline, "run", lambda *a: calls.append("child"))
    monkeypatch.setattr(sys, "argv", ["model_collapse", "worker", "--root", str(root)])
    __main__.main()
    child = campaign.job_roots(root)["reset50"]
    campaign._session(root, child, c["jobs"]["reset50"], owner_key="owner_campaign_id")
    monkeypatch.setattr(sys, "argv", ["model_collapse", "worker", "--root", str(child)])
    __main__.main()
    assert calls == ["parent", "child"]


def test_forecast_counts_new_work_and_both_handoffs():
    from model_collapse.analysis.forecasts import forecast_campaign
    bench = {"training_target_tokens_per_second": 30000, "generation_new_tokens_per_second": 1300,
             "full_split_blocks": {"train": 37946, "validation": 3922, "test": 4494}}
    value = forecast_campaign(bench, 10, .58)
    assert value["new_training_stages"] == 28 and value["executed_epochs"] == 140
    assert value["new_full_generation_stages"] == 25
    assert value["jobs"]["reset50"]["new_training_stages"] == 9
    assert value["estimated_hours"] == sum(x["estimated_hours"] for x in value["jobs"].values()) + .5


def test_resume_forecast_excludes_sealed_jobs_and_keeps_endpoints():
    from model_collapse.analysis.forecasts import forecast_campaign
    bench = {"training_target_tokens_per_second": 30000, "generation_new_tokens_per_second": 1300,
             "full_split_blocks": {"train": 37946, "validation": 3922, "test": 4494}}
    value = forecast_campaign(bench, 10, .58, completed_jobs=("reset50", "continued20"))
    assert value["new_training_stages"] == 9 and value["executed_epochs"] == 45
    assert value["new_full_generation_stages"] == 8
    assert value["jobs"]["reset50"]["estimated_hours"] == value["jobs"]["continued20"]["estimated_hours"] == 0
    assert value["jobs"]["reset20"]["synthetic_generations"] == 10


def test_campaign_report_records_renderer_and_original_runtime(session, tmp_path, monkeypatch):
    from model_collapse.analysis import reporting
    root, c, _ = session
    atomic_json(root / "manifest.json", {"code_sha256": "recorded-training-runtime"})
    for name, child in campaign.job_roots(root).items():
        atomic_json(child / "config.json", c["jobs"][name])
        atomic_json(child / "seed_42/gen_00/metrics.json", {"generation": 0})
    monkeypatch.setattr(reporting, "_rows", lambda *a: ([{"generation": 0, "test_perplexity": 33.89,
        "stage_origin": "current_run"}], {}, {}))
    monkeypatch.setattr(reporting, "report", lambda *a: {})
    out = tmp_path / "rendered"
    campaign.report_campaign(root, out=out)
    manifest = read_json(out / "analysis_manifest.json")
    from model_collapse.analysis import campaign as renderer
    assert manifest["renderer"]["sha256"] == sha256(renderer.__file__)
    assert manifest["source_runtime_sha256"] == "recorded-training-runtime"
    assert manifest["inputs"][root.name + "/manifest.json"]["sha256"] == sha256(root / "manifest.json")


@pytest.mark.parametrize("corrupt", [False, True])
def test_resume_discount_uses_pinned_seals_and_small_stage_metadata(session, monkeypatch, corrupt):
    root, c, resources = session
    atomic_json(root / "manifest.json", {"config_hash": digest(c), "code_sha256": resources["code_sha256"]})
    snapshots = {}
    for name in ("reset50", "continued20"):
        child = campaign.job_roots(root)[name]
        job = c["jobs"][name]
        campaign._session(root, child, job, owner_key="owner_campaign_id")
        atomic_json(child / "manifest.json", {"config_hash": digest(job)})
        atomic_json(child / "completion.json", {"complete": True, "generations": job["generations"]})
        for g in range(1 if name == "continued20" else 2, job["generations"] + 1):
            gen = child / f"seed_42/gen_{g:02d}"
            training = {"checkpoint_selection": "final_epoch", "selected_epoch": 5,
                        "terminal_weights_sha256": "a" * 64, "selected_weights_sha256": "a" * 64}
            atomic_json(gen / "training.json", training)
            atomic_json(gen / "training_history.json", [{"epoch": i} for i in range(1, 6)])
            files = {p.name: sha256(p) for p in gen.glob("*.json")}
            atomic_json(gen / "stage_completion.json", {"selected_weights_sha256": "a" * 64, "files": files})
        files = {p.relative_to(child).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size, "repo_type": "dataset"}
                 for p in child.rglob("*") if p.is_file()}
        atomic_json(child / "artifact_index.json", {"run_id": child.name, "repos": resources["repos"], "files": files})
        snapshot = {"run_id": child.name, "repos": resources["repos"],
                    "revisions": {"model": "a" * 40, "dataset": "b" * 40},
                    "artifact_index_sha256": sha256(child / "artifact_index.json")}
        snapshots[name] = snapshot
        atomic_json(child / "recovery_verified.json", {**snapshot, "metadata_only": True})
        atomic_json(root / f"job_seals/{name}.json", {"snapshot": snapshot, "files": files,
                    "configuration_sha256": sha256(child / "config.json"), "completion_sha256": sha256(child / "completion.json")})
    atomic_json(root / "child_snapshots.json", snapshots)
    parent_files = {p.relative_to(root).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size, "repo_type": "dataset"}
                    for p in root.rglob("*") if p.is_file()}
    atomic_json(root / "artifact_index.json", {"run_id": root.name, "repos": resources["repos"], "files": parent_files})
    receipt = {"run_id": root.name, "repos": resources["repos"], "revisions": {"model": "c" * 40, "dataset": "d" * 40},
               "artifact_index_sha256": sha256(root / "artifact_index.json")}
    monkeypatch.setattr(campaign, "fetch_campaign", lambda *a, **k: receipt)
    if corrupt:
        atomic_json(campaign.job_roots(root)["reset50"] / "seed_42/gen_02/training.json", {"corrupt": True})
        with pytest.raises(RuntimeError, match="corrupt"):
            campaign.verified_resume_jobs(root, resources, c)
    else:
        proof = campaign.verified_resume_jobs(root, resources, c)
        assert proof["sealed_completed_jobs"] == ["reset50", "continued20"]
        assert proof["parent_artifact_index_sha256"] == receipt["artifact_index_sha256"]
