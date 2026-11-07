import copy
import sys
from types import SimpleNamespace

import pytest

from model_collapse.common import PROJECT, atomic_json, config, event, read_json, sha256
from model_collapse.workflow import paired
from model_collapse.workflow.profiles import PAIR_CHECKS, protocol_fingerprint, require_acceptance


@pytest.fixture
def pair(tmp_path, monkeypatch):
    c = config(PROJECT / "comparisons/shared_start_final_epoch5/smoke.yaml")
    root = tmp_path / "pair"
    r = {"run_id": root.name, "launch_id": "launch", "pod_id": "pod", "volume_id": "volume",
         "created_at": 123, "session_allowance_usd": .75, "guard_hourly_usd": .66, "code_sha256": "runtime",
         "repos": {"model": "test/models", "dataset": "test/data"}}
    atomic_json(root / "config.json", c)
    atomic_json(root / "resources.json", r)
    monkeypatch.setattr(paired.shutil, "disk_usage", lambda *a: SimpleNamespace(free=30_000_000_000))
    return root, c, r


def test_smoke_and_full_profiles_match_after_explicit_scope_normalization():
    full = config(PROJECT / "comparisons/shared_start_final_epoch5/full.yaml")
    smoke = config(PROJECT / "comparisons/shared_start_final_epoch5/smoke.yaml")
    assert protocol_fingerprint(full) == protocol_fingerprint(smoke)
    assert full["arms"]["continued"]["generations"] == full["arms"]["reset"]["generations"] == 10
    for arm in full["arms"].values():
        assert arm["epochs"] == 5 and arm["checkpoint_selection"] == "final_epoch"


@pytest.mark.parametrize("change", ["runtime", "epochs", "policy", "gpu", "environment", "s0", "d1", "workflow", "checks"])
def test_acceptance_rejects_incompatible_protocol(pair, change):
    root, c, r = pair
    cert = {"workflow": "paired", "source_code_sha256": "runtime", "protocol_fingerprint": protocol_fingerprint(c),
            "checks": {key: True for key in PAIR_CHECKS}}
    require_acceptance(c, cert, "runtime")
    if change == "runtime":
        cert["source_code_sha256"] = "other"
    elif change in ("workflow", "checks"):
        cert[change] = "single" if change == "workflow" else {}
    else:
        a = c["arms"]["continued"]
        if change == "epochs": a["epochs"] = 1
        if change == "policy": a["checkpoint_selection"] = "lowest_fixed_real_validation_loss"
        if change == "gpu": a["required_gpu"] = "different"
        if change == "environment": a["required_environment"]["torch"] = "different"
        if change == "s0": a["start_checkpoint"]["weights_sha256"] = "0" * 64
        if change == "d1": a["initial_dataset"]["artifact_index_sha256"] = "0" * 64
    with pytest.raises(RuntimeError):
        require_acceptance(c, cert, "runtime")


def test_worker_dispatches_parent_and_child_separately(pair, monkeypatch):
    from model_collapse import __main__, pipeline
    root, c, r = pair
    calls = []
    monkeypatch.setattr(paired, "run_pair", lambda *a: calls.append("parent"))
    monkeypatch.setattr(pipeline, "run", lambda *a: calls.append("child"))
    monkeypatch.setattr(sys, "argv", ["model_collapse", "worker", "--root", str(root)])
    __main__.main()
    child = paired.child_roots(root)["continued"]
    paired._session(root, child, c["arms"]["continued"])
    monkeypatch.setattr(sys, "argv", ["model_collapse", "worker", "--root", str(child)])
    __main__.main()
    assert calls == ["parent", "child"]
    cr = read_json(child / "resources.json")
    assert cr["run_id"] == root.name and cr["logical_run_id"] == child.name
    assert all(cr[k] == r[k] for k in ("launch_id", "pod_id", "volume_id", "created_at", "session_allowance_usd"))


def test_child_pause_prevents_source_seal_and_reset(pair, monkeypatch):
    from model_collapse import hub, supervisor
    root, c, r = pair
    called = []
    def worker(child, **kwargs):
        called.append(child.name)
        atomic_json(child / "status.json", {"state": "budget_paused"})
    monkeypatch.setattr(supervisor, "run_worker", worker)
    monkeypatch.setattr(hub, "backup", lambda *a: None)
    paired.run_pair(root)
    assert called == [root.name + "-continued"]
    assert read_json(root / "status.json")["state"] == "budget_paused"
    assert not (root / "continued_seal.json").exists()
    assert not (paired.child_roots(root)["reset"] / "config.json").exists()


def test_handoff_exit_resumes_without_reopening_continued(pair, monkeypatch):
    from model_collapse import hub, supervisor
    from model_collapse.engine import ValidationInterruption
    root, c, r = pair
    calls = []
    def worker(child, **kwargs):
        calls.append(child.name)
        atomic_json(child / "status.json", {"state": "complete"})
        atomic_json(child / "completion.json", {"complete": True})
        atomic_json(child / "remote_validation.json", {"checks": {"restored_checkpoint_tensor_hash_and_evaluation": True}})
        for g in (1, 2, 3):
            atomic_json(child / f"seed_42/gen_{g:02d}/training.json", {"selected_epoch": 5, "executed_epochs": 5})
    def seal(root, c):
        value = {"continued_files": {}, "branch_source": c["arms"]["reset"]["branch_source"]}
        paired._write_fixed(root / "continued_seal.json", value)
        return c["arms"]["reset"], value
    monkeypatch.setattr(supervisor, "run_worker", worker)
    monkeypatch.setattr(paired, "seal_continued", seal)
    monkeypatch.setattr(paired, "verify_completed_child", lambda *a: {})
    monkeypatch.setattr(paired, "report_pair", lambda *a, **k: {})
    monkeypatch.setattr(hub, "backup", lambda *a: {})
    with pytest.raises(ValidationInterruption, match="source seal"):
        paired.run_pair(root, validation=True)
    event(root, "validation_worker_resumed", interruption=1)
    continued = paired.child_roots(root)["continued"]
    original = sha256(continued / "resources.json")
    paired.run_pair(root, validation=True)
    assert calls == [root.name + "-continued", root.name + "-reset-s0"]
    assert sha256(continued / "resources.json") == original
    assert read_json(root / "completion.json")["executed_epochs"] == 20


def test_final_backup_is_required_not_completion_file_alone(pair):
    root, c, r = pair
    child = paired.child_roots(root)["continued"]
    paired._session(root, child, c["arms"]["continued"])
    atomic_json(child / "completion.json", {"complete": True})
    with pytest.raises(RuntimeError, match="expected endpoint"):
        paired.verify_completed_child(child, c["arms"]["continued"])


def test_cleanup_requires_both_recoveries(pair):
    root, c, r = pair
    for children in ({}, {"continued": {}}):
        atomic_json(root / "recovery_verified.json", {"children": children})
        with pytest.raises(RuntimeError, match="Both child"):
            paired.verify_pair_recovery(root)


def test_pair_endpoints_cannot_be_changed_on_only_one_arm(pair):
    root, c, r = pair
    c["arms"]["continued"]["generations"] = 10
    with pytest.raises(ValueError, match="endpoints"):
        paired.validate_pair_config(c)


@pytest.mark.parametrize("condition", ["pretrained_reset", "continued_from_real_s0", "reset_to_real_s0"])
def test_legacy_smoke_full_acceptance_remains_compatible(condition):
    folder = PROJECT / "experiments/opt125m_wikitext2" / condition
    smoke, full = (config(folder / (kind + ".yaml")) for kind in ("smoke", "full"))
    assert smoke["epochs"] == 1 and full["epochs"] == 5
    assert protocol_fingerprint(smoke) == protocol_fingerprint(full)
    require_acceptance(full, {"source_code_sha256": "runtime", "protocol_fingerprint": protocol_fingerprint(smoke)}, "runtime")


@pytest.mark.parametrize("pending", ["continued", "reset"])
def test_completion_before_final_backup_recovers_without_training(pair, monkeypatch, pending):
    from model_collapse import hub, supervisor
    root, c, resources = pair
    roots = paired.child_roots(root)
    for name, child in roots.items():
        paired._session(root, child, c["arms"][name])
        atomic_json(child / "completion.json", {"complete": True})
        atomic_json(child / "status.json", {"state": "complete"})
    backup_calls, finalized = [], []
    def verify(child, c):
        if child == roots[pending] and not finalized:
            raise paired.FinalBackupPending("final backup interrupted")
        return {}
    def backup(child):
        backup_calls.append(child)
        if child == roots[pending]: finalized.append(True)
    def seal(root, c):
        value = {"continued_files": {}}
        paired._write_fixed(root / "continued_seal.json", value)
        return c["arms"]["reset"], value
    monkeypatch.setattr(paired, "verify_completed_child", verify)
    monkeypatch.setattr(paired, "seal_continued", seal)
    monkeypatch.setattr(paired, "report_pair", lambda *a, **k: {})
    monkeypatch.setattr(hub, "backup", backup)
    monkeypatch.setattr(supervisor, "run_worker", lambda *a, **k: pytest.fail("Complete stages must not retrain"))
    paired.run_pair(root)
    assert finalized and [p for p in backup_calls if p in roots.values()] == [roots[pending]]
    assert read_json(root / "completion.json")["complete"]


@pytest.mark.parametrize("wrong", ["revision", "index", "run_id", "repos"])
def test_stale_or_swapped_child_recovery_cannot_allow_cleanup(pair, monkeypatch, wrong):
    root, c, resources = pair
    snapshots, receipts = {}, {}
    for name, child in paired.child_roots(root).items():
        atomic_json(child / "artifact_index.json", {"files": {}})
        snapshot = {"run_id": child.name, "repos": resources["repos"],
                    "revisions": {"model": "a" * 40, "dataset": "b" * 40},
                    "artifact_index_sha256": sha256(child / "artifact_index.json")}
        snapshots[name] = snapshot
        receipt = {**copy.deepcopy(snapshot), "metadata_only": False}
        atomic_json(child / "recovery_verified.json", receipt)
        receipts[name] = receipt
    atomic_json(root / "recovery_verified.json", {"children": receipts})
    atomic_json(root / "child_snapshots.json", snapshots)
    atomic_json(root / "continued_seal.json", {"snapshot": snapshots["continued"]})
    paired.verify_pair_recovery(root)
    child = paired.child_roots(root)["reset"]
    receipt = read_json(child / "recovery_verified.json")
    if wrong == "revision": receipt["revisions"]["dataset"] = "0" * 40
    if wrong == "index": receipt["artifact_index_sha256"] = "0" * 64
    if wrong == "run_id": receipt["run_id"] = paired.child_roots(root)["continued"].name
    if wrong == "repos": receipt["repos"]["model"] = "wrong/models"
    atomic_json(child / "recovery_verified.json", receipt)
    with pytest.raises(RuntimeError, match="stale|different snapshot"):
        paired.verify_pair_recovery(root)


@pytest.mark.parametrize("accepted", [True, False])
def test_paired_full_launch_admission_stops_before_paid_boundary(tmp_path, monkeypatch, accepted):
    from model_collapse import remote
    c = config(PROJECT / "comparisons/shared_start_final_epoch5/full.yaml")
    class Boundary(Exception): pass
    class API:
        def candidates(self):
            return [{"gpu": c["required_gpu"], "data_center": "EU-RO-1", "hourly_usd": .57}]
        def request(self, method, path, body=None):
            assert method == "GET", "Admission must not provision"
            return {"pods": []} if path == "/pods" else {"templates": [{"id": "runpod-torch-v280", "image": c["required_image"]}]}
    monkeypatch.setattr(remote, "STATE", tmp_path)
    monkeypatch.setattr(remote, "Runpod", API)
    monkeypatch.setattr(remote, "ledger", lambda: {"launches": []})
    monkeypatch.setattr(remote, "preflight", lambda **k: {"account": {"clientBalance": 30}})
    monkeypatch.setattr(remote, "source_archive", lambda: ("unused", "runtime"))
    monkeypatch.setattr(paired, "resolve_pair_inputs", lambda c: c)
    monkeypatch.setattr(remote, "verify_start_checkpoint", lambda c: None)
    monkeypatch.setattr(remote, "register_key", lambda api: (_ for _ in ()).throw(Boundary()))
    cert = {"workflow": "paired", "source_code_sha256": "runtime", "protocol_fingerprint": protocol_fingerprint(c),
            "checks": {key: True for key in PAIR_CHECKS}}
    if not accepted: cert["checks"].pop("handoff_restart")
    atomic_json(tmp_path / "validation_paired.json", cert)
    atomic_json(tmp_path / "benchmark.json", {"training_target_tokens_per_second": 30000,
                "generation_new_tokens_per_second": 1300,
                "full_split_blocks": {"train": 37946, "validation": 3922, "test": 4494}})
    with pytest.raises(Boundary if accepted else RuntimeError):
        remote.launch(config_path=PROJECT / "comparisons/shared_start_final_epoch5/full.yaml", generations=10)


def test_combined_campaign_cap_rejects_an_unaffordable_retry(tmp_path, monkeypatch):
    from model_collapse import remote
    c = config(PROJECT / "comparisons/shared_start_final_epoch5/smoke.yaml")
    class API:
        def request(self, method, path, body=None):
            assert method == "GET", "Budget exhaustion must not provision"
            return {"pods": []}
        def candidates(self):
            return [{"gpu": c["required_gpu"], "data_center": "EU-RO-1", "hourly_usd": .57}]
    monkeypatch.setattr(remote, "STATE", tmp_path)
    monkeypatch.setattr(remote, "Runpod", API)
    monkeypatch.setattr(remote, "preflight", lambda **k: {"account": {"clientBalance": 30}})
    monkeypatch.setattr(remote, "ledger", lambda: {"launches": []})
    monkeypatch.setattr(remote, "source_archive", lambda: ("unused", "runtime"))
    monkeypatch.setattr(remote, "campaign_accrued", lambda *a: .74)
    monkeypatch.setattr(paired, "resolve_pair_inputs", lambda c: c)
    monkeypatch.setattr(remote, "register_key", lambda *a: pytest.fail("Budget exhaustion must not register keys"))
    with pytest.raises(RuntimeError, match="Insufficient budget"):
        remote.launch(smoke=True, config_path=PROJECT / "comparisons/shared_start_final_epoch5/smoke.yaml")


def test_completed_cloud_resume_is_rejected_before_provision(tmp_path, monkeypatch):
    import huggingface_hub
    from model_collapse import remote
    atomic_json(tmp_path / "completion.json", {"complete": True})
    monkeypatch.setattr(remote, "PROJECT", tmp_path)
    monkeypatch.setattr(remote, "HfApi", lambda: SimpleNamespace(repo_info=lambda *a, **k: SimpleNamespace(sha="a" * 40)))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(tmp_path / "completion.json"))
    with pytest.raises(RuntimeError, match="Completed cloud"):
        remote.reject_completed_cloud_resume("pair", {"dataset": "test/data"})
