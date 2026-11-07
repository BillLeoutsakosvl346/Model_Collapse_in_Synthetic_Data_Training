"""Acceptance cannot certify another ratio, initializer, runtime or incomplete campaign."""
import copy

import numpy as np
import pytest

from model_collapse.common import PROJECT, atomic_json, config, digest, read_json, sha256
from model_collapse.continued import _audit_mixed_input
from model_collapse.data import save_array
from model_collapse.mixing import certificate_candidates, certificate_name, mixture_input
from model_collapse.workflow.lineage import token_hash
from model_collapse.workflow.profiles import (
    CAMPAIGN_CHECKS, MIXED_CHECKS, PAIR_CHECKS, mixture_checks, protocol_fingerprint, require_acceptance,
)


RUNTIME = "a" * 64


def child(fraction=.2, mode="continued"):
    c = config(PROJECT / "experiments/opt125m_wikitext2/continued_mixed_50_final5/full.yaml")
    c.update(synthetic_fraction=fraction, initialization=mode)
    if mode == "reset_s0":
        c.pop("initial_dataset")
        c["branch_source"] = {"run_id": "source", "repos": {"model": "test/models", "dataset": "test/data"},
                              "first_new_cycle": 2}
    return c


def certificate(c, *, operational=True):
    return {"source_code_sha256": RUNTIME, "protocol_fingerprint": protocol_fingerprint(c),
            "checks": dict.fromkeys(mixture_checks(c, include_operational=operational), True)}


def campaign():
    return {"workflow": "mixed_campaign", "kind": "full", "jobs": {
        "reset50": child(.5, "reset_s0"), "continued20": child(), "reset20": child(.2, "reset_s0")}}


def campaign_certificate(c):
    return {"workflow": "mixed_campaign", "source_code_sha256": RUNTIME,
            "protocol_fingerprint": protocol_fingerprint(c), "checks": dict.fromkeys(CAMPAIGN_CHECKS, True),
            "unique_training_stages": 6, "executed_epochs": 30,
            "child_certificates": {name: certificate(job, operational=False) for name, job in c["jobs"].items()},
            "shared20_mixture": {"continued_tokens_sha256": "b" * 64, "reset_tokens_sha256": "b" * 64,
                                 "continued_selection_sha256": "c" * 64, "reset_selection_sha256": "c" * 64}}


@pytest.mark.parametrize("fraction,real_count,synthetic_count", [(.5, 128, 128), (.2, 205, 51)])
def test_actual_saved_mixture_acceptance_supports_both_ratios(tmp_path, fraction, real_count, synthetic_count):
    real = np.arange(256 * 64, dtype=np.int32).reshape(256, 64)
    synthetic = real + 20000
    save_array(tmp_path / "data/train.npy", real)
    atomic_json(tmp_path / "data/manifest.json", {"files": {"train.npy": sha256(tmp_path / "data/train.npy")}})
    source = tmp_path / "synthetic_for_gen_01"
    save_array(source / "tokens.npy", synthetic)
    parent = {"files": {"tokens.npy": sha256(source / "tokens.npy")},
              "continuation_tokens_sha256": token_hash(synthetic), "provenance": {"parent_weights_sha256": "generator"}}
    atomic_json(source / "manifest.json", parent)
    c = {"seed": 42, "synthetic_fraction": fraction}
    actual, _ = mixture_input(tmp_path, 1, c, synthetic, parent, "generator")
    reconstructed, manifest = _audit_mixed_input(tmp_path, 1, c, synthetic, parent, "generator")
    assert np.array_equal(actual, reconstructed)
    assert (manifest["real_blocks"], manifest["synthetic_blocks"]) == (real_count, synthetic_count)
    assert np.sum(actual[:, 0] < 20000) == real_count
    assert np.sum(actual[:, 0] >= 20000) == synthetic_count


def test_audit_independently_checks_selection_labels(tmp_path, monkeypatch):
    from model_collapse import mixing
    real = np.arange(256 * 64, dtype=np.int32).reshape(256, 64)
    synthetic = real + 20000
    save_array(tmp_path / "data/train.npy", real)
    atomic_json(tmp_path / "data/manifest.json", {"files": {"train.npy": sha256(tmp_path / "data/train.npy")}})
    save_array(tmp_path / "synthetic_for_gen_01/tokens.npy", synthetic)
    parent = {"files": {"tokens.npy": sha256(tmp_path / "synthetic_for_gen_01/tokens.npy")},
              "continuation_tokens_sha256": token_hash(synthetic), "provenance": {"parent_weights_sha256": "generator"}}
    atomic_json(tmp_path / "synthetic_for_gen_01/manifest.json", parent)
    c = {"seed": 42, "synthetic_fraction": .2}
    values, manifest = mixture_input(tmp_path, 1, c, synthetic, parent, "generator")
    selection = read_json(tmp_path / "mixed_for_gen_01/selection.json")
    selection["source_labels"][0] = 1 - selection["source_labels"][0]
    atomic_json(tmp_path / "mixed_for_gen_01/selection.json", selection)
    # A trusted cached reader alone must not mask an inconsistent audit label array.
    monkeypatch.setattr(mixing, "mixture_input", lambda *a, **kw: (values, manifest))
    with pytest.raises(AssertionError):
        _audit_mixed_input(tmp_path, 1, c, synthetic, parent, "generator")


@pytest.mark.parametrize("fraction,mode", [(.5, "continued"), (.5, "reset_s0"), (.2, "continued"), (.2, "reset_s0")])
def test_mixed_acceptance_requires_initializer_specific_evidence(fraction, mode):
    c = child(fraction, mode)
    cert = certificate(c)
    require_acceptance(c, cert, RUNTIME)
    key = "both_weights_and_data_handoffs" if mode == "continued" else "full_shared_mixture_reconstruction"
    cert["checks"].pop(key)
    with pytest.raises(RuntimeError, match="mixed-data"):
        require_acceptance(c, cert, RUNTIME)


def test_legacy_continued_check_set_still_matches_real_historical_evidence():
    c = child(.5)
    cert = {"source_code_sha256": RUNTIME, "protocol_fingerprint": protocol_fingerprint(c),
            "checks": dict.fromkeys(MIXED_CHECKS, True)}
    require_acceptance(c, cert, RUNTIME)


def test_certificates_coexist_and_legacy_lookup_does_not_certify_another_runtime():
    conditions = [child(.5), child(.2), child(.5, "reset_s0"), child(.2, "reset_s0"), campaign()]
    names = {certificate_name(c, RUNTIME) for c in conditions}
    assert len(names) == len(conditions)
    assert certificate_name(conditions[0], "d" * 64) not in names
    assert certificate_candidates(conditions[0], RUNTIME)[1] == "validation_continued_mixed.json"
    assert certificate_candidates(campaign(), RUNTIME) == (certificate_name(campaign(), RUNTIME),)
    c = child()
    with pytest.raises(RuntimeError, match="source version"):
        require_acceptance(c, certificate(c), "d" * 64)
    assert "/" not in certificate_name(c, "../../outside")


def test_campaign_smoke_and_full_scope_normalize_but_scientific_changes_do_not():
    c = campaign()
    smoke = copy.deepcopy(c)
    smoke["kind"] = "smoke"
    for name, job in smoke["jobs"].items():
        job.update(kind="smoke", train_blocks=256, validation_blocks=64, test_blocks=64,
                   generations=2 if name == "continued20" else 3)
    assert protocol_fingerprint(c) == protocol_fingerprint(smoke)
    for key, value in (("synthetic_fraction", .5), ("epochs", 1), ("checkpoint_selection", "lowest_fixed_real_validation_loss")):
        other = copy.deepcopy(c)
        other["jobs"]["reset20"][key] = value
        assert protocol_fingerprint(other) != protocol_fingerprint(c)


@pytest.mark.parametrize("fault", ["checks", "epochs", "child_runtime", "child_profile", "reset_mode_check", "tokens", "selection", "missing_child"])
def test_campaign_rejects_incomplete_or_mismatched_child_evidence(fault):
    c = campaign()
    cert = campaign_certificate(c)
    require_acceptance(c, cert, RUNTIME)
    if fault == "checks": cert["checks"].pop("three_private_restores")
    elif fault == "epochs": cert["executed_epochs"] = 10
    elif fault == "child_runtime": cert["child_certificates"]["continued20"]["source_code_sha256"] = "d" * 64
    elif fault == "child_profile": cert["child_certificates"]["reset20"]["protocol_fingerprint"] = "d" * 64
    elif fault == "reset_mode_check": cert["child_certificates"]["reset50"]["checks"].pop("fixed_s0_weights_and_separate_generator_handoffs")
    elif fault in ("tokens", "selection"): cert["shared20_mixture"][f"reset_{fault}_sha256"] = "d" * 64
    elif fault == "missing_child": cert["child_certificates"].pop("reset20")
    with pytest.raises(RuntimeError):
        require_acceptance(c, cert, RUNTIME)


def test_paired_mixed_acceptance_requires_child_audits_in_addition_to_parent_checks():
    c = {"workflow": "paired", "arms": {"continued": child(), "reset": child(.2, "reset_s0")}}
    cert = {"workflow": "paired", "source_code_sha256": RUNTIME, "protocol_fingerprint": protocol_fingerprint(c),
            "checks": dict.fromkeys(PAIR_CHECKS, True)}
    with pytest.raises(RuntimeError, match="child protocol"):
        require_acceptance(c, cert, RUNTIME)
    cert["child_certificates"] = {name: certificate(job, operational=False) for name, job in c["arms"].items()}
    cert["shared_cycle2_mixture"] = campaign_certificate(campaign())["shared20_mixture"]
    require_acceptance(c, cert, RUNTIME)


@pytest.fixture
def campaign_evidence(tmp_path, monkeypatch):
    from model_collapse import common, remote, runpod_api
    from model_collapse.workflow import mixed_campaign
    c = campaign()
    c["kind"] = "smoke"
    for name, job in c["jobs"].items():
        job.update(kind="smoke", train_blocks=256, validation_blocks=64, test_blocks=64,
                   generations=2 if name == "continued20" else 3)
    root = tmp_path / "campaign"
    resources = {"run_id": root.name, "launch_id": "launch", "pod_id": "pod", "volume_id": "volume",
                 "created_at": 123, "code_sha256": RUNTIME, "session_allowance_usd": .75,
                 "guard_hourly_usd": .66, "repos": {"model": "test/models", "dataset": "test/data"}}
    atomic_json(root / "config.json", c)
    atomic_json(root / "resources.json", resources)
    atomic_json(root / "artifact_index.json", {"files": {}})
    atomic_json(root / "watchdog_exit.json", {"pod_id": "pod", "reason": "validation_budget_guard"})
    observations = [{"epochs": 1, "new_epochs": 1, "finite_new_training": True},
                    {"epochs": 2, "new_epochs": 2, "finite_new_training": True}]
    atomic_json(root / "detached_verification.json", {"run_id": root.name, "launch_id": "launch", "pod_id": "pod",
                "progress_after_disconnect": True, "new_training_after_import_verified": True,
                "observations": observations, "observations_sha256": digest(observations)})
    cert = campaign_certificate(c)
    cert["checks"].pop("detached_progress_verified")
    cert["checks"].pop("independent_budget_guard_and_provider_stop_verified")
    cert["child_audits"] = {}
    for name, path in mixed_campaign.job_roots(root).items():
        atomic_json(path / "remote_validation.json", cert["child_certificates"][name])
        cert["child_audits"][name] = sha256(path / "remote_validation.json")
        atomic_json(path / "resources.json", {**resources, "owner_campaign_id": root.name, "logical_run_id": path.name})
    atomic_json(root / "remote_validation.json", cert)
    monkeypatch.setattr(common, "STATE", tmp_path / "acceptance")
    monkeypatch.setattr(mixed_campaign, "verify_campaign_recovery", lambda root: None)
    monkeypatch.setattr(remote, "local_resources", lambda run_id: resources)
    monkeypatch.setattr(remote, "source_archive", lambda: (None, RUNTIME))
    class Provider:
        def pod(self, identifier):
            assert identifier == "pod"
            return {"id": identifier, "status": "EXITED"}
    monkeypatch.setattr(runpod_api, "Runpod", Provider)
    return root, c, resources


def test_campaign_local_certificate_is_written_only_after_all_checks(campaign_evidence, tmp_path):
    from model_collapse.continued import validate_campaign_local
    root, c, resources = campaign_evidence
    result = validate_campaign_local(root)
    assert result["provider_pod_state"] == "EXITED"
    assert all(result["checks"].values())
    assert (tmp_path / "acceptance" / certificate_name(c, RUNTIME)).is_file()
    assert len(list((tmp_path / "acceptance").glob("validation*.json"))) == 1


def seal_prior_child(evidence, *, runtime=RUNTIME, volume="volume"):
    """Keep the original execution immutable while registering its recoverable finished job."""
    from model_collapse.workflow import mixed_campaign
    root, c, current = evidence
    child_root = mixed_campaign.job_roots(root)["reset50"]
    original = {**read_json(child_root / "resources.json"), "launch_id": "prior-launch", "pod_id": "prior-pod",
                "created_at": 100, "session_allowance_usd": .5, "code_sha256": runtime, "volume_id": volume}
    atomic_json(child_root / "resources.json", original)
    current["provider_sessions"] = [{key: original[key] for key in ("run_id", "launch_id", "pod_id", "volume_id")}]
    atomic_json(root / "resources.json", current)
    job = c["jobs"]["reset50"]
    atomic_json(child_root / "config.json", job)
    atomic_json(child_root / "manifest.json", {"config_hash": digest(job)})
    atomic_json(child_root / "completion.json", {"complete": True, "generations": job["generations"]})
    audit_path = child_root / "remote_validation.json"
    audit = read_json(audit_path)
    audit["source_code_sha256"] = runtime
    atomic_json(audit_path, audit)
    parent_audit = read_json(root / "remote_validation.json")
    parent_audit["child_certificates"]["reset50"] = audit
    parent_audit["child_audits"]["reset50"] = sha256(audit_path)
    atomic_json(root / "remote_validation.json", parent_audit)
    files = {name: {"sha256": sha256(child_root / name), "bytes": (child_root / name).stat().st_size,
                    "repo_type": "dataset"}
             for name in ("config.json", "manifest.json", "completion.json", "resources.json", "remote_validation.json")}
    atomic_json(child_root / "artifact_index.json", {"files": files})
    snapshot = {"run_id": child_root.name, "repos": current["repos"],
                "revisions": {"model": "a" * 40, "dataset": "b" * 40},
                "artifact_index_sha256": sha256(child_root / "artifact_index.json")}
    atomic_json(child_root / "backup_snapshot.json", snapshot)
    atomic_json(root / "child_snapshots.json", {"reset50": snapshot})
    atomic_json(root / "job_seals/reset50.json", {"campaign_id": root.name, "job": "reset50", "run_id": child_root.name,
                "configuration_sha256": sha256(child_root / "config.json"),
                "completion_sha256": sha256(child_root / "completion.json"), "snapshot": snapshot, "files": files})
    return child_root, original


def test_campaign_local_certificate_accepts_sealed_registered_prior_launch(campaign_evidence):
    from model_collapse.continued import validate_campaign_local
    root, _, _ = campaign_evidence
    child_root, original = seal_prior_child(campaign_evidence)
    prior_bytes = (child_root / "resources.json").read_bytes()
    result = validate_campaign_local(root)
    assert result["child_provider_sessions"]["reset50"]["current_session"] is False
    assert result["child_provider_sessions"]["reset50"]["pod_id"] == "prior-pod"
    assert result["child_provider_sessions"]["continued20"]["current_session"] is True
    assert read_json(child_root / "resources.json") == original
    assert (child_root / "resources.json").read_bytes() == prior_bytes


@pytest.mark.parametrize("fault", ["unregistered", "wrong_volume", "wrong_runtime", "unsealed", "unfinished", "altered_resource"])
def test_campaign_local_validation_rejects_invalid_prior_provider_evidence(campaign_evidence, tmp_path, fault):
    from model_collapse.continued import validate_campaign_local
    root, _, current = campaign_evidence
    child_root, _ = seal_prior_child(campaign_evidence, runtime="d" * 64 if fault == "wrong_runtime" else RUNTIME,
                                     volume="another-volume" if fault == "wrong_volume" else "volume")
    if fault == "unregistered":
        current["provider_sessions"] = []
        atomic_json(root / "resources.json", current)
    elif fault == "unsealed":
        (root / "job_seals/reset50.json").unlink()
    elif fault == "unfinished":
        atomic_json(child_root / "completion.json", {"complete": False})
    elif fault == "altered_resource":
        value = read_json(child_root / "resources.json")
        value["created_at"] = 101
        atomic_json(child_root / "resources.json", value)
    with pytest.raises(RuntimeError):
        validate_campaign_local(root)
    assert not (tmp_path / "acceptance").exists()


@pytest.mark.parametrize("fault", ["detached_launch", "provider_running", "registry_volume", "child_owner", "child_audit", "missing_restore",
                                  "missing_new_training", "imported_only", "nonfinite_training", "boolean_epochs", "stale_observations"])
def test_campaign_local_validation_rejects_unbound_evidence(campaign_evidence, tmp_path, monkeypatch, fault):
    from model_collapse import runpod_api
    from model_collapse.continued import validate_campaign_local
    from model_collapse.workflow import mixed_campaign
    root, c, resources = campaign_evidence
    if fault == "detached_launch":
        evidence = read_json(root / "detached_verification.json")
        evidence["launch_id"] = "another-launch"
        atomic_json(root / "detached_verification.json", evidence)
    elif fault == "provider_running":
        monkeypatch.setattr(runpod_api, "Runpod", lambda: type("Provider", (), {"pod": lambda self, identifier: {"status": "RUNNING"}})())
    elif fault == "registry_volume":
        resources["volume_id"] = "another-volume"
    elif fault == "child_owner":
        path = mixed_campaign.job_roots(root)["reset20"] / "resources.json"
        value = read_json(path)
        value["owner_campaign_id"] = "another-campaign"
        atomic_json(path, value)
    elif fault == "child_audit":
        path = mixed_campaign.job_roots(root)["reset50"] / "remote_validation.json"
        value = read_json(path)
        value["checks"]["exact_mixture_reconstruction"] = False
        atomic_json(path, value)
    elif fault == "missing_restore":
        def reject(root):
            raise RuntimeError("Missing child restore")
        monkeypatch.setattr(mixed_campaign, "verify_campaign_recovery", reject)
    else:
        evidence = read_json(root / "detached_verification.json")
        if fault == "missing_new_training": evidence.pop("new_training_after_import_verified")
        elif fault == "imported_only": evidence["observations"][-1]["new_epochs"] = 0
        elif fault == "nonfinite_training": evidence["observations"][-1]["finite_new_training"] = False
        elif fault == "boolean_epochs": evidence["observations"][-1]["new_epochs"] = True
        elif fault == "stale_observations": evidence["observations_sha256"] = "d" * 64
        if fault != "stale_observations": evidence["observations_sha256"] = digest(evidence["observations"])
        atomic_json(root / "detached_verification.json", evidence)
    with pytest.raises(RuntimeError):
        validate_campaign_local(root)
    assert not (tmp_path / "acceptance").exists()
