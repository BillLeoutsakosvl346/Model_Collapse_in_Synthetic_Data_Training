"""Three fixed mixed-data jobs under one detached provider owner and deadline."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

from ..common import atomic_json, digest, event, now, read_json, sha256, verify_files
from .paired import FinalBackupPending, _session, _write_fixed, verify_completed_child
from .selection import FINAL, selection_policy, verify_selection

CAMPAIGN = "mixed_fraction_final5"
JOB_ORDER = ("reset50", "continued20", "reset20")
JOB_SUFFIXES = {"reset50": "-reset-mixed50", "continued20": "-continued-mixed20", "reset20": "-reset-mixed20"}
VOLUME_GB = 60
MINIMUM_INITIAL_FREE_BYTES = 40_000_000_000


def validate_campaign_config(c):
    from ..common import validate_config
    from ..mixing import synthetic_fraction
    if c.get("workflow") != "mixed_campaign" or c.get("kind") not in ("full", "smoke"):
        raise ValueError("Mixed campaign requires its explicit workflow and full/smoke kind")
    if set(c.get("jobs", {})) != set(JOB_ORDER):
        raise ValueError("Mixed campaign requires reset50, continued20 and reset20 jobs")
    if c.get("network_volume_gb") != VOLUME_GB:
        raise ValueError("Mixed campaign requires one 60 GB network volume")
    expected = {"reset50": ("reset_s0", .5), "continued20": ("continued", .2), "reset20": ("reset_s0", .2)}
    for name in JOB_ORDER:
        job = c["jobs"][name]
        mode, fraction = expected[name]
        if job.get("workflow") is not None or job.get("kind") != c["kind"]:
            raise ValueError("Campaign children must be ordinary matching-kind experiment configurations")
        validate_config(job)
        if job.get("initialization") != mode or synthetic_fraction(job) != fraction:
            raise ValueError("Campaign jobs must preserve their fixed initialization and mixture fractions")
        if job["epochs"] != 5 or selection_policy(job) != FINAL:
            raise ValueError("Every mixed campaign stage must execute and retain epoch 5")
        end = 10 if c["kind"] == "full" else (2 if name == "continued20" else 3)
        if job["generations"] != end:
            raise ValueError("Campaign endpoints must be full 10/10/10 or smoke 3/2/3")
        if c["kind"] == "smoke" and [job[k] for k in ("train_blocks", "validation_blocks", "test_blocks")] != [256, 64, 64]:
            raise ValueError("Campaign smoke requires exactly 256/64/64 blocks")
        for key in ("required_gpu", "required_environment", "required_image"):
            if job.get(key) != c.get(key):
                raise ValueError("Every campaign job must use the parent's matched GPU/software image")
        if name == "continued20":
            if not job.get("initial_dataset") or job.get("branch_source"):
                raise ValueError("Only continued20 imports the historical D1 source pool")
        elif job.get("initial_dataset"):
            raise ValueError("Reset jobs import a verified shared prefix, not a new D1")
    if c["jobs"]["reset20"]["branch_source"].get("run_id") != "__campaign_source__":
        raise ValueError("The reset20 source is bound only after continued20 completes")
    if c["jobs"]["reset50"]["branch_source"].get("run_id") != "20261031T163428Z-full-continued":
        raise ValueError("The reset50 source must be the completed historical mixed50 experiment")
    normalized = []
    for name in JOB_ORDER:
        job = copy.deepcopy(c["jobs"][name])
        for key in ("initialization", "synthetic_fraction", "initial_dataset", "branch_source", "experiment_id", "generations"):
            job.pop(key, None)
        normalized.append(job)
    if any(job != normalized[0] for job in normalized[1:]):
        raise ValueError("Campaign scientific settings must match apart from mixture and initialization")


def resolve_campaign_inputs(c):
    """Freeze all historical inputs before provisioning; leave reset20 deliberately unbound."""
    from .branch import resolve_branch_source
    from .initial_data import resolve_initial_dataset
    c = copy.deepcopy(c)
    validate_campaign_config(c)
    c["jobs"]["reset50"] = resolve_branch_source(c["jobs"]["reset50"])
    c["jobs"]["continued20"] = resolve_initial_dataset(c["jobs"]["continued20"])
    validate_campaign_config(c)
    return c


def job_roots(root):
    root = Path(root)
    return {name: root.with_name(root.name + JOB_SUFFIXES[name]) for name in JOB_ORDER}


child_roots = job_roots


def _seal_path(root, job):
    return Path(root) / "job_seals" / (job + ".json")


def _check_child_owner(root, child, resources=None, *, require_sealed=False):
    resources = resources or read_json(Path(child) / "resources.json", {})
    if (resources.get("owner_campaign_id") != Path(root).name or resources.get("run_id") != Path(root).name
            or resources.get("logical_run_id") != Path(child).name
            or resources.get("owner_pair_id") is not None):
        raise RuntimeError("Campaign child resource ownership differs")
    parent = read_json(Path(root) / "resources.json", {})
    fields = ("run_id", "launch_id", "pod_id", "volume_id")
    name = next((key for key, path in job_roots(root).items() if path == Path(child)), None)
    if name is None or not parent.get("code_sha256") or resources.get("code_sha256") != parent["code_sha256"]:
        raise RuntimeError("Campaign child runtime or logical job differs from its provider owner")
    allowed = [parent]
    if read_json(Path(child) / "completion.json", {}).get("complete"):
        allowed += parent.get("provider_sessions", [])
    if not any(all(resources.get(k) == candidate.get(k) for k in fields)
               and candidate.get("run_id") == Path(root).name
               and candidate.get("volume_id") == parent.get("volume_id") for candidate in allowed):
        raise RuntimeError("Campaign child provider association is not a registered session of this owner and volume")
    current = all(resources.get(k) == parent.get(k) for k in fields)
    if current and any(resources.get(k) != parent.get(k) for k in ("created_at", "session_allowance_usd", "guard_hourly_usd")):
        raise RuntimeError("Campaign child uses a different current-session budget deadline")
    if require_sealed and not current:
        child = Path(child)
        seal = read_json(_seal_path(root, name), {})
        snapshot = read_json(Path(root) / "child_snapshots.json", {}).get(name)
        index = read_json(child / "artifact_index.json", {})
        if (not snapshot or seal.get("snapshot") != snapshot or snapshot.get("run_id") != child.name
                or snapshot.get("repos") != parent.get("repos")
                or snapshot.get("artifact_index_sha256") != sha256(child / "artifact_index.json")
                or seal.get("configuration_sha256") != sha256(child / "config.json")
                or seal.get("completion_sha256") != sha256(child / "completion.json")
                or not read_json(child / "completion.json", {}).get("complete")
                or seal.get("files") != index.get("files")
                or index.get("files", {}).get("resources.json", {}).get("sha256") != sha256(child / "resources.json")):
            raise RuntimeError("Prior-session campaign child has no matching immutable completion seal")
    return resources


def record_child_snapshot(child):
    """Record a successful private backup without creating a child provider owner."""
    from filelock import FileLock
    from ..hub import backup
    child = Path(child)
    resources = read_json(child / "resources.json", {})
    owner = resources.get("owner_campaign_id")
    if not owner:
        return
    if Path(owner).name != owner or owner in (".", ".."):
        raise RuntimeError("Campaign backup owner is invalid")
    parent = child.parent / owner
    name = next((k for k, p in job_roots(parent).items() if p == child), None)
    if name is None:
        raise RuntimeError("Campaign backup identifies an unknown child")
    _check_child_owner(parent, child, resources)
    with FileLock(str(parent / "backup.lock")):
        sources = read_json(parent / "child_snapshots.json", {})
        snapshot = read_json(child / "backup_snapshot.json")
        if not snapshot or snapshot.get("run_id") != child.name or snapshot.get("repos") != resources.get("repos"):
            raise RuntimeError("Campaign child backup identifies a different snapshot")
        sealed = read_json(_seal_path(parent, name))
        if sealed and sealed["snapshot"] != snapshot:
            raise RuntimeError("Cannot replace a sealed campaign child snapshot")
        sources[name] = snapshot
        atomic_json(parent / "child_snapshots.json", sources)
        backup(parent)


def seal_completed_job(root, name, child, c):
    """Freeze completed evidence after its final private backup; no new training occurs."""
    snapshot = verify_completed_child(child, c)
    _check_child_owner(root, child)
    index = read_json(Path(child) / "artifact_index.json")
    seal = {"campaign_id": Path(root).name, "job": name, "run_id": Path(child).name,
            "configuration_sha256": sha256(Path(child) / "config.json"),
            "completion_sha256": sha256(Path(child) / "completion.json"),
            "snapshot": snapshot, "files": index["files"]}
    _write_fixed(_seal_path(root, name), seal)
    snapshots = read_json(Path(root) / "child_snapshots.json", {})
    if snapshots.get(name) != snapshot:
        raise RuntimeError("Campaign completed-child snapshot was not recorded by its backup")
    return seal


def seal_continued20(root, c):
    """Bind reset20 only to the complete, privately backed-up continued20 run."""
    from .branch import resolve_branch_source
    root = Path(root)
    child = job_roots(root)["continued20"]
    completed = read_json(_seal_path(root, "continued20"))
    if not completed:
        raise RuntimeError("Continued20 must be sealed before binding reset20")
    snapshot = verify_completed_child(child, c["jobs"]["continued20"])
    if snapshot != completed["snapshot"]:
        raise RuntimeError("Continued20 backup changed after completion")
    verify_files(child, completed["files"])
    source = {"run_id": child.name, "repos": snapshot["repos"], "revisions": snapshot["revisions"],
              "artifact_index_sha256": snapshot["artifact_index_sha256"], "first_new_cycle": 2}
    old = read_json(root / "continued20_seal.json")
    if old:
        source = old["branch_source"]
        if source["run_id"] != child.name or source["revisions"] != snapshot["revisions"]:
            raise RuntimeError("Continued20 source seal identifies a different immutable backup")
    reset = copy.deepcopy(c["jobs"]["reset20"])
    reset["branch_source"] = source
    reset = resolve_branch_source(reset)
    source = reset["branch_source"]
    if not source.get("m2_tokens_sha256") or not source.get("m2_selection_sha256"):
        raise RuntimeError("Mixed shared source lacks verified cycle-2 mixture provenance")
    seal = {"campaign_id": root.name, "child_run_id": child.name,
            "runtime_sha256": read_json(root / "resources.json")["code_sha256"],
            "campaign_config_sha256": digest(c), "source_config_sha256": sha256(child / "config.json"),
            "source_completion_sha256": sha256(child / "completion.json"), "snapshot": snapshot,
            "branch_source": source, "reset_config_sha256": digest(reset),
            "exact_cycle2_mixture": {"training_tokens_sha256": source["m2_tokens_sha256"],
                                    "selection_sha256": source["m2_selection_sha256"]}}
    _write_fixed(root / "continued20_seal.json", seal)
    return reset, seal


def _phase(root, phase, active=None, **details):
    atomic_json(Path(root) / "status.json", {"state": "running", "phase": phase, "active_child": active,
                                             "time": now(), **details})
    event(root, "campaign_phase", phase=phase, active_child=active)


def _verify_shared20(root):
    roots = job_roots(root)
    configs = [read_json(roots[name] / "config.json") for name in ("continued20", "reset20")]
    if configs[0]["seed"] != configs[1]["seed"]:
        raise RuntimeError("Campaign shared seed differs")
    seed = f"seed_{configs[0]['seed']}"
    for g in (0, 1):
        for name in ("training.json", "metrics.json"):
            rel = f"{seed}/gen_{g:02d}/{name}"
            if sha256(roots["continued20"] / rel) != sha256(roots["reset20"] / rel):
                raise RuntimeError("Campaign shared S0/S1 evidence differs")
    left = read_json(roots["continued20"] / "synthetic_for_gen_02/manifest.json")
    right = read_json(roots["reset20"] / "synthetic_for_gen_02/manifest.json")
    if left["continuation_tokens_sha256"] != right["continuation_tokens_sha256"]:
        raise RuntimeError("Campaign shared D2 differs")
    result = {}
    for name, label in (("continued20", "continued"), ("reset20", "reset")):
        mixture = roots[name] / "mixed_for_gen_02"
        manifest = read_json(mixture / "manifest.json")
        if not manifest:
            raise RuntimeError("Campaign shared cycle-2 mixture is missing")
        verify_files(mixture, manifest["files"])
        result[label + "_tokens_sha256"] = manifest["training_tokens_sha256"]
        result[label + "_selection_sha256"] = sha256(mixture / "selection.json")
    if (result["continued_tokens_sha256"] != result["reset_tokens_sha256"] or
            result["continued_selection_sha256"] != result["reset_selection_sha256"]):
        raise RuntimeError("Campaign cycle-2 mixtures differ")
    return result


def run_campaign(root, validation=False):
    from filelock import FileLock
    from ..engine import ValidationInterruption
    from ..hub import backup
    from ..supervisor import run_worker
    root = Path(root)
    with FileLock(str(root / "worker.lock"), timeout=0):
        c, resources = read_json(root / "config.json"), read_json(root / "resources.json")
        validate_campaign_config(c)
        if not (root / "manifest.json").exists() and shutil.disk_usage(root).free < MINIMUM_INITIAL_FREE_BYTES:
            raise RuntimeError("Mixed campaign requires 40 GB free for three jobs, cache and recoverable optimizer state")
        _write_fixed(root / "manifest.json", {"config_hash": digest(c), "code_sha256": resources["code_sha256"], "kind": c["kind"]})
        roots = job_roots(root)
        _write_fixed(root / "children.json", {name: path.name for name, path in roots.items()})
        for position, name in enumerate(JOB_ORDER):
            child_config = c["jobs"][name]
            if name == "reset20":
                child_config, seal = seal_continued20(root, c)
                _phase(root, "source20_sealed")
                backup(root)
            if position:
                marker = root / f".injection_handoff_{position}_consumed"
                if validation and not marker.exists():
                    marker.touch()
                    event(root, "validation_handoff_interruption", before_job=name, position=position)
                    raise ValidationInterruption("Injected campaign coordinator exit after immutable handoff seal")
            _session(root, roots[name], child_config, owner_key="owner_campaign_id")
            child = roots[name]
            _check_child_owner(root, child)
            old_seal = read_json(_seal_path(root, name))
            if old_seal:
                verify_files(child, old_seal["files"])
            if not read_json(child / "completion.json", {}).get("complete"):
                _phase(root, name, child.name)
                run_worker(child, validation=validation)
            state = read_json(child / "status.json", {}).get("state")
            if state != "complete":
                atomic_json(root / "status.json", {"state": state or "failed", "phase": name,
                                                     "active_child": child.name, "time": now()})
                backup(root)
                return
            try:
                verify_completed_child(child, child_config)
            except FinalBackupPending:
                if _seal_path(root, name).exists():
                    raise RuntimeError("Sealed campaign child backup changed")
                backup(child)
                verify_completed_child(child, child_config)
            seal_completed_job(root, name, child, child_config)
            _phase(root, name + "_sealed")
            backup(root)
        _phase(root, "verifying")
        for name, child in roots.items():
            verify_files(child, read_json(_seal_path(root, name))["files"])
        shared = _verify_shared20(root)
        atomic_json(root / "shared20_mixture.json", shared)
        stages = sum(c["jobs"][name]["generations"] - (name != "continued20") for name in JOB_ORDER)
        if validation:
            from .profiles import protocol_fingerprint, mixture_checks
            audits = {name: read_json(child / "remote_validation.json") for name, child in roots.items()}
            for name in JOB_ORDER:
                audit = audits[name]
                job = read_json(roots[name] / "config.json")
                if (not audit or audit.get("source_code_sha256") != resources["code_sha256"]
                        or audit.get("protocol_fingerprint") != protocol_fingerprint(job)
                        or any(audit.get("checks", {}).get(k) is not True for k in mixture_checks(job, include_operational=False))):
                    raise RuntimeError("All campaign child GPU audits must pass for their actual protocols")
            trained = [read_json(roots[name] / f"seed_42/gen_{g:02d}/training.json")
                       for name, gs in (("reset50", (2, 3)), ("continued20", (1, 2)), ("reset20", (2, 3))) for g in gs]
            if any(t["selected_epoch"] != 5 or t["executed_epochs"] != 5 for t in trained):
                raise RuntimeError("Campaign smoke must retain six actual fifth-epoch checkpoints")
            events = [json.loads(line)["event"] for line in (root / "events.jsonl").read_text().splitlines()]
            for child in roots.values():
                _check_child_owner(root, child, require_sealed=True)
            checks = {"three_child_audits": True, "six_five_epoch_training_stages": len(trained) == 6,
                      "epoch_five_checkpoints": True, "exact_shared_prefix_and_cycle2_mixture": True,
                      "two_handoff_restarts": (all((root / f".injection_handoff_{p}_consumed").exists() for p in (1, 2))
                                                 and events.count("validation_handoff_interruption") == 2
                                                 and events.count("validation_worker_resumed") == 2),
                      "completed_evidence_unchanged": True, "single_owner_and_deadline": True,
                      "three_private_restores": all(a["checks"]["restored_checkpoint_tensor_hash_and_evaluation"] for a in audits.values())}
            if not all(checks.values()):
                raise RuntimeError("Campaign GPU acceptance is incomplete")
            atomic_json(root / "remote_validation.json", {"workflow": "mixed_campaign", "checks": checks,
                        "source_code_sha256": resources["code_sha256"], "protocol_fingerprint": protocol_fingerprint(c),
                        "child_audits": {name: sha256(child / "remote_validation.json") for name, child in roots.items()},
                        "child_certificates": audits, "shared20_mixture": shared,
                        "unique_training_stages": stages, "executed_epochs": stages * 5})
        report_campaign(root, for_backup=True)
        atomic_json(root / "status.json", {"state": "complete", "phase": "complete", "time": now()})
        atomic_json(root / "completion.json", {"complete": True, "workflow": "mixed_campaign", "time": now(),
                    "children": {name: child.name for name, child in roots.items()},
                    "source20_seal_sha256": sha256(root / "continued20_seal.json"),
                    "unique_training_stages": stages, "executed_epochs": stages * 5})
        backup(root)


def fetch_campaign(root, resources, metadata_only=False):
    from ..hub import fetch
    root = Path(root)
    result = fetch(root.name, resources["repos"], root, metadata_only=metadata_only)
    snapshots = read_json(root / "child_snapshots.json", {})
    children = {}
    for name, child in job_roots(root).items():
        snapshot = snapshots.get(name)
        if snapshot is None:
            continue
        if snapshot.get("run_id") != child.name or snapshot.get("repos") != resources["repos"]:
            raise RuntimeError("Campaign snapshot identifies a different child or repository")
        receipt = fetch(child.name, resources["repos"], child, metadata_only=metadata_only, revisions=snapshot["revisions"])
        if receipt["artifact_index_sha256"] != snapshot["artifact_index_sha256"]:
            raise RuntimeError("Fetched campaign child differs from its immutable backup")
        atomic_json(child / "recovery_verified.json", receipt)
        children[name] = receipt
    result["children"] = children
    atomic_json(root / "recovery_verified.json", result)
    return result


def verified_resume_jobs(root, resources, c):
    """Discount only fully sealed jobs recovered from fresh pinned private snapshots.

    Deep checkpoint/token verification stays on the retained volume before reuse.
    Unsealed partial jobs retain a conservative full-job forecast.
    """
    from ..hub import verify_recovered_files
    root = Path(root)
    receipt = fetch_campaign(root, resources, metadata_only=True)
    parent_index = read_json(root / "artifact_index.json")
    if (receipt.get("run_id") != root.name or receipt.get("repos") != resources["repos"]
            or receipt.get("artifact_index_sha256") != sha256(root / "artifact_index.json")
            or digest(read_json(root / "config.json")) != digest(c)
            or read_json(root / "manifest.json", {}).get("config_hash") != digest(c)
            or read_json(root / "manifest.json", {}).get("code_sha256") != resources["code_sha256"]):
        raise RuntimeError("Campaign resume metadata does not match the pinned parent configuration")
    verify_recovered_files(root, parent_index["files"])
    snapshots = read_json(root / "child_snapshots.json", {})
    completed = []
    for name, child in job_roots(root).items():
        seal_rel = f"job_seals/{name}.json"
        if seal_rel not in parent_index["files"]:
            continue
        seal, snapshot = read_json(root / seal_rel), snapshots.get(name)
        recovery = read_json(child / "recovery_verified.json", {})
        child_index = read_json(child / "artifact_index.json", {})
        job = read_json(child / "config.json", {})
        done = read_json(child / "completion.json", {})
        if (not snapshot or seal.get("snapshot") != snapshot or snapshot.get("run_id") != child.name
                or snapshot.get("repos") != resources["repos"] or recovery.get("repos") != snapshot["repos"]
                or recovery.get("revisions") != snapshot["revisions"]
                or recovery.get("artifact_index_sha256") != snapshot["artifact_index_sha256"]
                or sha256(child / "artifact_index.json") != snapshot["artifact_index_sha256"]
                or seal.get("files") != child_index.get("files")
                or seal.get("configuration_sha256") != sha256(child / "config.json")
                or seal.get("completion_sha256") != sha256(child / "completion.json")
                or not done.get("complete") or done.get("generations") != c["jobs"][name]["generations"]
                or read_json(child / "manifest.json", {}).get("config_hash") != digest(job)
                or protocol_identity(job) != protocol_identity(c["jobs"][name])):
            raise RuntimeError("Campaign resume job seal is stale or its completed metadata differs")
        small = {rel: meta for rel, meta in child_index["files"].items()
                 if not rel.endswith((".npy", ".safetensors"))}
        verify_recovered_files(child, small)
        _check_child_owner(root, child, require_sealed=True)
        for g in range(1 if name == "continued20" else 2, job["generations"] + 1):
            prefix = f"seed_{job['seed']}/gen_{g:02d}/"
            training = read_json(child / prefix / "training.json")
            history = read_json(child / prefix / "training_history.json")
            stage = read_json(child / prefix / "stage_completion.json")
            verify_selection(training, history, job)
            if not stage or stage.get("selected_weights_sha256") != training["selected_weights_sha256"]:
                raise RuntimeError("Campaign resume completed-stage checkpoint provenance differs")
            for rel, checksum in stage["files"].items():
                if child_index["files"].get(prefix + rel, {}).get("sha256") != checksum:
                    raise RuntimeError("Campaign resume completed-stage index differs")
        completed.append(name)
    return {"sealed_completed_jobs": completed, "parent_revisions": receipt["revisions"],
            "parent_artifact_index_sha256": receipt["artifact_index_sha256"],
            "scope": "Pinned complete-job metadata and stage/index checks; unsealed partial jobs remain fully forecast. Bulk verification occurs on the worker."}


def protocol_identity(c):
    from .profiles import protocol_fingerprint
    return protocol_fingerprint(c)


def verify_campaign_recovery(root):
    from ..continued import verify_recovery
    root = Path(root)
    recovered = read_json(root / "recovery_verified.json", {}).get("children", {})
    snapshots = read_json(root / "child_snapshots.json", {})
    if set(recovered) != set(JOB_ORDER) or set(snapshots) != set(JOB_ORDER):
        raise RuntimeError("All three campaign child recovery receipts and snapshots are required")
    source_seal = read_json(root / "continued20_seal.json", {})
    if source_seal.get("snapshot") != snapshots["continued20"]:
        raise RuntimeError("Campaign recovery does not match the sealed continued20 source")
    for name, child in job_roots(root).items():
        snapshot, receipt = snapshots[name], read_json(child / "recovery_verified.json", {})
        seal = read_json(_seal_path(root, name), {})
        if (snapshot.get("run_id") != child.name or receipt != recovered[name]
                or receipt.get("run_id") != child.name or receipt.get("repos") != snapshot["repos"]
                or receipt.get("revisions") != snapshot["revisions"]
                or receipt.get("artifact_index_sha256") != snapshot["artifact_index_sha256"]
                or sha256(child / "artifact_index.json") != snapshot["artifact_index_sha256"]
                or seal.get("snapshot") != snapshot):
            raise RuntimeError("Campaign child recovery is stale or belongs to a different snapshot")
        _check_child_owner(root, child, require_sealed=True)
        verify_recovery(child)


def report_campaign(root, out=None, for_backup=False):
    """Report into a new parent destination without rewriting sealed child evidence."""
    import csv
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..analysis.reporting import Evidence, _analysis_identity, _output_directory, _rows, report as arm_report
    root = Path(root)
    out = _output_directory(root, out, root.name, for_backup=for_backup)
    out.mkdir(parents=True, exist_ok=True)
    rows, inputs = [], {}
    for name, child in job_roots(root).items():
        if not (child / "config.json").is_file():
            continue
        evidence = Evidence(child)
        c = evidence.read(child / "config.json")
        seed_root = child / f"seed_{c['seed']}"
        metrics = [evidence.read(p) for p in sorted(seed_root.glob("gen_*/metrics.json"))]
        if not metrics:
            continue
        for m in metrics:
            if m["generation"]:
                verify_selection(m["training"], evidence.read(seed_root / f"gen_{m['generation']:02d}/training_history.json"), c)
        values, _, _ = _rows(metrics, seed_root, evidence, c)
        for row in values:
            row.update(job=name, synthetic_fraction=c["synthetic_fraction"], initialization=c["initialization"])
            rows.append(row)
        inputs.update({f"{child.name}/{rel}": meta for rel, meta in evidence.files.items()})
        arm_report(child, out / name)
    if not rows:
        raise RuntimeError("No campaign metrics to report")
    for rel in ("config.json", "manifest.json", "continued20_seal.json", "shared20_mixture.json", "children.json", "child_snapshots.json"):
        path = root / rel
        if path.is_file():
            inputs[f"{root.name}/{rel}"] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    for path in sorted((root / "job_seals").glob("*.json")):
        inputs[f"{root.name}/{path.relative_to(root).as_posix()}"] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, key in zip(axes.flat, ("test_perplexity", "validation_perplexity", "distinct_2", "distinct_3", "repeated_4gram_rate", "token_entropy_nats")):
        for name in JOB_ORDER:
            for origin, style in (("current_run", "o-"), ("imported_shared_stage", "s--")):
                selected = [row for row in rows if row["job"] == name and row.get("stage_origin") == origin]
                if not selected:
                    continue
                # Historical full-data prefix measurements are not a tiny smoke baseline.
                if read_json(root / "config.json")["kind"] == "smoke" and name == "reset50" and origin == "imported_shared_stage":
                    continue
                ax.plot([row["generation"] for row in selected], [row.get(key, math.nan) for row in selected], style,
                        label=name + (" (imported source)" if origin == "imported_shared_stage" else ""))
        ax.set_title(key.replace("_", " "))
        ax.set_xlabel("Synthetic training cycle")
        ax.grid(alpha=.2)
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)
    (out / "report.md").write_text(
        "# Mixed-fraction final-epoch campaign\n\n"
        "Three sequential logical jobs share one GPU session. Reset50 imports the historical 50/50 S0/S1/D2; "
        "continued20 and reset20 share their newly trained S1, generated D2 and exact cycle-2 mixture. "
        "Every new stage executes five epochs and retains epoch 5. Imported work is not new execution.\n\n"
        "Imported points retain their original evaluation scope and execution identity. In smoke reports, reset50's "
        "full-data S0/S1 points remain in the CSV but are omitted from the combined plot; the newly executed reset50 "
        "points use the small active evaluation split. They must not be read as one matched evaluation trajectory.\n\n"
        "![Campaign metrics](comparison.png)\n\n"
        "Per-job tables and graph families: [50/50 reset](reset50/report.md), "
        "[20/80 continued](continued20/report.md), [20/80 reset](reset20/report.md).\n\n"
        "Smoke trajectories validate the pipeline; full trajectories describe one seed. Pooled diversity, within-output "
        "repetition, formatting sensitivity and real-text prediction are distinct measurements.\n", encoding="utf-8")
    atomic_json(out / "analysis_manifest.json", {"inputs": inputs, "analysis": _analysis_identity(),
                "renderer": {"path": "model_collapse/workflow/mixed_campaign.py", "sha256": sha256(Path(__file__))},
                "source_runtime_sha256": read_json(root / "manifest.json", {}).get("code_sha256"),
                "campaign_config_sha256": sha256(root / "config.json"),
                "outputs": {p.relative_to(out).as_posix(): sha256(p) for p in out.rglob("*")
                            if p.is_file() and p != out / "analysis_manifest.json"}})
    return {"report": str(out / "report.md"), "metrics_rows": len(rows)}
