"""Two logical experiment arms under one detached, budgeted provider owner."""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

from ..common import STATE, atomic_json, digest, event, now, read_json, sha256, verify_files
from .selection import FINAL, selection_policy, verify_selection


class FinalBackupPending(RuntimeError):
    """Stages are complete; their final immutable backup still needs publishing."""


def validate_pair_config(c):
    from ..common import validate_config
    if c.get("kind") not in ("full", "smoke") or set(c.get("arms", {})) != {"continued", "reset"}:
        raise ValueError("A paired configuration requires continued and reset arms")
    expected = {"continued": "continued", "reset": "reset_s0"}
    for name, mode in expected.items():
        arm = c["arms"][name]
        if arm.get("workflow") is not None or arm.get("initialization") != mode or arm["kind"] != c["kind"]:
            raise ValueError("Paired children must be ordinary matching-kind experiment configurations")
        validate_config(arm)
        if selection_policy(arm) != FINAL or arm["epochs"] != 5:
            raise ValueError("This paired protocol requires five epochs and final_epoch selection")
        end = 10 if c["kind"] == "full" else (2 if name == "continued" else 3)
        if arm["generations"] != end:
            raise ValueError("Paired endpoints must be full 10/10 or smoke 2/3")
        if arm.get("required_gpu") != c.get("required_gpu") or arm.get("required_environment") != c.get("required_environment"):
            raise ValueError("Both arms must use the parent's matched GPU/software environment")
        if arm.get("required_image") != c.get("required_image"):
            raise ValueError("Both arms must use the parent's pinned image")
    a, b = (copy.deepcopy(c["arms"][key]) for key in ("continued", "reset"))
    if not a.get("initial_dataset") or b.get("initial_dataset"):
        raise ValueError("Only continued imports the pinned original D1")
    for value in (a, b):
        for key in ("initialization", "initial_dataset", "branch_source", "experiment_id", "generations"):
            value.pop(key, None)
    if a != b:
        raise ValueError("Paired scientific settings must match apart from initialization")


def resolve_pair_inputs(c):
    from .initial_data import resolve_initial_dataset
    c = copy.deepcopy(c)
    c["arms"]["continued"] = resolve_initial_dataset(c["arms"]["continued"])
    # Input pinning adds evidence only to continued, which is intentionally arm-specific.
    validate_pair_config(c)
    return c


def child_roots(root):
    root = Path(root)
    return {"continued": root.with_name(root.name + "-continued"),
            "reset": root.with_name(root.name + "-reset-s0")}


def _write_fixed(path, value):
    old = read_json(path)
    if old is not None and old != value:
        raise RuntimeError(f"Recorded paired state changed: {Path(path).name}")
    if old is None:
        atomic_json(path, value)


def _session(root, child, c):
    root, child = Path(root), Path(child)
    _write_fixed(child / "config.json", c)
    old = read_json(child / "resources.json")
    if not read_json(child / "completion.json", {}).get("complete"):
        resources = {**read_json(root / "resources.json"), "owner_pair_id": root.name, "logical_run_id": child.name}
        atomic_json(child / "resources.json", resources)
    elif old is None:
        raise RuntimeError("Completed child has no original resource provenance")
    requirements = root / "requirements-tested.txt"
    if requirements.exists() and not (child / requirements.name).exists():
        shutil.copyfile(requirements, child / requirements.name)


def verify_completed_child(child, c):
    from .branch import import_shared_prefix
    from .lineage import training_lineage, verify_completed_stage
    child = Path(child)
    completion = read_json(child / "completion.json", {})
    if (not completion.get("complete") or completion.get("generations") != c["generations"] or
            read_json(child / "status.json", {}).get("state") != "complete"):
        raise RuntimeError("Child has not completed its expected endpoint")
    if read_json(child / "manifest.json", {}).get("config_hash") != digest(c):
        raise RuntimeError("Completed child configuration differs")
    data_manifest = read_json(child / "data/manifest.json")
    verify_files(child / "data", data_manifest["files"])
    if read_json(child / "manifest.json")["data_manifest_sha256"] != sha256(child / "data/manifest.json"):
        raise RuntimeError("Completed child real split provenance differs")
    if c["initialization"] == "reset_s0":
        import_shared_prefix(child, c)
    for g in range(2 if c["initialization"] == "reset_s0" else 0, c["generations"] + 1):
        gen = child / f"seed_{c['seed']}/gen_{g:02d}"
        t = read_json(gen / "training.json")
        verify_completed_stage(child, g, c, t)
        if g:
            verify_selection(t, read_json(gen / "training_history.json"), c)
            blocks, expected, checkpoint, lineage = training_lineage(child, g, c, None)
            if t["initial_weights_sha256"] != expected or t["lineage"] != lineage:
                raise RuntimeError("Completed child training lineage differs")
    snapshot = read_json(child / "backup_snapshot.json")
    index = read_json(child / "artifact_index.json")
    if (not snapshot or not index or snapshot["artifact_index_sha256"] != sha256(child / "artifact_index.json") or
            index["run_id"] != child.name or snapshot["run_id"] != child.name):
        raise FinalBackupPending("Completed child final backup is not sealed")
    verify_files(child, index["files"])
    for name in ("completion.json", "config.json", "manifest.json"):
        if index["files"].get(name, {}).get("sha256") != sha256(child / name):
            raise FinalBackupPending("Final backup does not contain the completed child evidence")
    return snapshot


def record_child_snapshot(child):
    """Called after each successful child backup; parent stores small pinned references."""
    from filelock import FileLock
    from ..hub import backup
    child = Path(child)
    resources = read_json(child / "resources.json", {})
    if not resources.get("owner_pair_id"):
        return
    parent = child.parent / resources["owner_pair_id"]
    roots = child_roots(parent)
    name = next((name for name, path in roots.items() if path == child), None)
    if name is None or resources.get("logical_run_id") != child.name:
        raise RuntimeError("Child backup owner association differs")
    with FileLock(str(parent / "backup.lock")):
        sources = read_json(parent / "child_snapshots.json", {})
        snapshot = read_json(child / "backup_snapshot.json")
        if name == "continued" and (parent / "continued_seal.json").exists() and sources.get(name) != snapshot:
            raise RuntimeError("Cannot replace a sealed continued snapshot")
        sources[name] = snapshot
        atomic_json(parent / "child_snapshots.json", sources)
        backup(parent)


def seal_continued(root, c):
    from .branch import resolve_branch_source
    root = Path(root)
    child = child_roots(root)["continued"]
    snapshot = verify_completed_child(child, c["arms"]["continued"])
    old = read_json(root / "continued_seal.json")
    source = {"run_id": child.name, "repos": snapshot["repos"], "revisions": snapshot["revisions"],
              "artifact_index_sha256": snapshot["artifact_index_sha256"], "first_new_cycle": 2}
    if old:
        source = old["branch_source"]
        if source["run_id"] != child.name or source["revisions"] != snapshot["revisions"]:
            raise RuntimeError("Continued seal identifies a different child backup")
    reset = copy.deepcopy(c["arms"]["reset"])
    reset["branch_source"] = source
    reset = resolve_branch_source(reset)
    seal = {"pair_id": root.name, "child_run_id": child.name,
            "runtime_sha256": read_json(root / "resources.json")["code_sha256"],
            "pair_config_sha256": digest(c), "source_config_sha256": sha256(child / "config.json"),
            "source_completion_sha256": sha256(child / "completion.json"),
            "branch_source": reset["branch_source"], "snapshot": snapshot,
            "continued_files": read_json(child / "artifact_index.json")["files"],
            "reset_config_sha256": digest(reset)}
    _write_fixed(root / "continued_seal.json", seal)
    return reset, seal


def _phase(root, phase, active=None, **details):
    atomic_json(Path(root) / "status.json", {"state": "running", "phase": phase, "active_child": active,
                                            "time": now(), **details})
    event(root, "pair_phase", phase=phase, active_child=active)


def run_pair(root, validation=False):
    from filelock import FileLock
    from ..engine import ValidationInterruption
    from ..hub import backup
    from ..supervisor import run_worker
    root = Path(root)
    with FileLock(str(root / "worker.lock"), timeout=0):
        c, resources = read_json(root / "config.json"), read_json(root / "resources.json")
        validate_pair_config(c)
        if not (root / "manifest.json").exists() and shutil.disk_usage(root).free < 20_000_000_000:
            raise RuntimeError("Insufficient volume space for both arms and recoverable optimizer state")
        _write_fixed(root / "manifest.json", {"config_hash": digest(c), "code_sha256": resources["code_sha256"], "kind": c["kind"]})
        roots = child_roots(root)
        _write_fixed(root / "children.json", {name: path.name for name, path in roots.items()})
        _session(root, roots["continued"], c["arms"]["continued"])
        for name in ("continued", "reset"):
            if name == "reset":
                reset, seal = seal_continued(root, c)
                _phase(root, "source_sealed")
                backup(root)
                marker = root / ".injection_handoff_consumed"
                if validation and not marker.exists():
                    marker.touch()
                    event(root, "validation_handoff_interruption", seal_sha256=sha256(root / "continued_seal.json"))
                    raise ValidationInterruption("Injected coordinator exit after immutable source seal")
                _session(root, roots[name], reset)
            child = roots[name]
            child_config = read_json(child / "config.json")
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
                if name == "continued" and (root / "continued_seal.json").exists():
                    raise RuntimeError("Sealed continued backup changed")
                backup(child)
                verify_completed_child(child, child_config)
        _phase(root, "verifying")
        seal = read_json(root / "continued_seal.json")
        verify_files(roots["continued"], seal["continued_files"])
        if validation:
            audits = {name: read_json(child / "remote_validation.json") for name, child in roots.items()}
            if any(not a or not all(a["checks"].values()) for a in audits.values()):
                raise RuntimeError("Both child GPU audits must pass")
            stages = [read_json(roots[name] / f"seed_42/gen_{g:02d}/training.json")
                      for name, gs in (("continued", (1, 2)), ("reset", (2, 3))) for g in gs]
            if any(t["selected_epoch"] != 5 or t["executed_epochs"] != 5 for t in stages):
                raise RuntimeError("GPU smoke did not retain all four fifth-epoch checkpoints")
            import json
            events = [json.loads(line)["event"] for line in (root / "events.jsonl").read_text().splitlines()]
            checks = {"both_child_audits": True, "four_five_epoch_training_stages": len(stages) == 4,
                      "epoch_five_checkpoints": True, "exact_shared_prefix": True,
                      "handoff_restart": ((root / ".injection_handoff_consumed").exists() and
                                          events.count("validation_handoff_interruption") == 1 and
                                          events.count("validation_worker_resumed") == 1),
                      "continued_evidence_unchanged": True, "single_owner_and_deadline": True,
                      "both_private_restores": all(a["checks"]["restored_checkpoint_tensor_hash_and_evaluation"] for a in audits.values())}
            for child in roots.values():
                r = read_json(child / "resources.json")
                for key in ("run_id", "launch_id", "pod_id", "volume_id", "created_at", "session_allowance_usd", "guard_hourly_usd"):
                    if r[key] != resources[key]:
                        raise RuntimeError("Child owns a different provider session or budget deadline")
            if not all(checks.values()):
                raise RuntimeError("Paired validation did not verify actual handoff recovery")
            from .profiles import protocol_fingerprint
            atomic_json(root / "remote_validation.json", {"workflow": "paired", "checks": checks,
                        "source_code_sha256": resources["code_sha256"], "protocol_fingerprint": protocol_fingerprint(c),
                        "child_audits": {name: sha256(child / "remote_validation.json") for name, child in roots.items()},
                        "unique_training_stages": 4, "executed_epochs": 20})
        report_pair(root, for_backup=True)
        atomic_json(root / "status.json", {"state": "complete", "phase": "complete", "time": now()})
        atomic_json(root / "completion.json", {"complete": True, "workflow": "paired", "time": now(),
                    "children": {name: child.name for name, child in roots.items()},
                    "source_seal_sha256": sha256(root / "continued_seal.json"),
                    "unique_training_stages": 4 if validation else 19, "executed_epochs": 20 if validation else 95})
        backup(root)


def fetch_pair(root, resources, metadata_only=False):
    from ..hub import fetch
    root = Path(root)
    result = fetch(root.name, resources["repos"], root, metadata_only=metadata_only)
    snapshots = read_json(root / "child_snapshots.json", {})
    children = {}
    for name, child in child_roots(root).items():
        snapshot = snapshots.get(name)
        if snapshot is None:
            continue
        receipt = fetch(child.name, resources["repos"], child, metadata_only=metadata_only, revisions=snapshot["revisions"])
        if receipt["artifact_index_sha256"] != snapshot["artifact_index_sha256"]:
            raise RuntimeError("Fetched child differs from the pinned paired snapshot")
        atomic_json(child / "recovery_verified.json", receipt)
        children[name] = receipt
    result["children"] = children
    atomic_json(root / "recovery_verified.json", result)
    return result


def verify_pair_recovery(root):
    from ..continued import verify_recovery
    root = Path(root)
    if set(read_json(root / "recovery_verified.json", {}).get("children", {})) != {"continued", "reset"}:
        raise RuntimeError("Both child recovery receipts are required before paired cleanup")
    snapshots = read_json(root / "child_snapshots.json", {})
    seal = read_json(root / "continued_seal.json", {})
    if set(snapshots) != {"continued", "reset"} or seal.get("snapshot") != snapshots["continued"]:
        raise RuntimeError("Paired recovery does not match its sealed source snapshots")
    for name, child in child_roots(root).items():
        receipt = read_json(child / "recovery_verified.json", {})
        snapshot = snapshots[name]
        if (snapshot.get("run_id") != child.name or receipt.get("run_id") != child.name or receipt.get("repos") != snapshot["repos"] or
                receipt.get("revisions") != snapshot["revisions"] or
                receipt.get("artifact_index_sha256") != snapshot["artifact_index_sha256"] or
                sha256(child / "artifact_index.json") != snapshot["artifact_index_sha256"]):
            raise RuntimeError("Child recovery receipt or index is stale or belongs to a different snapshot")
        verify_recovery(child)


def validate_pair_local(root):
    from ..hub import verify_recovered_files
    from ..remote import local_resources, source_archive
    from ..runpod_api import Runpod
    from .profiles import PAIR_CHECKS, protocol_fingerprint
    root = Path(root)
    verify_pair_recovery(root)
    verify_recovered_files(root, read_json(root / "artifact_index.json")["files"])
    result = read_json(root / "remote_validation.json")
    r = local_resources(root.name)
    guard, pod = read_json(root / "watchdog_exit.json", {}), Runpod().pod(r["pod_id"])
    if guard.get("pod_id") != r["pod_id"] or guard.get("reason") != "validation_budget_guard" or pod["status"] != "EXITED":
        raise RuntimeError("Paired watchdog shutdown is not independently verified")
    if result["source_code_sha256"] != source_archive()[1]:
        raise RuntimeError("Runtime changed after paired GPU acceptance")
    detached = read_json(root / "detached_verification.json", {})
    if detached.get("pod_id") != r["pod_id"] or detached.get("progress_after_disconnect") is not True:
        raise RuntimeError("Detached continued progress has not been verified")
    result["checks"].update(detached_progress_verified=True, independent_budget_guard_and_provider_stop_verified=True)
    c = read_json(root / "config.json")
    if result["protocol_fingerprint"] != protocol_fingerprint(c) or any(result["checks"].get(k) is not True for k in PAIR_CHECKS):
        raise RuntimeError("Paired acceptance is incomplete")
    result.update(validated_run_id=root.name, provider_pod_state=pod["status"])
    atomic_json(root / "validation_results.json", result)
    atomic_json(STATE / "validation_paired.json", result)
    benchmark = read_json(child_roots(root)["continued"] / "benchmark.json")
    if benchmark:
        atomic_json(STATE / "benchmark.json", benchmark)
    return result


def report_pair(root, out=None, for_backup=False):
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
    for arm, child in child_roots(root).items():
        evidence = Evidence(child)
        child_config = evidence.read(child / "config.json")
        seed_root = child / f"seed_{child_config['seed']}"
        metrics = [evidence.read(p) for p in sorted(seed_root.glob("gen_*/metrics.json"))]
        if not metrics:
            continue
        for m in metrics:
            if m["generation"]:
                history = evidence.read(seed_root / f"gen_{m['generation']:02d}/training_history.json")
                verify_selection(m["training"], history, child_config)
        arm_rows, imports, refs = _rows(metrics, seed_root, evidence, child_config)
        for row in arm_rows:
            row["arm"] = arm
            reference = evidence.read(seed_root / f"gen_{row['generation']:02d}/reference_sequences.json")
            if reference and reference.get("sequence_mean_nll"):
                import numpy as np
                row["frozen_s0_block_nll_p95"] = float(np.percentile(reference["sequence_mean_nll"], 95))
            rows.append(row)
        inputs.update({f"{child.name}/{rel}": meta for rel, meta in evidence.files.items()})
        # New derived files live under the parent output; sealed child artifacts remain untouched.
        arm_report(child, out / arm)
    if not rows:
        raise RuntimeError("No paired metrics to report")
    roots = child_roots(root)
    for g in (0, 1):
        rel = f"seed_42/gen_{g:02d}/metrics.json"
        if all((child / rel).exists() for child in roots.values()):
            if sha256(roots["continued"] / rel) != sha256(roots["reset"] / rel):
                raise RuntimeError("Cannot report mismatched shared-stage measurements")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    keys = ("test_perplexity", "validation_perplexity", "distinct_2", "distinct_3", "repeated_4gram_rate", "token_entropy_nats")
    for ax, key in zip(axes.flat, keys):
        for arm in ("continued", "reset"):
            chosen = [r for r in rows if r["arm"] == arm]
            ax.plot([r["generation"] for r in chosen], [r.get(key, math.nan) for r in chosen], marker="o", label=arm + " (S0/S1 shared)")
        ax.set_title(key.replace("_", " "))
        ax.set_xlabel("Synthetic training cycle")
        ax.grid(alpha=.2)
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)
    text = ("# Matched final-epoch comparison\n\n"
            "S0 and the newly trained S1 are shared. Every synthetic stage executes and retains epoch 5; validation is diagnostic. "
            "Continued weights carry five updates per cycle; reset always initializes from S0.\n\n"
            "![Paired metrics](comparison.png)\n\n"
            "Complete metric tables and existing graph families: [continued](continued/report.md), [reset](reset/report.md). "
            "The CSV includes corpus NLL/PPL, diversity, duplicates, entropy, repetition, frozen-S0 scores, token counts, selected epochs and timing.\n\n"
            "Smoke results validate the pipeline and do not establish model collapse. Full results use one seed and are descriptive. "
            "Token entropy measures pooled token frequencies; it is not predictive entropy or within-output repetition.\n")
    (out / "report.md").write_text(text, encoding="utf-8")
    atomic_json(out / "analysis_manifest.json", {"inputs": inputs, "analysis": _analysis_identity(),
                "pair_config_sha256": sha256(root / "config.json"),
                "outputs": {p.relative_to(out).as_posix(): sha256(p) for p in out.rglob("*")
                            if p.is_file() and p != out / "analysis_manifest.json"}})
    return {"report": str(out / "report.md"), "metrics_rows": len(rows)}
