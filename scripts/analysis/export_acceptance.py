"""Publish small, checksummed GPU acceptance evidence; never copy bulk artifacts.

Run only after fetch --metadata-only, validate, and cleanup have succeeded.
This reads local files and generates figures; it makes no cloud requests.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from model_collapse.analysis.forecasts import forecast
from model_collapse.common import PROJECT, atomic_json, read_json, sha256
from model_collapse.hub import verify_recovered_files
from model_collapse.report import report


def export(run_id, *, superseded=False, offline_tests=None):
    root = PROJECT / "runs" / run_id
    dest = PROJECT / "reports" / run_id
    if Path(run_id).name != run_id or not root.is_dir():
        raise ValueError("Expected an existing local run ID")
    if dest.exists() and any(p.name != "generated" for p in dest.iterdir()):
        raise ValueError("Published acceptance already exists; its evidence must stay immutable")
    validation = read_json(root / "validation_results.json")
    cleanup = read_json(root / "cleanup_verified.json")
    completion = read_json(root / "completion.json")
    if not validation or not all(validation["checks"].values()) or not completion.get("complete"):
        raise RuntimeError("Completed, successful acceptance evidence is required")
    if not cleanup or not cleanup.get("pod_removed") or not cleanup.get("volume_removed"):
        raise RuntimeError("Temporary cloud resource cleanup must be verified first")
    c, r = read_json(root / "config.json"), read_json(root / "resources.json")
    recovery = read_json(root / "recovery_snapshot.json")
    index = read_json(root / "artifact_index.json")
    selected = {}
    for rel, meta in index["files"].items():
        if rel.endswith(".json") and not any(part in ("checkpoint", "tokenizer", "generated") for part in Path(rel).parts):
            selected[rel] = meta
    verify_recovered_files(root, selected)
    files = {}
    for rel, meta in selected.items():
        source = root / rel
        if not source.is_file() or sha256(source) != meta["sha256"]:
            redirect = recovery["preserved_control_files"][rel]["recovered_path"]
            source = root / redirect
        target = dest / "records" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files["records/" + rel] = {**meta, "integrity_scope": "remote_artifact_index"}
    for name in ("artifact_index.json", "validation_results.json", "recovery_snapshot.json", "recovery_verified.json",
                 "watchdog_exit.json", "launch_verified.json", "events.jsonl", "requirements-tested.txt"):
        if not (root / name).is_file():
            continue
        target = dest / "records" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
        files["records/" + name] = {"sha256": sha256(target), "bytes": target.stat().st_size,
                                   "integrity_scope": "publication_snapshot_outside_remote_index"}
    generated = report(root)
    generated_root = Path(generated["report"]).parent
    for source in generated_root.iterdir():
        if source.is_file():
            target = dest / "figures" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            files["figures/" + source.name] = {"sha256": sha256(target), "bytes": target.stat().st_size,
                                            "integrity_scope": "derived_analysis"}
    b = read_json(root / "benchmark.json")
    rate = r["quoted_gpu_hourly_usd"] + 30 * .07 / 730 + 20 * .10 / 730
    estimate = forecast(b, 10, rate, initialization="reset_s0")
    elapsed = (completion["time"] - r["created_at"]) / 60
    stages = []
    for g in range(2, c["generations"] + 1):
        m = read_json(root / f"seed_{c['seed']}/gen_{g:02d}/metrics.json")
        t = m["training"]
        stages.append({"cycle": g, "test_perplexity": m["test"]["perplexity"],
                       "initial_weights_sha256": t["initial_weights_sha256"],
                       "selected_weights_sha256": t["selected_weights_sha256"],
                       "generator_weights_sha256": t["lineage"]["generator_weights_sha256"],
                       "input_tokens_sha256": t["input_tokens_sha256"]})
    receipt = {"run_id": run_id, "source_git_commit": r.get("git_commit"), "runtime_sha256": r["code_sha256"],
               "status": "historical_acceptance_superseded" if superseded else "accepted_for_full_launch",
               "finished_utc": datetime.fromtimestamp(completion["time"], timezone.utc).isoformat(),
               "minutes_provisioning_to_pipeline_complete": elapsed,
               "observed_account_debit_usd": cleanup["actual_account_balance_delta_usd"],
               "account_debit_scope": "Observed account balance difference, includes concurrent retained storage; not an itemized invoice.",
               "pod_id": r["pod_id"], "volume_id": r["volume_id"], "pod_removed": True, "volume_removed": True,
               "checks": validation["checks"], "offline_tests_passed": offline_tests,
               "restored_test": validation["restored_test"], "max_nll_error": validation["max_nll_calculation_error"],
               "recovery": {"metadata_verified_files": recovery["verified_files"], "local_bytes": recovery["verified_bytes"],
                            "bulk_restored_and_verified_remotely": validation["artifact_recovery"]["verified_bytes"],
                            "bulk_files_verified_remotely": validation["artifact_recovery"]["verified_files"]},
               "source_revisions": c["branch_source"]["revisions"], "stages": stages,
               "remaining_full_run_forecast": estimate, "full_experiment_launched": False}
    atomic_json(dest / "execution_receipt.json", receipt)
    lines = ["# Shared-start reset branch: GPU acceptance", "", f"Run: `{run_id}`.", "",
             "**Pipeline validation only. No full-data reset experiment was launched.**", ""]
    if superseded:
        lines += ["This successful acceptance covers its recorded source version. A subsequent source version is validated separately; this receipt must not authorize that changed runtime.", ""]
    lines += ["## What ran", "",
              "Imported the exact Week 2 S0 and S1 checkpoints and D2 source dataset. The two new cycles, 2 and 3, each trained from fixed S0 with a fresh optimizer for one epoch on 256 continuation-only 64-token blocks. Active real validation/test sets each have 64 blocks. Smoke D2 is explicitly the first 256 rows of the pinned full D2; full execution uses all 37,946 rows.", "",
              "Imported S0/S1 metrics retain their original full-data measurement scope. They are not directly comparable to the tiny smoke evaluation. New cycles have 4,032 test targets each.", "",
              f"All **{len(validation['checks'])} GPU/provider acceptance checks passed**: exact shared-prefix import, distinct generator/initializer lineage, actual weight changes, both injected worker exits, shard recovery, unchanged completed stage, private backup restore, checkpoint re-evaluation, direct NLL calculation and actual budget-guard shutdown.", "",
              f"Restored final-checkpoint test PPL: **{validation['restored_test']['perplexity']:.8f}**. Direct shifted-NLL maximum discrepancy: **{validation['max_nll_calculation_error']:.1g}**.", "",
              "| New cycle | Real-test PPL, 64 blocks | Initializer | Data generator |", "| --- | ---: | --- | --- |"]
    for stage in stages:
        lines.append(f"| {stage['cycle']} | {stage['test_perplexity']:.8f} | fixed S0 | preceding selected model |")
    lines += ["", "## Execution and recovery", "",
              f"GPU: {r['gpu']}, {r['data_center']}, quoted ${r['quoted_gpu_hourly_usd']:.2f}/hour. Pipeline completion took **{elapsed:.2f} minutes** from provisioning. Observed account debit through cleanup: **${cleanup['actual_account_balance_delta_usd']:.6f}**, including other retained storage and subject to provider billing timing.", "",
              f"Temporary Pod `{r['pod_id']}` and volume `{r['volume_id']}` were deleted after recovery verification. Historical full-run resources were preserved. Private backups remain in the existing model/dataset repositories under this run ID.", "",
              f"Local metadata recovery: {recovery['verified_files']} files / {recovery['verified_bytes']:,} bytes. Bulk objects were restored and checked on the GPU, not downloaded to the laptop. Runtime SHA-256: `{r['code_sha256']}`.", "",
              "## Measured remaining-work estimate", "",
              f"Warmed training: {b['training_target_tokens_per_second']:,.0f} target tokens/s; deterministic five-beam generation: {b['generation_new_tokens_per_second']:,.0f} new tokens/s. Production batch sizes: effective 128, microbatch 32, generation batch 32.", "",
              f"Nine new full training stages and eight full generation stages plus the final probe: **{estimate['estimated_hours']:.2f} hours / ${estimate['estimated_compute_and_running_storage_usd']:.2f}** conservatively, including the forecast's 1.5x compute margin and upload/setup allowances. This is an estimate, not a completed experiment; retained-storage reserves and prior setup spend are separate.", "",
              "[Execution receipt](execution_receipt.json) · [Raw records](records/) · [Generated analysis](figures/report.md)", "",
              "Regenerate analysis with the maintained report command. Export this acceptance from verified local metadata using `python scripts/analysis/export_acceptance.py RUN_ID`; the exporter refuses to overwrite an existing published acceptance."]
    (dest / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name in ("execution_receipt.json", "report.md"):
        path = dest / name
        files[name] = {"sha256": sha256(path), "bytes": path.stat().st_size, "integrity_scope": "authored_publication"}
    atomic_json(dest / "publication_manifest.json", {"schema_version": 1, "run_id": run_id,
                "source_artifact_index_sha256": recovery["artifact_index_sha256"], "files": files,
                "manifest_hash_scope": "Excludes this self-referential publication manifest."})
    return {"report": str(dest / "report.md"), "checks": len(validation["checks"]), "files": len(files)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--superseded", action="store_true")
    parser.add_argument("--offline-tests", type=int)
    args = parser.parse_args()
    print(json.dumps(export(args.run_id, superseded=args.superseded, offline_tests=args.offline_tests), indent=2))
