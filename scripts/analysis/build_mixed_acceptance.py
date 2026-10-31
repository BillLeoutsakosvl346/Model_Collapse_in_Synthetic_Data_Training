"""Rebuild the mixed-data acceptance report offline from its frozen publication.

Reads the dated receipt rather than today's account/ledger or live provider state.
Does not train, download a tokenizer, reconstruct absent bulk arrays or edit a Doc.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from model_collapse.common import PROJECT, read_json
from model_collapse.report import report

try:
    from .publication import new_destination, verified_publication, write_provenance
except ImportError:  # Direct script invocation.
    from publication import new_destination, verified_publication, write_provenance

DEFAULT_SOURCE = PROJECT / "reports/20261031T153536Z-smoke-continued"


def render(run_id, receipt, metrics, counts):
    checks = len(receipt["checks"])
    r = receipt["metadata_recovery"]
    bulk = receipt["remote_bulk_recovery"]
    estimate = receipt["full_forecast"]
    lines = ["# Mixed-data continued training: GPU acceptance", "", f"Experiment: `{run_id}`.", "",
        "**Pipeline validation only. This regenerated report describes the acceptance-date evidence, not current cloud state or a later full run. It is not evidence of collapse prevention.**", "",
        "## Method and verification", "",
        f"- {receipt['offline_tests_passed']} offline tests passed before acceptance; the dated receipt separately records any later test-only regression.",
        f"- All {checks} recorded GPU/provider checks passed, including both model/data handoffs, injected worker exits, shard recovery, detached progress, private restore/re-evaluation and watchdog shutdown.",
        f"- {receipt['new_training_stages']} new stages executed {receipt['executed_epochs']} epochs in total, retaining epoch {receipt['selected_epoch_each_stage']} each time. Optimizer/scheduler restart each stage; selected weights continue.",
        "- Exact historical S0/D1 were imported. No all-synthetic-trained S1 was reused. Real validation/test blocks remained fixed."]
    for x in counts:
        lines.append(f"- Cycle {x['generation']}: {x['real_blocks']} real + {x['synthetic_blocks']} synthetic 64-token blocks; {x['real_tokens']:,} and {x['synthetic_tokens']:,} tokens respectively.")
    lines += ["", "## Smoke measurements", "", "| Cycle | Selected epoch | Validation NLL | Validation PPL | Test NLL | Test PPL |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for m in metrics:
        lines.append(f"| {m['generation']} | {m['training'].get('selected_epoch') or 'imported S0'} | {m['validation']['loss']:.6f} | {m['validation']['perplexity']:.4f} | {m['test']['loss']:.6f} | {m['test']['perplexity']:.4f} |")
    lines += ["", "These are two tiny training cycles with 64 real test blocks; full-data trajectories are a separate experiment.", "",
              "![Held-out smoke evaluation](analysis/perplexity.png)", "",
              "![Generated-text diversity](analysis/diversity.png)", "",
              "![Within-output and newline diagnostics](analysis/probe_diagnostics.png)", "",
              "## Recorded recovery and execution", "",
              f"- Pipeline completion: {receipt['elapsed_minutes_to_completion']:.2f} minutes from provisioning. Provider shutdown: {receipt['elapsed_minutes_to_provider_shutdown']:.2f} minutes. Temporary resource deletion was recorded after verification.",
              f"- Historical remote restore checked {bulk['verified_files']} files / {bulk['verified_bytes']:,} bytes, including checkpoints. Restored final-checkpoint loss difference: {receipt['restored_checkpoint_test_loss_difference']:.1g}; direct shifted-NLL error: {receipt['max_direct_shifted_nll_error']:.1g}.",
              f"- Local metadata recovery: {r['verified_files']} files / {r['verified_bytes']:,} bytes. The dated receipt also records small-array recovery and independent mixture reconstruction; this report builder does not repeat absent bulk recovery or model evaluation.",
              "- [Saved dataset examples](dataset_preview.json) retain token IDs, source labels and source indices. The original publication records the independent reconstruction performed during acceptance.", "",
              "## Acceptance-date costs and forecast", "",
              f"- Observed account debit: ${receipt['observed_account_balance_delta_usd']:.6f}, including concurrent storage/billing timing; not an itemized compute invoice.",
              f"- Conservative acceptance campaign estimate: ${receipt['conservative_acceptance_campaign_estimate_usd']:.4f} under the recorded ${receipt['acceptance_ceiling_usd']:.2f} allowance.",
              f"- Ten-cycle forecast: {estimate['estimated_hours']:.2f} hours / ${estimate['estimated_compute_and_running_storage_usd']:.2f}, including conservative margins. This is a forecast, not full-run timing.",
              f"- The acceptance-date cumulative ceiling was ${receipt['budget_ceiling_usd']:.0f}; remaining allowance after reserve was ${receipt['remaining_after_2_usd_reserve']:.4f}. Full-run affordability then: {receipt['full_run_affordable_under_current_ceiling']}. Later authorization/launch observations are separate receipts.", "",
              "## Reproduction scope", "",
              "This offline reconstruction verifies the original publication's checksums, saved mixture/selection consistency and recorded finite NLL/perplexity arithmetic, then regenerates tables/figures. It makes no cloud requests and preserves the original publication. It does not certify a new runtime or claim a new GPU test.", "",
              f"Recorded runtime: `{receipt['runtime_sha256']}`. Protocol: `{receipt['protocol_fingerprint']}`.", "",
              "Input/output hashes, script identity, revision and environment are in `publication_reproduction.json`. Historical finance/provider facts come from the frozen validation receipt rather than today's private state."]
    return "\n".join(lines) + "\n"


def build(source, out):
    source = Path(source).resolve()
    out = new_destination(source, out)
    manifest = verified_publication(source)
    receipt = read_json(source / "validation_summary.json")
    c = read_json(source / "records/config.json")
    if c.get("kind") != "smoke" or c.get("synthetic_fraction") != .5 or c.get("checkpoint_selection") != "final_epoch" or c.get("epochs") != 5:
        raise ValueError("Expected the accepted 50/50 five-epoch smoke condition")
    if not all(receipt["checks"].values()) or receipt["runtime_sha256"] != manifest["runtime_sha256"]:
        raise RuntimeError("Successful source-bound acceptance is required")
    metrics = [read_json(p) for p in sorted((source / f"records/seed_{c['seed']}").glob("gen_*/metrics.json"))]
    if len(metrics) != c["generations"] + 1:
        raise RuntimeError("Acceptance stage evidence is incomplete")
    counts = []
    for m in metrics:
        for split in ("validation", "test"):
            loss, ppl = m[split]["loss"], m[split]["perplexity"]
            if not math.isfinite(loss) or not math.isclose(math.exp(loss), ppl, rel_tol=1e-10):
                raise RuntimeError("Recorded corpus NLL/perplexity arithmetic differs")
        g = m["generation"]
        if not g:
            continue
        mixture = read_json(source / f"records/mixed_for_gen_{g:02d}/manifest.json")
        selection = read_json(source / f"records/mixed_for_gen_{g:02d}/selection.json")
        if mixture["training_tokens_sha256"] != m["training"]["input_tokens_sha256"]:
            raise RuntimeError("Mixture/training token identity differs")
        n = mixture["training_blocks"]
        if mixture["real_blocks"] != mixture["synthetic_blocks"] or n != c["train_blocks"] or sorted(selection["permutation"]) != list(range(n)):
            raise RuntimeError("Mixture counts or shuffle differ")
        for key in ("real_indices", "synthetic_indices"):
            if len(set(selection[key])) != n // 2 or len(selection[key]) != n // 2:
                raise RuntimeError("Sampling without replacement differs")
        labels = [int(i >= n // 2) for i in selection["permutation"]]
        if labels != selection["source_labels"] or m["training"]["selected_epoch"] != 5:
            raise RuntimeError("Mixture alignment or final checkpoint policy differs")
        counts.append(mixture)
    previews = read_json(source / "dataset_preview.json")
    if any(x["token_count"] != 64 or len(x["token_ids"]) != 64 for x in previews):
        raise RuntimeError("Saved preview token length differs")
    out.mkdir(parents=True, exist_ok=True)
    report(source, out / "analysis")
    shutil.copyfile(source / "dataset_preview.json", out / "dataset_preview.json")
    (out / "report.md").write_text(render(manifest["run_id"], receipt, metrics, counts), encoding="utf-8")
    inputs = [source / "publication_manifest.json"] + [source / p for p in manifest["files"]]
    write_provenance(source, out, __file__, inputs,
                     scope="Offline mixed acceptance reconstruction from preserved publication; no live state, training, model evaluation or bulk recovery.")
    return {"report": str(out / "report.md"), "recorded_gpu_checks": len(receipt["checks"]), "mixtures_checked": len(counts)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, required=True, help="New empty destination; published evidence is protected")
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.out), indent=2))
