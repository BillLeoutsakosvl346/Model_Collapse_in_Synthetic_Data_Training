"""Audit and plot the shared-start pair using saved statistics and tiny probes.

No model loading, training, GPU use or cloud mutations. Retrieve metadata and the
nine new first-shard probes before the full audit. --figures-only uses published
small records/metrics and writes to a new output directory.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model_collapse.common import PROJECT, atomic_json, read_json, sha256
from model_collapse.analysis.records import read_metrics_csv
from model_collapse.science.metrics import diversity

RESET = "20261027T211852Z-full-reset-s0"
CONTINUE = "20261026T192740Z-full-continued"
PUBLISHED = PROJECT / "reports" / RESET
COLORS = {"reset_s0": "#755493", "continued": "#cc6b27", "shared": "#666666"}


def csv_write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def records(run_id):
    """Return immutable verified source files; keep live resource controls separate."""
    root = PROJECT / "runs" / run_id
    public = PROJECT / "reports" / run_id
    index_path = root / "artifact_index.json"
    if not index_path.exists():
        index_path = public / "artifact_index.json"
    index = read_json(index_path)
    if not index or index["run_id"] != run_id:
        raise RuntimeError("Retrieve the complete metadata snapshot first")
    redirects = read_json(root / "recovery_snapshot.json", {}).get("preserved_control_files", {})
    used = {}

    def get(rel):
        meta = index["files"][rel]
        candidates = [root / rel, public / "records" / rel]
        if rel in redirects:
            candidates.insert(0, root / redirects[rel]["recovered_path"])
        for p in candidates:
            if p.is_file() and p.stat().st_size == meta["bytes"] and sha256(p) == meta["sha256"]:
                used[rel] = {"sha256": meta["sha256"], "bytes": meta["bytes"]}
                return read_json(p), p
        raise RuntimeError(f"Missing/corrupt statistics: {run_id}/{rel}")

    return root, index, index_path, used, get


def audit(out):
    rows, metrics, histories, inputs, examples, evidence, sources = [], {}, [], {}, {}, {}, {}
    expected_blocks = {"train": 37946, "validation": 3922, "test": 4494}
    parent_index = read_json(PROJECT / "reports" / CONTINUE / "artifact_index.json")
    completed = read_json(PROJECT / "runs" / RESET / "completion_status_verified.json")
    probe_bytes, common_prompts = 0, None
    for arm, run_id in (("continued", CONTINUE), ("reset_s0", RESET)):
        root, index, index_path, used, get = records(run_id)
        sources[arm] = {"run_id": run_id, "index_sha256": sha256(index_path)}
        c, _ = get("config.json")
        env, _ = get("environment.json")
        data, _ = get("data/manifest.json")
        completion, _ = get("completion.json")
        assert completion["complete"] and completion["generations"] == 10
        assert c["initialization"] == arm and c["epochs"] == 5 and c["seed"] == 42
        for split, n in expected_blocks.items():
            assert data["stats"][split]["used_blocks"] == n
            assert index["files"][f"data/{split}.npy"]["sha256"] == parent_index["files"][f"data/{split}.npy"]["sha256"]
        metrics[arm] = []
        examples[arm] = {}
        for g in range(11):
            prefix = f"seed_42/gen_{g:02d}"
            m, metric_file = get(prefix + "/metrics.json")
            training, _ = get(prefix + "/training.json")
            seq, _ = get(prefix + "/test_sequences.json")
            reference, _ = get(prefix + "/reference_sequences.json")
            samples, _ = get(prefix + "/samples.json")
            prompts = [sample["prompt"] for sample in samples]
            if common_prompts is None:
                common_prompts = prompts
            assert prompts == common_prompts, "Fixed-probe prompt order changed"
            assert m["training"] == training
            metrics[arm].append(m)
            examples[arm][str(g)] = samples
            if arm == "reset_s0" and g < 2:
                assert index["files"][prefix + "/metrics.json"] == parent_index["files"][prefix + "/metrics.json"]
                assert m == metrics["continued"][g]
                get(prefix + "/import.json")
            is_new = (arm == "continued" and g >= 1) or (arm == "reset_s0" and g >= 2)
            if is_new:
                parent = metrics[arm][g - 1]
                expected = parent["training"]["selected_weights_sha256"] if arm == "continued" else c["start_checkpoint"]["weights_sha256"]
                assert training["initial_weights_sha256"] == expected
                assert training["input_tokens_sha256"] == parent["generation_timing"]["continuation_tokens_sha256"]
                assert training["selected_weights_sha256"] != expected and training["training_blocks"] == 37946
                assert training["optimizer_initialization"] == "fresh_each_cycle"
                if arm == "reset_s0":
                    assert training["lineage"]["generator_weights_sha256"] == parent["training"]["selected_weights_sha256"]
                    get(prefix + "/stage_completion.json")
                history, _ = get(prefix + "/training_history.json")
                assert len(history) == 5
                assert training["selected_epoch"] == min(history, key=lambda h: h["validation_loss"])["epoch"]
                for h in history:
                    histories.append({"arm": arm, "cycle": g, "stage_origin": "executed", **h})
            if not (arm == "reset_s0" and g < 2):
                generated, _ = get(f"synthetic_for_gen_{g+1:02d}/manifest.json")
                assert generated == m["generation_timing"]
                assert generated["provenance"]["parent_weights_sha256"] == training["selected_weights_sha256"]
                assert generated["provenance"]["prompt_count"] == (37946 if g < 10 else 256)
            probe_run = CONTINUE if arm == "reset_s0" and g < 2 else run_id
            probe = PROJECT / "runs" / probe_run / "probe_tokens" / f"gen_{g:02d}.npy"
            probe_index = parent_index if probe_run == CONTINUE else index
            probe_rel = f"synthetic_for_gen_{g+1:02d}/shard_0000000.npy"
            assert sha256(probe) == probe_index["files"][probe_rel]["sha256"]
            values = np.load(probe, allow_pickle=False)
            assert values.shape == (256, 64)
            measured = diversity(values)
            assert all(math.isclose(measured[k], v, abs_tol=1e-12) for k, v in m["diversity"].items())
            if probe_run == RESET:
                probe_bytes += probe.stat().st_size
            inputs[f"{probe_run}/probe/gen_{g:02d}"] = {"sha256": sha256(probe), "bytes": probe.stat().st_size}
            per_block = np.asarray(seq["sequence_mean_nll"])
            ref = np.asarray(reference["sequence_mean_nll"])
            assert len(per_block) == 4494 and len(ref) == 256 and len(samples) == 256
            assert np.isclose(per_block.mean(), m["test"]["loss"], atol=1e-12, rtol=0)
            assert np.isclose(ref.mean(), m["frozen_g0_reference"]["loss"], atol=1e-12, rtol=0)
            for split, count in (("validation", 3922), ("test", 4494)):
                score = m[split]
                assert score["scored_tokens"] == count * 63
                assert math.isclose(math.exp(score["nll_sum"] / score["scored_tokens"]), score["perplexity"], rel_tol=1e-12)
            rows.append({"arm": arm, "cycle": g, "stage_origin": "shared_reference" if g < 2 else "executed",
                         "original_stage_run_id": CONTINUE if g < 2 else run_id,
                         "validation_loss": m["validation"]["loss"], "validation_perplexity": m["validation"]["perplexity"],
                         "validation_nll_sum": m["validation"]["nll_sum"], "validation_scored_tokens": m["validation"]["scored_tokens"],
                         "test_loss": m["test"]["loss"], "test_perplexity": m["test"]["perplexity"],
                         "test_nll_sum": m["test"]["nll_sum"], "test_scored_tokens": m["test"]["scored_tokens"],
                         "test_block_nll_p50": float(np.percentile(per_block, 50)), "test_block_nll_p95": float(np.percentile(per_block, 95)),
                         "test_passage_ppl_mean": m["test"]["mean_sequence_perplexity"],
                         "test_passage_ppl_sd": m["test"]["std_sequence_perplexity"],
                         "frozen_s0_nll": m["frozen_g0_reference"]["loss"], "selected_epoch": training["selected_epoch"],
                         "recorded_training_seconds": training["seconds"], "recorded_generation_seconds": m["generation_timing"]["seconds"],
                         "new_branch_training_seconds": training["seconds"] if arm == "reset_s0" and g >= 2 else 0,
                         "new_branch_generation_seconds": m["generation_timing"]["seconds"] if arm == "reset_s0" and g >= 2 else 0,
                         "recorded_peak_allocated_gpu_bytes": training.get("peak_allocated_gpu_bytes"), "training_blocks": training["training_blocks"],
                         "initial_weights_sha256": training["initial_weights_sha256"], "selected_weights_sha256": training["selected_weights_sha256"],
                         "training_tokens_sha256": training["input_tokens_sha256"], **measured})
        if arm == "continued":
            original_config, original_environment = c, env
        else:
            for key in ("seed", "model", "model_revision", "dataset", "dataset_revision", "dataset_config", "block_size", "new_tokens", "num_beams",
                        "repetition_penalty", "epochs", "learning_rate", "min_learning_rate", "weight_decay", "microbatch", "effective_batch",
                        "generation_batch", "eval_batch", "precision", "probe_count"):
                assert c[key] == original_config[key], key
            assert env == original_environment
            shared, _ = get("shared_prefix.json")
            assert shared["d2_full_tokens_sha256"] == shared["d2_active_tokens_sha256"]
            assert metrics[arm][2]["training"]["input_tokens_sha256"] == metrics["continued"][2]["training"]["input_tokens_sha256"]
            get("synthetic_for_gen_02/manifest.json")
        inputs.update({f"{run_id}/{rel}": meta for rel, meta in used.items()})
        evidence[arm] = {"statistic_files_verified": len(used), "run_id": run_id}
        if arm == "reset_s0":
            for rel in used:
                _, src = get(rel)
                dest = PUBLISHED / "records" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    assert sha256(dest) == sha256(src), "Never overwrite published evidence"
                else:
                    shutil.copyfile(src, dest)
            target = PUBLISHED / "artifact_index.json"
            if target.exists():
                assert sha256(target) == sha256(index_path)
            else:
                shutil.copyfile(index_path, target)

    by_arm = {arm: [r for r in rows if r["arm"] == arm] for arm in metrics}
    a, b = by_arm["reset_s0"], by_arm["continued"]
    aa, _ = records(RESET)[-1]("seed_42/gen_10/test_sequences.json")
    bb, _ = records(CONTINUE)[-1]("seed_42/gen_10/test_sequences.json")
    block_gain = np.asarray(bb["sequence_mean_nll"]) - np.asarray(aa["sequence_mean_nll"])
    summary = {"test_ppl_shared_s0": a[0]["test_perplexity"], "test_ppl_shared_s1": a[1]["test_perplexity"],
               "first_branch_cycle": 2, "cycle2_ppl_reset": a[2]["test_perplexity"], "cycle2_ppl_continued": b[2]["test_perplexity"],
               "cycle2_ppl_reset_relative_to_continued_percent": 100 * (a[2]["test_perplexity"] / b[2]["test_perplexity"] - 1),
               "final_ppl_reset": a[10]["test_perplexity"], "final_ppl_continued": b[10]["test_perplexity"],
               "final_ppl_reset_relative_to_continued_percent": 100 * (a[10]["test_perplexity"] / b[10]["test_perplexity"] - 1),
               "final_test_nll_continued_minus_reset": float(block_gain.mean()),
               "final_test_blocks_lower_nll_with_reset_percent": float(np.mean(block_gain > 0) * 100),
               "final_test_blocks_lower_nll_with_reset": int(np.sum(block_gain > 0)),
               "final_ppl_reset_change_from_s0_percent": 100 * (a[10]["test_perplexity"] / a[0]["test_perplexity"] - 1),
               "reset_late_ppl_min_cycle3_to10": min(r["test_perplexity"] for r in a[3:]),
               "reset_late_ppl_max_cycle3_to10": max(r["test_perplexity"] for r in a[3:]),
               "reset_distinct2_change_cycle2_to10_percent": 100 * (a[10]["distinct_2"] / a[2]["distinct_2"] - 1),
               "reset_repeated4_change_cycle2_to10_pp": 100 * (a[10]["repeated_4gram_rate"] - a[2]["repeated_4gram_rate"]),
               "reset_entropy_change_cycle2_to10_nats": a[10]["token_entropy_nats"] - a[2]["token_entropy_nats"],
               "duplicate_outputs_final_reset": round(a[10]["duplicate_continuation_rate"] * 256),
               "duplicate_outputs_final_continued": round(b[10]["duplicate_continuation_rate"] * 256),
               "new_training_input_token_exposures": 9 * 5 * 37946 * 64,
               "new_full_generation_tokens": 8 * 37946 * 64,
               "new_final_probe_tokens": 256 * 64,
               "new_training_epochs_executed": 45,
               "newly_selected_epochs_reset": [h["selected_epoch"] for h in completed["stage_histories"]],
               "completion": completed}
    changes = [{"cycle": g, "shared_measurement": g < 2,
                "reset_test_ppl": a[g]["test_perplexity"], "continued_test_ppl": b[g]["test_perplexity"],
                "continued_minus_reset_test_nll": b[g]["test_loss"] - a[g]["test_loss"],
                "reset_minus_continued_distinct2_pp": 100 * (a[g]["distinct_2"] - b[g]["distinct_2"]),
                "reset_minus_continued_repeated4_pp": 100 * (a[g]["repeated_4gram_rate"] - b[g]["repeated_4gram_rate"])} for g in range(11)]
    csv_write(out / "metrics.csv", rows)
    csv_write(out / "cycle_comparison.csv", changes)
    csv_write(out / "training_history.csv", histories)
    atomic_json(out / "generated_examples.json", examples)
    atomic_json(out / "summary.json", summary)
    assert len(histories) == 95
    audit_record = {"scope": "Saved statistics and lineage plus recomputed first-256-output probes; no bulk checkpoint re-evaluation or full synthetic-array download.",
                    "sources": sources, "evidence": evidence, "checks": {"all_statistic_files_checksum_verified": True,
                    "shared_s0_s1_metrics_exactly_identical": True, "cycle2_training_data_exactly_identical": True,
                    "all_new_fixed_s0_and_generator_handoffs": True, "same_real_arrays": True, "same_recorded_environments": True,
                    "all_probe_diversity_statistics_recomputed": True, "all_probe_prompt_text_and_order_identical": True,
                    "all_corpus_ppl_and_test_block_nll_recomputed": True,
                    "all_95_executed_epochs_and_selected_checkpoints_checked": len(histories) == 95},
                    "new_probe_bytes_downloaded": probe_bytes, "input_hashes": inputs,
                    "scope_notes": "Shared measurements are referenced twice for arm-shaped tables, never counted as independent replicates. 50 historical continued epochs plus 45 new reset epochs; no shared stage re-execution."}
    atomic_json(out / "consistency_audit.json", audit_record)
    return rows, histories, summary, audit_record


def plot(out, rows, histories):
    by_arm = {arm: [r for r in rows if r["arm"] == arm] for arm in ("continued", "reset_s0")}
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(3, 2, figsize=(7.1, 7.0), layout="constrained")
    definitions = [("test_perplexity", 1, "A  Fixed real-test perplexity", "Corpus perplexity"),
                   ("distinct_2", 100, "B  Distinct-2", "Unique / total bigrams (%)"),
                   ("repeated_4gram_rate", 100, "C  Repeated 4-grams", "Repeated occurrences (%)"),
                   ("duplicate_continuation_rate", 100, "D  Exact duplicate outputs", "Duplicate continuations (%)"),
                   ("token_entropy_nats", 1, "E  Empirical token entropy", "Nats"),
                   ("frozen_s0_nll", 1, "F  Outputs scored under frozen S0", "Mean continuation NLL")]
    for ax, (key, scale, title, label) in zip(axes.flat, definitions):
        for arm, legend in (("continued", "Continue previous weights (B)"), ("reset_s0", "Reset to real S0 (A)")):
            values = [r[key] * scale for r in by_arm[arm]]
            ax.plot(range(1, 11), values[1:], "o-", color=COLORS[arm], label=legend, ms=3, lw=1.35)
        ax.plot([0, 1], [by_arm["continued"][g][key] * scale for g in (0, 1)], "s-", color=COLORS["shared"], label="Shared S0/S1", ms=3.4, zorder=5)
        ax.axvline(1, color="#999999", linestyle=":", lw=.75)
        ax.set(title=title, ylabel=label, xlabel="Synthetic-training cycle", xticks=[0, 2, 4, 6, 8, 10])
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.savefig(out / "shared_start_metrics.png", dpi=240)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1), layout="constrained")
    for arm, rid, name in (("continued", CONTINUE, "Continued B10"), ("reset_s0", RESET, "Reset A10")):
        p = PROJECT / "reports" / rid / "records/seed_42/gen_10/test_sequences.json"
        values = np.sort(read_json(p)["sequence_mean_nll"])
        axes[0].plot(values, np.arange(1, len(values) + 1) / len(values) * 100, color=COLORS[arm], label=name)
        for g, style in ((2, "--"), (10, "-")):
            h = [r for r in histories if r["arm"] == arm and r["cycle"] == g]
            if h:
                axes[1].plot([r["epoch"] for r in h], [r["validation_loss"] for r in h], style, marker="o", ms=3,
                             color=COLORS[arm], label=("B" if arm == "continued" else "A") + str(g))
                best = min(h, key=lambda r: r["validation_loss"])
                axes[1].scatter(best["epoch"], best["validation_loss"], marker="*", s=65, color=COLORS[arm], zorder=5)
    axes[0].set(title="A  Final real-test block losses", xlabel="Mean next-token NLL per 64-token block", ylabel="Cumulative test blocks (%)", xlim=(1, 7))
    axes[0].legend(fontsize=8)
    axes[1].set(title="B  Within-cycle checkpoint selection", xlabel="Epoch", ylabel="Fixed real-validation NLL", xticks=range(1, 6))
    axes[1].legend(fontsize=8, ncol=2)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(out / "generalization_and_selection.png", dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    out = args.out or PUBLISHED / "generated" / ("shared-start-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Choose a new output directory; published or prior analysis must not be overwritten")
    out.mkdir(parents=True, exist_ok=True)
    if args.figures_only:
        def load_csv(name):
            return read_metrics_csv(PUBLISHED / "analysis" / name)
        rows, histories = load_csv("metrics.csv"), load_csv("training_history.csv")
        audit_record = {"scope": "Figures only from committed statistics; no repeat probe or bulk verification."}
    else:
        rows, histories, _, audit_record = audit(out)
    plot(out, rows, histories)
    outputs = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "analysis_script": str(Path(__file__).resolve().relative_to(PROJECT)),
                "analysis_script_sha256": sha256(__file__), "analysis_base_git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
                "environment": {"python": platform.python_version(), **{k: importlib.metadata.version(k) for k in ("numpy", "matplotlib")}},
                "sources": audit_record.get("sources"), "input_hashes": audit_record.get("input_hashes"),
                "output_files": outputs, "scope": audit_record["scope"], "hash_scope": "Output hashes exclude this self-referential analysis manifest."}
    atomic_json(out / "analysis_manifest.json", manifest)
    print(json.dumps({"output": str(out), "checks": audit_record.get("checks"), "rows": len(rows), "histories": len(histories)}, indent=2))


if __name__ == "__main__":
    main()
