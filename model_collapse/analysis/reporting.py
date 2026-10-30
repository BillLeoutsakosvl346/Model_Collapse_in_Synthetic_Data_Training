"""Regenerate analysis from immutable records without cloud or model imports."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..common import PROJECT, atomic_json, digest, sha256
from .forecasts import forecast
from .records import METRIC_DEFINITIONS


class Evidence:
    def __init__(self, root):
        self.root = root
        self.files = {}

    def read(self, path, default=None):
        path = Path(path)
        if not path.is_file():
            return default
        data = path.read_bytes()
        self.files[path.relative_to(self.root).as_posix()] = {
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        return json.loads(data)


def _output_directory(source, out, run_id, *, for_backup=False):
    published = (PROJECT / "reports").resolve()
    if for_backup and out is not None:
        raise ValueError("Backup analysis chooses its own dedicated generated directory; do not pass out")
    if out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        out = (source / "generated" / stamp) if for_backup else (published / run_id / "generated" / stamp)
    out = Path(out).resolve()
    if out.is_relative_to(published):
        parts = out.relative_to(published).parts
        if len(parts) < 3 or parts[1] != "generated":
            raise ValueError("Refusing to overwrite published evidence or authored reports; use reports/<run>/generated/<analysis> or an empty external directory")
    elif (out == source or out.is_relative_to(source)) and not (for_backup and out.parent == source / "generated"):
        raise ValueError("Analysis output must be separate from source evidence")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Analysis output directory must be new or empty")
    return out


def _rows(metrics, seed_root, evidence, c):
    import numpy as np
    rows, imports, reference_sequences = [], {}, {}
    for m in metrics:
        g = m["generation"]
        stage = seed_root / f"gen_{g:02d}"
        receipt = evidence.read(stage / "import.json")
        imported = bool(receipt)
        if imported:
            imports[str(g)] = receipt
        t = m["training"]
        old_train = t.get("seconds", 0.0)
        old_gen = m.get("generation_timing", {}).get("seconds", 0.0)
        row = {"generation": g,
               "validation_loss": m["validation"]["loss"], "validation_perplexity": m["validation"]["perplexity"],
               "test_loss": m["test"]["loss"], "test_perplexity": m["test"]["perplexity"],
               "test_sequence_perplexity_mean": m["test"].get("mean_sequence_perplexity"),
               "test_sequence_perplexity_sd": m["test"].get("std_sequence_perplexity"),
               "frozen_s0_nll": m.get("frozen_g0_reference", {}).get("loss"),
               "selected_epoch": t.get("selected_epoch"), "train_seconds": old_train,
               "checkpoint_selection": t.get("checkpoint_selection", "lowest_fixed_real_validation_loss"),
               "executed_epochs": t.get("executed_epochs"),
               "generation_seconds": old_gen, "new_train_seconds": 0.0 if imported else old_train,
               "new_generation_seconds": 0.0 if imported else old_gen,
               "stage_origin": "imported_shared_stage" if imported else "current_run",
               "source_run_id": receipt.get("source_run_id", "") if imported else "",
               "source_stage": receipt.get("source_stage_index", g) if imported else "",
               "training_blocks": t.get("training_blocks", 0),
               "training_tokens_per_epoch": t.get("training_blocks", 0) * c.get("block_size", 64),
               "initial_weights_sha256": t.get("initial_weights_sha256"),
               "selected_weights_sha256": t.get("selected_weights_sha256"),
               "input_tokens_sha256": t.get("input_tokens_sha256"),
               "peak_allocated_gpu_bytes": t.get("peak_allocated_gpu_bytes", 0),
               **m["diversity"]}
        sequences = evidence.read(stage / "test_sequences.json")
        if sequences and sequences.get("sequence_mean_nll"):
            values = sequences["sequence_mean_nll"]
            row.update(test_block_nll_p50=float(np.percentile(values, 50)),
                       test_block_nll_p95=float(np.percentile(values, 95)))
        values = evidence.read(stage / "reference_sequences.json")
        if values and values.get("sequence_mean_nll"):
            reference_sequences[g] = values["sequence_mean_nll"]
        rows.append(row)
    return rows, imports, reference_sequences


def _plots(rows, refs, out, label, policy):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    x = [r["generation"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for metric in ("validation", "test"):
        for origin, style in (("current_run", "o-"), ("imported_shared_stage", "s--")):
            subset = [r for r in rows if r["stage_origin"] == origin]
            if not subset:
                continue
            suffix = " (imported source evaluation)" if origin.startswith("imported") else ""
            for ax, quantity in zip(axes, ("loss", "perplexity")):
                ax.plot([r["generation"] for r in subset], [r[f"{metric}_{quantity}"] for r in subset], style, label=metric + suffix)
    for ax, ylabel in zip(axes, ("Token-weighted mean NLL", "Corpus perplexity")):
        ax.set(xlabel="Synthetic-training cycle (0 = real-trained reference)", ylabel=ylabel, xticks=x)
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle(f"{label} — held-out real WikiText-2")
    fig.savefig(out / "perplexity.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for n in (1, 2, 3):
        axes[0].plot(x, [r[f"distinct_{n}"] for r in rows], "o-", label=f"distinct-{n}")
    for key, text in (("repeated_4gram_rate", "repeated 4-grams"), ("duplicate_continuation_rate", "duplicate continuations")):
        axes[1].plot(x, [r[key] for r in rows], "o-", label=text)
    for ax in axes:
        ax.set(xlabel="Synthetic-training cycle", ylabel="Fraction", xticks=x)
        ax.legend()
        ax.grid(alpha=.2)
    counts = sorted({r.get("probe_count", 256) for r in rows})
    fig.suptitle(f"Fixed prompt probe; counts {counts}; continuation tokens only")
    fig.savefig(out / "diversity.png", dpi=180)
    plt.close(fig)
    if refs:
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        for g, values in refs.items():
            ax.hist(values, bins=30, histtype="step", density=True, label=f"S{g}")
        ax.set(xlabel="Continuation mean NLL under frozen real G0/S0 (63 shifted targets)", ylabel="Density")
        ax.legend(fontsize=8, ncols=2)
        fig.savefig(out / "reference_scores.png", dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.bar(np.asarray(x) - .18, [r["new_train_seconds"] / 60 for r in rows], width=.36, label="new training/checkpoint")
    ax.bar(np.asarray(x) + .18, [r["new_generation_seconds"] / 60 for r in rows], width=.36, label="new generation")
    ax.set(xlabel="Synthetic-training cycle", ylabel="New-work minutes (imported work excluded)", xticks=x)
    ax.legend()
    fig.savefig(out / "runtime.png", dpi=180)
    plt.close(fig)


def _analysis_identity():
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True, text=True, check=False).stdout.strip() or None
    except OSError:
        revision = None  # Source bundles do not require Git to be installed.
    source = {p.relative_to(PROJECT).as_posix(): sha256(p) for p in sorted((PROJECT / "model_collapse" / "analysis").glob("*.py"))}
    source["model_collapse/report.py"] = sha256(PROJECT / "model_collapse" / "report.py")
    packages = {}
    for name in ("numpy", "matplotlib"):
        packages[name] = importlib.metadata.version(name)
    return {"git_revision": revision, "source_files": source, "source_sha256": digest(source),
            "environment": {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}}


def report(root, out=None, *, for_backup=False):
    """Write a fresh analysis directory; never replace curated or raw evidence.

    ``root`` may be a recovered run directory or a published report directory
    containing ``records/``. Missing benchmarks, baseline or sequence arrays do
    not prevent plotting the available saved metrics. No remote access occurs.
    A remote worker uses ``for_backup=True`` to place fresh analysis under the
    run's dedicated generated/ directory, which its artifact backup includes.
    """
    root = Path(root).resolve()
    source = root / "records" if (root / "records" / "config.json").is_file() else root
    run_id = root.parent.name if root.name == "records" else root.name
    evidence = Evidence(source)
    c = evidence.read(source / "config.json")
    if not c:
        raise RuntimeError("No saved config.json; fetch metadata first")
    seed_root = source / f"seed_{c['seed']}"
    metrics = [evidence.read(p) for p in sorted(seed_root.glob("gen_*/metrics.json"))]
    if not metrics:
        raise RuntimeError("No completed generation metrics to report")
    rows, imports, refs = _rows(metrics, seed_root, evidence, c)
    policy = c.get("initialization", "fresh_base")
    labels = {"fresh_base": "original pretrained reset", "continued": "continued selected weights",
              "reset_s0": "reset to fixed real-data-trained S0 after shared S1"}
    label = labels.get(policy, policy)
    out = _output_directory(source, out, run_id, for_backup=for_backup)
    out.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _plots(rows, refs, out, label, policy)
    baseline = evidence.read(seed_root / "pretrained" / "metrics.json")
    resources = evidence.read(source / "resources.json", {})
    manifest = evidence.read(source / "manifest.json", {})
    evidence.read(source / "environment.json")
    evidence.read(source / "shared_prefix.json")
    bench = evidence.read(source / "benchmark.json")
    benchmark_resources = evidence.read(source / "benchmark_resources.json", resources)
    forecasts = []
    if bench and benchmark_resources:
        rate = benchmark_resources.get("actual_pod_hourly_usd", benchmark_resources.get("quoted_gpu_hourly_usd"))
        if rate is not None:
            hourly = rate + 30 * .07 / 730 + 20 * .1 / 730
            forecasts = [forecast(bench, g, hourly, initialization=policy) for g in (5, 10)]
            atomic_json(out / "forecast.json", {"benchmark_gpu": bench["gpu"], "hourly_usd": hourly, "forecasts": forecasts,
                                               "storage_assumption": "30 GB network + 20 GB container, historical rates; estimates, not billing"})
    lines = [f"# Generated {c['kind']} analysis: {label}", "", f"Experiment: `{run_id}`. Seed: {c['seed']}.", "",
             "This is regenerated analysis of saved evidence, separate from the authored report. No model training, generation, re-evaluation or cloud reads were performed.", "",
             "Pipeline validation only; a smoke experiment is not evidence of collapse." if c["kind"] == "smoke" else "One seed; descriptive results do not establish statistical significance.", ""]
    if baseline:
        lines.extend([f"Untouched pretrained real-test corpus perplexity: {baseline['test']['perplexity']:.4f}.", ""])
    if imports:
        lines.extend(["Imported shared-stage measurements retain their original source split and execution identity. Their recorded runtime remains in metrics.csv, but new-work runtime excludes it. Imported points use a separate plot series; smoke-run active splits may differ from the imported full-data splits.", ""])
    lines.extend(["| Cycle | Origin | Validation PPL | Test PPL | Distinct-2 | Repeated 4-grams |", "|---|---|---:|---:|---:|---:|"])
    for row in rows:
        lines.append(f"| {row['generation']} | {row['stage_origin']} | {row['validation_perplexity']:.4f} | {row['test_perplexity']:.4f} | {row['distinct_2']:.4f} | {row['repeated_4gram_rate']:.4f} |")
    descriptions = {
        "fresh_base": "Every training generation starts from the pinned original pretrained OPT weights; generation 0 trains on real WikiText-2.",
        "continued": "Cycle 0 loads the saved real-data-trained G0/S0. Each next cycle starts from the preceding selected weights. Optimizer and scheduler restart.",
        "reset_s0": "S0 and S1 are imported from the shared continued-run prefix. First new training is cycle 2 on the exact D2 generated by S1. Every new cycle initializes from the same fixed real-data-trained S0; later data comes from this reset branch's preceding selected model."}
    selection = ("Epoch 5 is retained regardless of validation loss; validation is diagnostic and test data only reports results."
                 if c.get("checkpoint_selection") == "final_epoch" else
                 "Validation selects the best newly trained epoch; test data only reports results.")
    lines.extend(["", descriptions.get(policy, "See resolved configuration for initialization policy."), "",
                  "Fixed real prompts generate continuations; only continuation tokens train subsequent cycles. " + selection, "",
                  "## Metric definitions", ""])
    lines.extend(f"- **{key}:** {value}" for key, value in METRIC_DEFINITIONS.items())
    if forecasts:
        lines.extend(["", "## Optional benchmark forecasts", "", "Conservative throughput estimates, not completed runs or itemized invoices. Setup and retained-storage reserves are additional. Imported shared work is excluded.", ""])
        lines.extend(f"- Through cycle {f['synthetic_generations']}: {f['estimated_hours']:.2f} hours; ${f['estimated_compute_and_running_storage_usd']:.2f}." for f in forecasts)
    else:
        lines.extend(["", "No forecast was produced: a benchmark and resource rate were not both available."])
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(out / "metric_definitions.json", METRIC_DEFINITIONS)
    outputs = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()}
    atomic_json(out / "analysis_manifest.json", {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
        "initialization": policy, "scientific_config_sha256": digest(c),
        "source_runtime_identity": resources.get("code_sha256", manifest.get("code_sha256")),
        "source_root": str(source), "inputs": evidence.files, "analysis": _analysis_identity(), "outputs": outputs,
        "imported_stage_receipts": imports, "scope": "Offline saved measurements; no independent model re-evaluation; original evidence untouched.",
        "manifest_hash_scope": "Output hashes exclude this self-referential manifest."})
    return {"report": str(out / "report.md"), "completed_generations": len(metrics),
            "analysis_manifest": str(out / "analysis_manifest.json")}
