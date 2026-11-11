"""Offline plots and tables for the matched FineWeb completion comparison."""
from __future__ import annotations

import csv
import importlib.metadata
import platform
from pathlib import Path

from ..common import atomic_json, read_json, sha256, digest
from .reporting import _output_directory
from ..workflow.completion import arm_specs, child_roots
from ..mixing import mixture_counts


def _collect(root):
    c = read_json(root / "config.json")
    rows, inputs, examples = [], {}, {}
    definitions = {item["id"]: item for item in arm_specs(c)}
    for arm, child in child_roots(root, c).items():
        for cycle in range(c["generations"] + 1):
            stage = child / f"seed_42/gen_{cycle:02d}"
            path = stage / "metrics.json"
            m = read_json(path)
            if m is None:
                continue
            inputs[f"{child.name}/seed_42/gen_{cycle:02d}/metrics.json"] = sha256(path)
            t = m.get("training", {})
            fraction = definitions[arm]["synthetic_fraction"]
            real_pairs, synthetic_pairs = mixture_counts(c["train_pairs_per_cycle"], fraction) if cycle else (0, 0)
            mixture = m.get("mixture", {})
            row = {"arm": arm, "cycle": cycle, "comparison_id": root.name,
                   "training_decoder": definitions[arm].get("decoder", "beam" if fraction else "real_targets"),
                   "synthetic_fraction": fraction,
                   "real_training_pairs": mixture.get("real_pairs", real_pairs),
                   "synthetic_training_pairs": mixture.get("synthetic_pairs", synthetic_pairs),
                   "mixture_manifest_sha256": mixture.get("mixture_manifest_sha256"), "weights_sha256": m["weights_sha256"],
                   "validation_loss": m["validation"]["loss"], "validation_perplexity": m["validation"]["perplexity"],
                   "test_loss": m["test"]["loss"], "test_perplexity": m["test"]["perplexity"],
                   "test_targets": m["test"]["scored_tokens"], "test_nll_p95": m["test"]["p95_example_nll"],
                   "test_nll_median": m["test"]["median_example_nll"], "training_seconds": t.get("seconds", 0),
                   "global_step": t.get("global_step", 0), "selected_epoch": t.get("selected_epoch"),
                   "training_loss": m.get("training_evaluation", {}).get("loss"),
                   "peak_allocated_gpu_bytes": t.get("peak_allocated_gpu_bytes", 0), **m["probe"],
                   **{f"secondary_{name}_perplexity": v["perplexity"] for name, v in m["secondary"].items()}}
            history = read_json(stage / "training_history.json", [])
            row["learning_rate_end"] = history[-1]["learning_rate_end"] if history else c["learning_rate"]
            row["optimization_loss"] = history[-1]["train_loss"] if history else None
            if c.get("training_chunk_policy") == "repeat_fixed":
                anchor = m["fixed_real_anchor"]
                row.update(training_chunk_policy="repeat_fixed", source_collection="cycle_01",
                           fixed_real_anchor_loss=anchor["loss"],
                           fixed_real_anchor_perplexity=anchor["perplexity"],
                           real_anchor_generalization_gap=m["test"]["loss"] - anchor["loss"])
            rows.append(row)
            example_path = stage / "probe_examples.json" if cycle else root / "seed_42/gen_00/probe_examples.json"
            if example_path.exists():
                inputs[root.name + ":" + arm + ":" + str(cycle) + ":examples"] = sha256(example_path)
                examples[f"{arm}_{cycle}"] = read_json(example_path)
    return rows, inputs, examples


def comparison_contract(root):
    root = Path(root)
    c = read_json(root / "config.json")
    keys = ("model", "model_revision", "dataset", "dataset_revision", "dataset_config", "seed",
            "learning_rate", "min_learning_rate", "weight_decay", "precision", "epochs", "checkpoint_selection",
            "initialization", "optimizer_policy", "max_sequence_length", "microbatch", "effective_batch",
            "generation_batch", "block_size", "new_tokens", "num_beams", "repetition_penalty", "probe_count",
            "secondary_corpora", "corpus_bank", "generations", "train_pairs_per_cycle", "evaluation_pairs",
            "required_gpu", "required_image", "required_environment")
    baseline = read_json(root / "seed_42/gen_00/metrics.json")
    return {"settings": {k: c.get(k) for k in keys}, "bank_manifest_sha256": c["prepared_bank"]["manifest_sha256"],
            "baseline_weights_sha256": baseline["weights_sha256"],
            "primary_scored_tokens": {name: baseline[name]["scored_tokens"] for name in ("validation", "test")},
            "objective": "bos_prompt64_continuation64_score_positions65_to128"}


def fraction_label(fraction):
    # Preserve decimal percentages: 99.9% must not be labelled as 100%.
    return f"{100 * fraction:.12g}% synthetic"


def report(root, out=None, *, for_backup=False, reference=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    c = read_json(root / "config.json")
    contract = comparison_contract(root)
    rows, inputs, examples = _collect(root)
    reference_contract = None
    execution_difference = None
    if reference is not None:
        reference = Path(reference)
        if not read_json(reference / "completion.json", {}).get("complete"):
            raise RuntimeError("Reference comparison must be complete before combining trajectories")
        reference_contract = comparison_contract(reference)
        comparable_reference = {**reference_contract, "settings": dict(reference_contract["settings"])}
        declared = c.get("reference_execution_difference")
        if declared == {"run_id": reference.name, "field": "required_gpu",
                        "old": reference_contract["settings"]["required_gpu"], "new": contract["settings"]["required_gpu"]}:
            execution_difference = declared
            comparable_reference["settings"]["required_gpu"] = declared["new"]
        if comparable_reference != contract:
            raise RuntimeError("Reference bank, evaluation, initialization or scientific settings differ")
        extra_rows, extra_inputs, extra_examples = _collect(reference)
        reference_config = read_json(reference / "config.json")
        expected = len(arm_specs(reference_config)) * (reference_config["generations"] + 1)
        if len(extra_rows) != expected:
            raise RuntimeError("Reference comparison has missing metric records")
        rows += extra_rows
        inputs.update(extra_inputs)
        examples.update({reference.name + ":" + k: v for k, v in extra_examples.items()})
    destination = _output_directory(root, out, root.name, for_backup=for_backup)
    destination.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No completed completion metrics available")
    atomic_json(destination / "metrics.json", rows)
    atomic_json(destination / "examples.json", examples)
    with (destination / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = {"test_perplexity": "Real continuation perplexity", "test_nll_p95": "P95 real-example NLL",
               "distinct_2": "Pooled distinct-2", "token_entropy_nats": "Pooled token entropy (nats)",
               "mean_within_entropy_nats": "Mean within-output entropy (nats)",
               "mean_within_repeated_4gram_rate": "Mean within-output repeated 4-grams",
               "repeated_4gram_rate": "Pooled repeated 4-grams", "duplicate_continuation_rate": "Duplicate continuations",
               "learning_rate_end": "Learning rate", "training_seconds": "Training and persistence time (seconds)"}
    metrics.update({f"secondary_{name}_perplexity": name.replace("_", " ") + " full-text perplexity" for name in c["secondary_corpora"]})
    if c.get("training_chunk_policy") == "repeat_fixed":
        metrics.update(fixed_real_anchor_loss="Fixed real first-chunk NLL",
                       fixed_real_anchor_perplexity="Fixed real first-chunk perplexity",
                       real_anchor_generalization_gap="Test minus fixed real first-chunk NLL")
    for metric, label in metrics.items():
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_box_aspect(1)
        groups = sorted({(row["synthetic_fraction"], row.get("training_chunk_policy", "fresh_per_cycle"), row["training_decoder"]) for row in rows})
        for fraction, policy, decoder in groups:
            selected = sorted((row for row in rows if row["synthetic_fraction"] == fraction
                and row.get("training_chunk_policy", "fresh_per_cycle") == policy and row["training_decoder"] == decoder
                and metric in row), key=lambda row: row["cycle"])
            if not selected:
                continue
            label_text = fraction_label(fraction)
            if c.get("decoding_arms") is not None and fraction:
                label_text += " / " + decoder
            if c.get("training_chunk_policy") == "repeat_fixed":
                label_text += " / " + ("repeated prompts" if policy == "repeat_fixed" else "fresh prompts")
            ax.plot([row["cycle"] for row in selected], [row[metric] for row in selected], "o-", label=label_text)
        ax.set(xlabel="Training cycle", ylabel=label, title=label)
        ax.grid(alpha=.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(destination / (metric + ".png"), dpi=180)
        plt.close(fig)
    text = ("# Matched FineWeb completion comparison\n\n"
            "All conditions begin at original pretrained OPT. Each cycle supplies the same real 64-token prompts. "
            "Real targets are the actual next 64 source tokens. Each condition generates its synthetic targets using its own preceding checkpoint. "
            "Mixtures replace selected continuations at their original prompt rows and retain the remaining real targets. "
            "BOS and prompt positions remain visible, while the continuation alone contributes to training loss.\n\n"
            "Primary perplexity scores fixed real continuations. Secondary full-text perplexities use the frozen 512-token source-bank arrays. "
            "Output statistics use a fixed validation probe. The shared cycle-zero point is reused evidence.\n\n"
            "| Arm | Cycle | Validation PPL | Test PPL | Test targets |\n|---|---:|---:|---:|---:|\n")
    for row in rows:
        text += f"| {row['arm']} | {row['cycle']} | {row['validation_perplexity']:.5f} | {row['test_perplexity']:.5f} | {row['test_targets']} |\n"
    text += "\n## Figures\n\n"
    for name, label in metrics.items():
        text += f"![{label}]({name}.png)\n\n"
    text += "One seed gives descriptive trajectories. Pooled and within-output statistics have separate denominators. Exact training and evaluation counts are recorded in the saved manifests.\n"
    if c.get("training_chunk_policy") == "repeat_fixed":
        text += ("\nBoth new arms reuse bank cycle_01. Real continuations remain fixed; synthetic continuations are regenerated "
                 "from that arm's preceding checkpoint every epoch. The fixed real anchor is training data, not held-out data. "
                 "Optimizer state and the historical global cosine learning-rate schedule continue across epochs.\n")
    if c.get("decoding_arms") is not None:
        text += ("\nOnly training-target decoding changes: greedy chooses one highest-probability next token; "
                 "pure sampling draws at temperature 1 with top-k disabled and top-p 1. "
                 "All output-statistic probes retain the historical deterministic five-beam decoder. "
                 "The cosine learning-rate schedule remains 5e-6 to 1e-6.\n")
    if execution_difference:
        text += ("\nExecution difference: historical control used " + execution_difference["old"] + "; new arms used " +
                 execution_difference["new"] + ". All other comparison-contract fields match. Hardware-dependent "
                 "floating-point differences may affect individual outputs; this is not a hardware-controlled comparison.\n")
    (destination / "report.md").write_text(text, encoding="utf-8")
    atomic_json(destination / "analysis_manifest.json", {"inputs": inputs, "configuration_sha256": sha256(root / "config.json"),
                "script_sha256": sha256(__file__), "runtime_sha256": read_json(root / "manifest.json", {}).get("code_sha256"),
                "python": platform.python_version(), "matplotlib": importlib.metadata.version("matplotlib"),
                "training_environment": read_json(root / "environment.json", {}),
                "bank_manifest_sha256": c["prepared_bank"]["manifest_sha256"],
                "comparison_contract_sha256": digest(contract), "reference_contract": reference_contract,
                "reference_execution_difference": execution_difference,
                "outputs": {p.name: sha256(p) for p in destination.iterdir() if p.is_file()}})
    return {"out": str(destination), "rows": len(rows)}
