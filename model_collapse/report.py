from __future__ import annotations

import csv
import math
from pathlib import Path

from .common import read_json


def report(root, out=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    root = Path(root)
    out = Path(out) if out else root / "report"
    out.mkdir(parents=True, exist_ok=True)
    c = read_json(root / "config.json")
    seed_root = root / f"seed_{c['seed']}"
    metrics = [read_json(p) for p in sorted(seed_root.glob("gen_*/metrics.json"))]
    if not metrics:
        raise RuntimeError("No completed generations to report")
    rows = [{"generation": m["generation"], "validation_loss": m["validation"]["loss"],
             "validation_perplexity": m["validation"]["perplexity"], "test_loss": m["test"]["loss"],
             "test_perplexity": m["test"]["perplexity"], "train_seconds": m["training"]["seconds"],
             "generation_seconds": m["generation_timing"]["seconds"], **m["diversity"]} for m in metrics]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    x = [r["generation"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for name in ("validation", "test"):
        axes[0].plot(x, [r[name + "_loss"] for r in rows], "o-", label=name)
        axes[1].plot(x, [r[name + "_perplexity"] for r in rows], "o-", label=name)
    axes[0].set_ylabel("Token-weighted mean NLL")
    axes[1].set_ylabel("Corpus perplexity")
    for a in axes:
        a.set_xlabel("Synthetic generation (G0 = real fine-tuning)")
        a.set_xticks(x)
        a.legend()
        a.grid(alpha=.2)
    fig.suptitle(f"{c['kind'].upper()} — held-out real WikiText-2")
    fig.savefig(out / "perplexity.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for n in (1, 2, 3):
        axes[0].plot(x, [r[f"distinct_{n}"] for r in rows], "o-", label=f"distinct-{n}")
    for key, label in [("repeated_4gram_rate", "repeated 4-grams"), ("duplicate_continuation_rate", "duplicate continuations")]:
        axes[1].plot(x, [r[key] for r in rows], "o-", label=label)
    for a in axes:
        a.set_xlabel("Generation")
        a.set_xticks(x)
        a.set_ylabel("Fraction")
        a.legend()
        a.grid(alpha=.2)
    fig.suptitle("Fixed 256-prompt probe; continuation tokens only")
    fig.savefig(out / "diversity.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for m in metrics:
        p = seed_root / f"gen_{m['generation']:02d}" / "reference_sequences.json"
        vals = read_json(p)["sequence_mean_nll"]
        ax.hist(vals, bins=30, histtype="step", density=True, label=f"G{m['generation']}")
    ax.set(xlabel="Continuation mean NLL under frozen G0 (63 shifted targets)", ylabel="Density")
    ax.legend()
    fig.savefig(out / "reference_scores.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.bar(np.asarray(x) - .18, [r["train_seconds"] / 60 for r in rows], width=.36, label="training/checkpoint")
    ax.bar(np.asarray(x) + .18, [r["generation_seconds"] / 60 for r in rows], width=.36, label="generation")
    ax.set(xlabel="Generation", ylabel="Minutes", xticks=x)
    ax.legend()
    fig.savefig(out / "runtime.png", dpi=180)
    plt.close(fig)
    baseline = read_json(seed_root / "pretrained" / "metrics.json")
    lines = [f"# Week 1 {c['kind']} experiment", "", f"Experiment: `{root.name}`. Seed: {c['seed']}.", "",
             "This small smoke experiment validates plumbing, not evidence of model collapse." if c["kind"] == "smoke" else "One seed is preliminary evidence; non-monotonic changes are reported without adjustment.", "",
             f"Untouched pretrained test corpus perplexity: {baseline['test']['perplexity']:.4f}.", "",
             "| Generation | Validation PPL | Test PPL | Distinct-2 | Repeated 4-grams |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| G{r['generation']} | {r['validation_perplexity']:.4f} | {r['test_perplexity']:.4f} | {r['distinct_2']:.4f} | {r['repeated_4gram_rate']:.4f} |")
    lines += ["", "Corpus PPL = exp(sum of NLL / count of scored shifted targets); each real block scores 63 targets. No average of batch perplexities.",
              "", "Distinct-n counts unique n-grams divided by all n-grams within individual continuations. Repeated 4-gram rate counts occurrences beyond the first divided by all within-continuation 4-grams. Exact duplicate rate = 1 − unique continuations / continuation count. Special tokens remain in token metrics.",
              "", "All training starts from the pinned original pretrained model. Prompts are fixed real training blocks; only continuation token IDs are used as subsequent training targets. Validation selects the best epoch; test data only reports results.",
              "", "Reconstruction differences: split-wide concatenation (authors group map batches); explicit fresh-base initialization; BF16 autocast; corrected token-weighted corpus PPL; repetition penalty 1.0 for the paper's main condition (released code hardcodes 3.0). No real training targets retained after G0.",
              "", "Paper: https://doi.org/10.1038/s41586-024-07566-y ; authors' code: https://github.com/iliaishacked/curse_recurse"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report": str(out / "report.md"), "completed_generations": len(metrics)}
