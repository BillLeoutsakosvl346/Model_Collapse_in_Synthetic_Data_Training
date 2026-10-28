"""Render the continued-training experiment on its own, without comparison arms.

Uses frozen public statistics only. Writes a new presentation file and never
changes the original Week 2 report, CSV, figure or evidence records.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import platform
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_collapse.analysis.records import read_metrics_csv
from model_collapse.common import PROJECT, atomic_json, read_json, sha256


def render(out):
    source = PROJECT / "reports/20261026T192740Z-full-continued"
    if out.exists():
        raise ValueError("Choose a new figure path; existing publications are preserved")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = read_metrics_csv(source / "metrics.csv")
    assert [r["generation"] for r in rows] == list(range(11))
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 7.45), layout="constrained")
    x = range(11)
    orange, purple = "#cc6b27", "#7b5c91"
    def values(key, scale=1):
        return [r[key] * scale for r in rows]
    axes[0, 0].plot(x, values("test_perplexity"), "o-", color=orange, ms=3)
    axes[0, 0].set(title="A  Fixed real-test perplexity", ylabel="Corpus perplexity")
    for n, color in ((1, "#437f97"), (2, orange), (3, "#4a8c63")):
        axes[0, 1].plot(x, values(f"distinct_{n}", 100), "o-", color=color, label=f"Distinct-{n}", ms=3)
    axes[0, 1].set(title="B  Output diversity", ylabel="Unique / total n-grams (%)")
    axes[0, 1].legend(fontsize=6.8, loc="lower center",
                      bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False)
    axes[0, 1].set_title("B  Output diversity", pad=26)
    axes[1, 0].plot(x, values("repeated_4gram_rate", 100), "o-", color=orange, label="Repeated 4-grams", ms=3)
    axes[1, 0].plot(x, values("duplicate_continuation_rate", 100), "s--", color=purple, label="Exact duplicates", ms=3)
    axes[1, 0].set(title="C  Repetition and duplication", ylabel="Probe fraction (%)")
    axes[1, 0].legend(fontsize=7.5)
    axes[1, 1].plot(x, values("token_entropy_nats"), "o-", color=orange, ms=3)
    axes[1, 1].set(title="D  Empirical token entropy", ylabel="Nats")
    axes[2, 0].plot(x, values("frozen_s0_nll"), "o-", color=orange, ms=3)
    axes[2, 0].set(title="E  Outputs scored under frozen S0", ylabel="Mean continuation NLL")
    inputs = {"metrics.csv": sha256(source / "metrics.csv")}
    for g, color in ((1, "#437f97"), (5, orange), (10, purple)):
        rel = f"records/seed_42/gen_{g:02d}/training_history.json"
        h = read_json(source / rel)
        assert len(h) == 5
        best = min(h, key=lambda r: r["validation_loss"])
        axes[2, 1].plot(range(1, 6), [r["validation_loss"] for r in h], "o-", color=color, label=f"S{g}", ms=3)
        axes[2, 1].scatter(best["epoch"], best["validation_loss"], marker="*", s=65, color=color, zorder=5)
        inputs[rel] = sha256(source / rel)
    axes[2, 1].set(title="F  Within-cycle validation loss", ylabel="Real-validation NLL", xlabel="Epoch", xticks=range(1, 6))
    axes[2, 1].legend(fontsize=7.5)
    for i, axis in enumerate(axes.flat):
        axis.grid(alpha=.2)
        if i < 5:
            axis.set(xlabel="Synthetic-training cycle", xticks=[0, 2, 4, 6, 8, 10])
    fig.savefig(out, dpi=240)
    plt.close(fig)
    atomic_json(out.with_suffix(".manifest.json"), {
        "condition": "continued", "run_id": source.name,
        "scope": "Presentation revision only; one experiment, no Week 1 or reset-arm overlays.",
        "source_hashes": inputs, "script_sha256": sha256(__file__),
        "analysis_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
        ).strip(),
        "environment": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ("numpy", "matplotlib")
        }},
        "output": {"file": out.name, "sha256": sha256(out)},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.out.resolve())
