"""Render the Week 1 six-panel report figure from frozen published statistics.

The explicit block-loss CSV is a new presentation input exported from verified
saved test_sequence records. Never overwrite a previous figure or its manifest.
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
import numpy as np

from model_collapse.analysis.records import read_metrics_csv
from model_collapse.common import PROJECT, atomic_json, sha256


def render(block_losses: Path, out: Path):
    if out.exists() or out.with_suffix(".manifest.json").exists():
        raise ValueError("Choose a new destination; published figures are preserved")
    source = PROJECT / "reports/20261020T211121Z-full/metrics.csv"
    rows = read_metrics_csv(source)
    assert [r["generation"] for r in rows] == list(range(11))
    arrays = np.genfromtxt(block_losses, delimiter=",", names=True)
    assert len(arrays) == 4494 and all(np.isfinite(arrays[k]).all() for k in ("G0", "G1", "G10"))
    bins = np.linspace(1, 8, 36)
    assert all(arrays[k].min() >= bins[0] and arrays[k].max() <= bins[-1] for k in ("G0", "G1", "G10"))
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(3, 2, figsize=(7, 7.45), layout="constrained")
    def values(key, scale=1):
        return [r[key] * scale for r in rows]
    axes[0, 0].plot(range(11), values("test_perplexity"), "o-", color="#2268a7", ms=3, label="Test")
    axes[0, 0].plot(range(11), values("validation_perplexity"), "--", color="#777777", label="Validation")
    axes[0, 0].set(title="(a) Held-out real text", ylabel="Corpus perplexity")
    axes[0, 0].legend(fontsize=7)
    for n, color in ((1, "#7974b5"), (2, "#1b9e77"), (3, "#d95f02")):
        axes[0, 1].plot(range(11), values(f"distinct_{n}", 100), "o-", color=color, ms=3, label=f"Distinct-{n}")
    axes[0, 1].set(ylabel="Unique / total n-grams (%)")
    axes[0, 1].set_title("(b) N-gram diversity", pad=26)
    axes[0, 1].legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=3, fontsize=6.5, frameon=False)
    for ax, key, title, ylabel, color in (
        (axes[1, 0], "repeated_4gram_rate", "(c) Repeated 4-grams", "Excess occurrences (%)", "#b85c00"),
        (axes[1, 1], "duplicate_continuation_rate", "(d) Duplicate continuations", "Excess complete copies (%)", "#7c329e"),
        (axes[2, 0], "token_entropy_nats", "(e) Empirical token entropy", "Entropy (nats)", "#008c7d")):
        ax.plot(range(11), values(key, 1 if key == "token_entropy_nats" else 100), "o-", ms=3, color=color)
        ax.set(title=title, ylabel=ylabel)
    for label, color in (("G0", "#2268a7"), ("G1", "#b85c00"), ("G10", "#7c329e")):
        axes[2, 1].hist(arrays[label], bins=bins, density=True, histtype="step", lw=1.4, color=color, label=label)
    axes[2, 1].set(title="(f) Real-text loss distributions", xlabel="Block-mean NLL (nats/target)", ylabel="Density")
    axes[2, 1].legend(fontsize=7)
    for i, ax in enumerate(axes.flat):
        ax.grid(alpha=.2)
        if i < 5:
            ax.set(xlabel="Generation", xticks=[0, 2, 4, 6, 8, 10])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240)
    plt.close(fig)
    atomic_json(out.with_suffix(".manifest.json"), {
        "scope": "New presentation revision; original measurements and historical figure preserved.",
        "inputs": {str(p.relative_to(PROJECT)): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in (source, block_losses)},
        "histogram": {"common_bin_edges": bins.tolist(), "density": True, "blocks_per_series": 4494, "all_blocks_in_range": True},
        "script_sha256": sha256(__file__),
        "analysis_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "environment": {"python": platform.python_version(), **{k: importlib.metadata.version(k) for k in ("numpy", "matplotlib")}},
        "output": {"file": out.name, "sha256": sha256(out)},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-losses", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.block_losses.resolve(), args.out.resolve())
