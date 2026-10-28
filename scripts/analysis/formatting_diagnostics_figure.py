"""Plot supplementary formatting diagnostics from frozen audit CSVs.

Exclusion is post-hoc sensitivity, with unequal retained sample sizes. It must
never replace the original raw-token metrics or be presented as semantic quality.
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
import pandas as pd

from model_collapse.common import PROJECT, atomic_json, sha256


def render(inputs: Path, out: Path):
    if out.exists() or out.with_suffix(".manifest.json").exists():
        raise ValueError("Use a new output path; prior publications are preserved")
    cycle_path, row_path = inputs / "probe_diagnostics.csv", inputs / "final_output_metrics.csv"
    cycles, rows = pd.read_csv(cycle_path), pd.read_csv(row_path)
    colors, labels = {"continued": "#cc6b27", "reset_s0": "#755493"}, {"continued": "Continued B", "reset_s0": "Reset A"}
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 7.3, "axes.titlesize": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(6.26, 4.45), layout="constrained")
    for arm in colors:
        g = cycles[cycles.arm == arm].sort_values("cycle")
        assert g.cycle.tolist() == list(range(11))
        axes[0, 0].plot(g.cycle, g.equals_tokens / 16384 * 100, "o-", color=colors[arm], ms=2.5, label=labels[arm])
        axes[0, 1].plot(g.cycle, g.pooled_entropy_nats, color=colors[arm], label=labels[arm] + ": raw")
        axes[0, 1].plot(g.cycle, g.without_equal_entropy_nats, "--", color=colors[arm], label=labels[arm] + ": exclude =")
        final = rows[rows.arm == arm].sort_values("prompt_index")
        assert len(final) == 256 and final.cycle.eq(10).all()
        axes[1, 0].hist(final.entropy_nats, bins=np.linspace(0, 4.2, 22), histtype="step", lw=1.5, color=colors[arm], label=labels[arm])
    axes[0, 0].set(title="A  Generated '=' tokens", xlabel="Cycle", ylabel="Probe tokens (%)", ylim=(0, 29), xticks=[0, 2, 4, 6, 8, 10])
    axes[0, 1].set(title="B  Formatting sensitivity", xlabel="Cycle", ylabel="Pooled entropy (nats)", xticks=[0, 2, 4, 6, 8, 10])
    axes[0, 0].legend(fontsize=6.5, frameon=False, loc="upper left")
    axes[0, 1].legend(fontsize=6, frameon=False, loc="lower left")
    axes[1, 0].set(title="C  Entropy within each final output", xlabel="Within-output entropy (nats)", ylabel="Number of outputs")
    axes[1, 0].legend(fontsize=6.5, frameon=False, loc="upper left")
    a, b = [rows[rows.arm == arm].sort_values("prompt_index").set_index("prompt_index") for arm in ("reset_s0", "continued")]
    assert a.index.equals(b.index)
    axes[1, 1].scatter(b.entropy_nats, a.entropy_nats, color="#777777", alpha=.4, s=8, edgecolor="none")
    axes[1, 1].plot([0, 4.2], [0, 4.2], "--", color="#999999", lw=.7)
    point = (b.loc[0, "entropy_nats"], a.loc[0, "entropy_nats"])
    axes[1, 1].scatter(*point, s=32, color="#b35d8d", zorder=4)
    axes[1, 1].annotate("Quoted prompt 0", point, xytext=(.2, 3.9), color="#9a416e", fontsize=6.5,
                       arrowprops={"arrowstyle": "->", "color": "#b35d8d", "lw": .7})
    axes[1, 1].set(title="D  One prompt can reverse the ranking", xlabel="Continued B10 entropy (nats)", ylabel="Reset A10 entropy (nats)", xlim=(-.1, 4.3), ylim=(-.1, 4.3))
    for ax in axes.flat:
        ax.grid(alpha=.15)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=280)
    plt.close(fig)
    atomic_json(out.with_suffix(".manifest.json"), {
        "scope": "Supplementary post-hoc diagnostic, fixed first256 prompts from six adjacent articles. No new inference.",
        "caveat": "Excluding equality IDs renormalizes frequencies and changes retained sample sizes; higher entropy is not semantic quality.",
        "input_hashes": {str(p.relative_to(PROJECT)): sha256(p) for p in (cycle_path, row_path)},
        "script_sha256": sha256(__file__),
        "analysis_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "environment": {"python": platform.python_version(), **{k: importlib.metadata.version(k) for k in ("numpy", "matplotlib", "pandas")}},
        "output": {"file": out.name, "sha256": sha256(out)},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.inputs.resolve(), args.out.resolve())
