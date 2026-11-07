from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

root = Path(__file__).resolve().parent
rows = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
colors = {0: "#0072B2", .2: "#009E73", .5: "#E69F00", .8: "#D55E00", 1: "#CC79A7"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
for split in ("test", "validation"):
    metric = split + "_perplexity"
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_box_aspect(1)
    for fraction, color in colors.items():
        selected = [r for r in rows if r["synthetic_fraction"] == fraction]
        ax.plot([r["cycle"] for r in selected], [r[metric] for r in selected],
                color=color, marker="o", linewidth=2.2, markersize=4.5,
                label=f"{fraction:.0%} synthetic / {1-fraction:.0%} real")
    ax.set_title(f"Fixed FineWeb {split} perplexity", fontsize=16, pad=15)
    ax.set_xlabel("Training cycle (0 = pretrained OPT-125M)", labelpad=10)
    ax.set_ylabel("Corpus perplexity (lower is better)", labelpad=10)
    ax.set_xticks(range(11))
    ax.set_xlim(-.2, 10.3)
    ax.set_ylim(0, max(r[metric] for r in rows) * 1.10)
    ax.grid(alpha=.20)
    ax.legend(loc="upper left", framealpha=.95, fontsize=10)
    inset = ax.inset_axes([.48, .13, .48, .30])
    low = [r[metric] for r in rows if r["synthetic_fraction"] < 1]
    for fraction, color in colors.items():
        if fraction == 1:
            continue
        selected = [r for r in rows if r["synthetic_fraction"] == fraction]
        inset.plot([r["cycle"] for r in selected], [r[metric] for r in selected],
                   color=color, marker="o", linewidth=1.7, markersize=3)
    inset.set_title("0-80% synthetic (zoom)", fontsize=9, pad=7)
    inset.set_xlim(-.2, 10.3)
    inset.set_ylim(min(low) - .12, max(low) + .12)
    inset.set_xticks([0, 5, 10])
    inset.yaxis.set_major_locator(MaxNLocator(4))
    inset.tick_params(labelsize=9)
    inset.grid(alpha=.20)
    scored = 457664 if split == "test" else 468608
    fig.text(.5, .025, f"Fixed real continuations: {scored:,} scored tokens. One epoch per fresh chunk. Seed 42.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.savefig(root / (metric + "_five_conditions.png"), dpi=200)
    plt.close(fig)
