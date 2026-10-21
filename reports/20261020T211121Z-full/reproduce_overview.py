"""Rebuild the overview from the published CSV without cloud access or a GPU."""
from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    folder = Path(__file__).resolve().parent
    with (folder / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    generations = [int(row["generation"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    for key, label in (("validation_perplexity", "Validation"), ("test_perplexity", "Test")):
        axes[0].plot(generations, [float(row[key]) for row in rows],
                     marker="o", markersize=4, label=label)
    axes[0].set(xlabel="Generation (G0 trained on real data)",
                ylabel="Corpus perplexity", xticks=generations,
                title="Fixed held-out real text")
    axes[0].legend(frameon=False)
    axes[1].plot(generations, [100 * float(row["repeated_4gram_rate"]) for row in rows],
                 marker="o", color="#c0613b", markersize=4)
    axes[1].set(xlabel="Generation", ylabel="Repeated 4-grams (%)",
                xticks=generations, title="Same 256 real training prompts")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("OPT-125M / WikiText-2 recursive training (seed 42)", fontsize=12)
    destination = folder / "results_overview.png"
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    print(destination)


if __name__ == "__main__":
    main()
