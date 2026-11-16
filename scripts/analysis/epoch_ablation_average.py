"""Add the arithmetic mean of the verified real/synthetic branch perplexities."""
from __future__ import annotations

import argparse
import csv
import math
import platform
from pathlib import Path

from model_collapse.common import PROJECT, atomic_json, read_json, sha256


RUN = "20261116T195118Z-epoch-ablation"
SOURCE = PROJECT / "reports" / RUN / "generated/verified-20261116"
HISTORY = PROJECT / "reports/20261114T001352Z-full-completion64-mixtures/generated/update-gradients-20261114"


def build(out):
    out = Path(out)
    if out.exists():
        raise FileExistsError("Choose a new output directory; existing evidence is preserved")
    inputs = {}
    for folder, manifest_name in ((SOURCE, "analysis_manifest.json"), (HISTORY, "publication_manifest.json")):
        manifest = folder / manifest_name
        expected = read_json(manifest)
        expected = (expected["outputs"]["per_update.csv"] if folder == SOURCE
                    else expected["files"]["per_update.csv"]["sha256"])
        path = folder / "per_update.csv"
        if sha256(path) != expected:
            raise RuntimeError("Published input checksum differs: " + str(path))
        inputs[path.relative_to(PROJECT).as_posix()] = sha256(path)
        inputs[manifest.relative_to(PROJECT).as_posix()] = sha256(manifest)
    with (SOURCE / "per_update.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (HISTORY / "per_update.csv").open(encoding="utf-8", newline="") as stream:
        history = {(int(r["cycle"]), int(r["step"])): float(r["mixed_probe_perplexity"])
                   for r in csv.DictReader(stream) if r["arm"] == "mixed99"}
    if len(rows) != 711:
        raise RuntimeError("Expected nine complete 79-update branches")
    values, endpoints = [], []
    for cycle in (3, 6, 10):
        branches = {source: sorted((r for r in rows if int(r["cycle"]) == cycle and r["source"] == source),
                                  key=lambda r: int(r["local_step"]))
                    for source in ("both", "real", "synthetic")}
        for rr in branches.values():
            if [int(r["local_step"]) for r in rr] != list(range(1, 80)):
                raise RuntimeError("Missing/duplicate branch updates")
            for r in rr:
                if not math.isclose(float(r["probe_perplexity"]), math.exp(float(r["probe_nll"])), rel_tol=1e-12):
                    raise RuntimeError("Perplexity/NLL identity differs")
        initial = float(branches["both"][0]["probe_nll_before"])
        if any(float(rr[0]["probe_nll_before"]) != initial for rr in branches.values()):
            raise RuntimeError("Branches do not share an initializer probe score")
        curves = {source: [math.exp(initial)] + [float(r["probe_perplexity"]) for r in rr]
                  for source, rr in branches.items()}
        for i in range(79):
            if len({(rr[i]["step"], rr[i]["original_rows"]) for rr in branches.values()}) != 1:
                raise RuntimeError("Branch schedules differ")
        for step in range(80):
            real, synthetic, both = (curves[s][step] for s in ("real", "synthetic", "both"))
            mean = (real + synthetic) / 2.0
            if not min(real, synthetic) <= mean <= max(real, synthetic):
                raise RuntimeError("Arithmetic mean lies outside its components")
            values.append({"cycle": cycle, "local_step": step, "cycle_position": cycle-1+step/79,
                           "real_only_ppl": real, "synthetic_only_ppl": synthetic,
                           "arithmetic_mean_ppl": mean, "both_ppl": both,
                           "both_minus_arithmetic_mean": both-mean})
        endpoints.append(values[-1])
    out.mkdir(parents=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for cycle in (3, 6, 10):
        rr = [r for r in values if r["cycle"] == cycle]
        x = [r["cycle_position"] for r in rr]
        fig, ax = plt.subplots(figsize=(8.5, 8.5))
        ax.set_box_aspect(1)
        for key, color, style, label in (
            ("both_ppl", "#7651af", "-", "Both: replay of 99% mixture"),
            ("real_only_ppl", "#767676", "--", "Real-only: original 1% weight"),
            ("synthetic_only_ppl", "#444444", ":", "Synthetic-only: original 99% weight"),
            ("arithmetic_mean_ppl", "#d68a00", "-.", "Arithmetic mean of the two grey curves"),
        ):
            ax.plot(x, [r[key] for r in rr], color=color, ls=style, lw=2.2, label=label)
        ax.scatter([cycle], [history[(cycle, cycle*79)]], marker="*", s=110, c="black",
                   label="Historical mixed endpoint", zorder=5)
        fig.suptitle(f"Cycle {cycle}: one epoch from identical M{cycle-1} weights", y=.985, fontsize=14)
        ax.set(xlabel="Training cycle (fraction = update position)",
               ylabel="Fixed real-validation probe perplexity")
        ax.grid(alpha=.2)
        ax.legend(fontsize=9, loc="lower center", bbox_to_anchor=(.5, 1.015),
                  ncol=2, frameon=False, columnspacing=1.2)
        fig.text(.5, .027, "Orange = (real-only PPL + synthetic-only PPL) / 2 at the same update.",
                 ha="center", fontsize=8)
        fig.text(.5, .01, "128 fixed real pairs. The average is a visual reference, not a training-update prediction.",
                 ha="center", fontsize=8)
        fig.subplots_adjust(left=.13, right=.98, bottom=.115, top=.855)
        name = f"cycle_{cycle:02d}_probe_perplexity_with_average"
        fig.savefig(out / (name+".png"), dpi=180)
        fig.savefig(out / (name+".pdf"), metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)
    with (out / "pointwise_average.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
    text = ["# Point-by-point average of the real-only and synthetic-only branches", "",
            "The orange curve is the equal-weight arithmetic mean of the two branch perplexities at each corresponding update, including the shared starting point. It is calculated directly in perplexity space, not by averaging NLL and exponentiating. The original branch source weighting is unchanged.", "",
            "The mean is a visual reference. It does not describe an averaged gradient, combined model, or expected trajectory of the mixed training branch. Nonlinear loss evaluation and diverging branch weights prevent that interpretation.", "",
            "| Cycle | Mean of branch endpoint PPLs | Actual mixed endpoint PPL |", "| --- | ---: | ---: |"]
    for r in endpoints:
        text.append(f"| {r['cycle']} | {r['arithmetic_mean_ppl']:.2f} | {r['both_ppl']:.2f} |")
    text += ["", "All 240 points (baseline plus 79 updates for each of three cases) are in pointwise_average.csv. This analysis uses only previously verified published data and starts no compute.", ""]
    for cycle in (3, 6, 10):
        text += [f"![Cycle {cycle} with arithmetic mean](cycle_{cycle:02d}_probe_perplexity_with_average.png)", ""]
    (out / "report.md").write_text("\n".join(text), encoding="utf-8")
    atomic_json(out / "analysis_manifest.json", {
        "formula": "(real_only_probe_perplexity + synthetic_only_probe_perplexity) / 2",
        "equal_weight_curve_average": True, "target_metric": "perplexity", "points": len(values),
        "inputs": inputs, "script_sha256": sha256(__file__), "python": platform.python_version(),
        "matplotlib": matplotlib.__version__, "new_training_or_inference": False,
        "outputs": {p.name: sha256(p) for p in out.iterdir() if p.is_file()},
    })
    return {"out": str(out), "points": len(values), "endpoints": endpoints}


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    print(json.dumps(build(parser.parse_args().out), indent=2))
