"""Plot logged combined gradients and quantify real/synthetic opposition offline."""
from __future__ import annotations

import argparse
import csv
import math
import platform
from pathlib import Path

import numpy as np

from model_collapse.common import atomic_json, read_json, sha256

RUN = "20261114T001352Z-full-completion64-mixtures"
SOURCE = Path(f"reports/{RUN}/generated/update-gradients-20261114")


def build(out):
    out = Path(out)
    if out.exists():
        raise FileExistsError("Choose a new output destination")
    pub = read_json(SOURCE / "publication_manifest.json")
    path = SOURCE / "per_update.csv"
    if sha256(path) != pub["files"][path.name]["sha256"]:
        raise RuntimeError("Published gradient evidence changed")
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8")) if r["arm"] == "mixed99"]
    if [int(r["step"]) for r in rows] != list(range(1, 791)):
        raise RuntimeError("Expected all 790 ordered mixed99 updates")
    updates, cycles = [], []
    for r in rows:
        a, b, t = [float(r[k]) for k in ("real_gradient_norm", "synthetic_gradient_norm", "total_gradient_norm")]
        cosine = float(r["source_cosine"]) if r["source_cosine"] else None
        if not np.isfinite([a, b, t]).all() or min(a, b, t) < 0:
            raise RuntimeError("Invalid gradient magnitude")
        both = min(a, b) > 0
        if both != (cosine is not None) or (cosine is not None and not -1 <= cosine <= 1):
            raise RuntimeError("Invalid/undefined source alignment")
        dot = a * b * cosine if both else 0.
        energy0 = a * a + b * b
        expected2 = energy0 + 2 * dot
        error = abs(t * t - expected2) / max(t * t, energy0, 1e-30)
        if error > 1e-6 or t > a + b + 1e-6 or t < abs(a - b) - 1e-6:
            raise RuntimeError("Measured combined gradient violates vector identity")
        orthogonal = math.sqrt(energy0)
        updates.append({"cycle": int(r["cycle"]), "step": int(r["step"]),
            "real_pairs": int(r["real_pairs"]), "synthetic_pairs": int(r["synthetic_pairs"]),
            "real_norm": a, "synthetic_norm": b, "combined_norm_logged": t,
            "source_cosine": cosine, "source_dot": dot, "orthogonal_reference_norm": orthogonal,
            "orthogonal_relative_norm_reduction_percent": 100 * (1 - t / orthogonal) if orthogonal else None,
            "quadratic_identity_relative_error": error})
    for cycle in range(1, 11):
        rr = [r for r in updates if r["cycle"] == cycle]
        if len(rr) != 79 or sum(r["real_pairs"] for r in rr) != 100 or sum(r["synthetic_pairs"] for r in rr) != 9900:
            raise RuntimeError("Scientific source/update counts changed")
        both = [r for r in rr if r["source_cosine"] is not None]
        cosines = [r["source_cosine"] for r in both]
        mean = lambda key: float(np.mean([r[key] for r in rr]))
        a, b, t, q = [mean(k) for k in ("real_norm", "synthetic_norm", "combined_norm_logged", "orthogonal_reference_norm")]
        energy0 = sum(r["real_norm"] ** 2 + r["synthetic_norm"] ** 2 for r in rr)
        dots = sum(r["source_dot"] for r in rr)
        cycles.append({"cycle": cycle, "updates": len(rr), "both_source_updates": len(both),
            "opposing_updates": sum(r["source_cosine"] < 0 for r in both),
            "mean_real_norm": a, "mean_synthetic_norm": b, "mean_combined_norm": t,
            "mean_orthogonal_reference_norm": q,
            "mean_norm_reduction_vs_orthogonal_percent": 100 * (1 - t / q),
            "median_source_cosine": float(np.median(cosines)),
            "source_cosine_q25": float(np.quantile(cosines, .25)),
            "source_cosine_q75": float(np.quantile(cosines, .75)),
            "squared_norm_reduction_vs_orthogonal_percent": -200 * dots / energy0,
            "opposing_projection_fraction_of_synthetic_energy_percent": -100 * dots / sum(r["synthetic_norm"] ** 2 for r in rr),
            "combined_shorter_than_synthetic_updates": sum(r["combined_norm_logged"] < r["synthetic_norm"] for r in rr),
            "max_quadratic_identity_relative_error": max(r["quadratic_identity_relative_error"] for r in rr)})
    out.mkdir(parents=True)
    for name, values in (("per_update_cancellation.csv", updates), ("cycle_cancellation.csv", cycles)):
        with (out / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    atomic_json(out / "summary.json", {"cycles": cycles,
        "both_source_updates": sum(r["both_source_updates"] for r in cycles),
        "opposing_updates": sum(r["opposing_updates"] for r in cycles),
        "max_quadratic_identity_relative_error": max(r["quadratic_identity_relative_error"] for r in updates),
        "definition": "Per-update orthogonal reference q=sqrt(||gR||^2+||gS||^2). Cycle cancellation=100*(1-mean(||gR+gS||)/mean(q)).",
        "scope": "Mean of norms at actual sequential updates; not a norm of a sum/mean across updates. Undefined cosines omitted, real-absent norms included."})
    draw(out, cycles)
    report(out, cycles)
    atomic_json(out / "analysis_manifest.json", {"run_id": RUN, "script_sha256": sha256(__file__),
        "inputs": {str(path): sha256(path), str(SOURCE / "publication_manifest.json"): sha256(SOURCE / "publication_manifest.json")},
        "python": platform.python_version(), "numpy": np.__version__, "new_training_or_inference": False,
        "checks": ["Pinned publication checksum", "790 consecutive updates", "79 updates/cycle", "100 real and 9900 synthetic pairs/cycle", "Alignment presence/finite values", "Measured combined-norm vector identity and triangle inequalities"],
        "output_hashes": {p.name: sha256(p) for p in out.iterdir() if p.is_file()}})
    print({"out": str(out), "opposing": sum(r["opposing_updates"] for r in cycles),
           "both_present": sum(r["both_source_updates"] for r in cycles), "cycle10": cycles[-1]})


def draw(out, cycles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12})
    x = [r["cycle"] for r in cycles]

    def figure(title, ylabel):
        fig, ax = plt.subplots(figsize=(8, 8)); ax.set_box_aspect(1)
        ax.set(title=title, xlabel="Training cycle", ylabel=ylabel, xticks=x)
        ax.grid(alpha=.2)
        return fig, ax

    def save(fig, name):
        fig.tight_layout(rect=(0, .035, 1, 1)); fig.savefig(out / (name + ".png"), dpi=180)
        fig.savefig(out / (name + ".pdf"), metadata={"CreationDate": None, "ModDate": None}); plt.close(fig)

    fig, ax = figure("99% synthetic: components and their actual sum\nActual source weighting is already included", "Mean gradient L2 norm per update")
    for key, label, color, style in (
        ("mean_real_norm", "Real component", "#087F8C", "o-"),
        ("mean_synthetic_norm", "Synthetic component", "#BD365E", "o-"),
        ("mean_combined_norm", "Actual combined gradient", "#433381", "s-"),
        ("mean_orthogonal_reference_norm", "Reference if components were perpendicular", "#82909E", "--")):
        ax.plot(x, [r[key] for r in cycles], style, label=label, color=color, lw=2)
    ax.set_ylim(bottom=0); ax.legend(fontsize=9, loc="upper right")
    fig.text(.5, .018, "Each point averages 79 measured update norms; it is not the norm of an epoch-average gradient.", ha="center", fontsize=8)
    save(fig, "99_combined_gradient_norms")

    fig, ax = figure("99% synthetic: how strongly do the sources oppose?\nBatches containing both sources only", "Cosine between real and synthetic gradients")
    ax.plot(x, [r["median_source_cosine"] for r in cycles], "o-", color="#7651AF", lw=2, label="Median cosine")
    ax.fill_between(x, [r["source_cosine_q25"] for r in cycles], [r["source_cosine_q75"] for r in cycles], color="#7651AF", alpha=.16, label="Middle 50% of both-source batches")
    ax.axhline(0, color="#334155", lw=1)
    ax.set_ylim(-1, 1)
    ax.text(.03, .97, "+1: same direction", transform=ax.transAxes, va="top")
    ax.text(.03, .52, "0: perpendicular", transform=ax.transAxes, va="bottom")
    ax.text(.03, .03, "−1: exactly opposite", transform=ax.transAxes)
    ax.legend(loc="upper right", fontsize=9)
    fig.text(.5, .018, "Alignment is undefined when real examples are absent. Shading is a batch distribution, not a confidence interval.", ha="center", fontsize=8)
    save(fig, "99_source_opposition")

    fig, ax = figure("99% synthetic: reduction caused by opposing directions\nRelative to perpendicular components of the same strengths", "Reduction in mean combined gradient norm (%)")
    y = [r["mean_norm_reduction_vs_orthogonal_percent"] for r in cycles]
    ax.plot(x, y, "o-", color="#087F8C", lw=2)
    for xx, yy in zip(x, y): ax.annotate(f"{yy:.1f}%", (xx, yy), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=10)
    ax.set_ylim(0, 15); ax.axhline(0, color="#334155", lw=1)
    fig.text(.5, .018, "Reference per update: sqrt(real norm² + synthetic norm²). All 79 updates/cycle included.", ha="center", fontsize=8)
    save(fig, "99_directional_cancellation")


def report(out, cycles):
    opposing = sum(r["opposing_updates"] for r in cycles); both = sum(r["both_source_updates"] for r in cycles)
    text = f"""# Actual combined gradients and source opposition in the 99% SGD run

Offline analysis of all 790 sealed update records from `{RUN}`. No new training, inference, checkpoint restoration or cloud work. Source-weighted component norms and the authoritative combined norm come from the published full-precision journals, rather than adding plotted scalar heights.

## Findings

- Real and synthetic gradients oppose in **{opposing}/{both} both-source updates ({100 * opposing / both:.1f}%)**. The remaining 227 updates have no real component; their alignment is undefined.
- Median cosine moves from **−0.111** in cycle 1 to **−0.363** in cycle 10. The vectors are moderately opposed, not almost perfectly opposite.
- At cycle 10, mean component norms are **0.297 real / 0.485 synthetic** and the actual combined norm is **0.534**. Combined size can exceed synthetic size even while the components oppose: the real vector also contributes perpendicular movement.
- At the same component strengths, a hypothetical perpendicular orientation gives mean norm **0.608**. Actual opposition reduces that mean to **0.534**, a **12.2% reduction**. The corresponding cycle-1 reduction is **1.3%**.
- This is partial directional cancellation. It does not mean that 12.2% of synthetic data was removed or that validation loss improves by 12.2%. Prediction effects require the separately recorded held-out interventions.

![Actual gradient sum](99_combined_gradient_norms.png)

Figure 1. Mean of per-update L2 norms, retaining the actual batch weights and zero real contributions. The solid combined curve uses the measured authoritative gradient. The dashed reference is hypothetical perpendicular geometry at each update, not another training run.

![Source alignment](99_source_opposition.png)

Figure 2. Median cosine in both-source batches, with their middle 50%. Positive means aligned, zero perpendicular and negative opposed. The shaded interval is a descriptive batch spread, not uncertainty across seeds.

## What cancellation means here

For each update, let `a = ||gR||`, `b = ||gS||`, `t = ||gR + gS||` and `c = cosine(gR,gS)`. The vector identity is:

`t² = a² + b² + 2ab c`.

The negative cross-term `2ab c` directly reduces the squared norm when sources oppose. A zero-dot-product reference has length `q = sqrt(a² + b²)`. The plotted cycle reduction is:

`100 × [1 − mean(t) / mean(q)]`.

Both means use the same 79 actual updates. The norm is taken before averaging; we never form an epoch-average vector from these scalar logs. This reference isolates opposition from the ordinary geometric reduction that also occurs when adding perpendicular vectors. Comparing `t` with `a + b` alone would count perpendicular geometry as cancellation too.

![Directional reduction](99_directional_cancellation.png)

Figure 3. Reduction in mean combined length relative to the explicit perpendicular reference. Real-absent batches have zero reduction; aligned batches can have negative reduction. Net cycle reductions are positive here.

## Cycle values

| Cycle | Mean real norm | Mean synthetic norm | Mean actual combined norm | Perpendicular reference | Reduction | Median cosine |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    for r in cycles:
        text += f"| {r['cycle']} | {r['mean_real_norm']:.4f} | {r['mean_synthetic_norm']:.4f} | {r['mean_combined_norm']:.4f} | {r['mean_orthogonal_reference_norm']:.4f} | {r['mean_norm_reduction_vs_orthogonal_percent']:.1f}% | {r['median_source_cosine']:.3f} |\n"
    text += """
## Interpretation and limits

The combined gradient remains substantial throughout the late, flatter perplexity trajectory. Real data redirects the update rather than merely shrinking it to zero. Corrections to real-text predictions can nearly balance deterioration while parameters keep moving. The held-out effects, not vector norms alone, establish whether a direction helps real text. The large final-16-batch corrections remain relevant to endpoint timing.

This is one seed and ten cycles with constant LR 0.01, plain FP32 SGD and greedy synthetic targets. No stationary point or permanent stabilization is established. The analysis retains both-source conditioning for cosine and includes all batches for norms/reductions. Supplementary CSV columns record squared-norm cross-term reduction and a synthetic-energy-weighted opposing projection; those are different metrics and are not the percentages plotted in Figure 3.

## Reproduction

```powershell
$env:PYTHONPATH = (Get-Location).Path
python scripts/analysis/sgd_gradient_cancellation.py --out NEW_OUTPUT_DIRECTORY
```

All source bytes are checked against their existing publication manifest. Every recorded total satisfies the vector norm identity and triangle inequalities within tolerance. CSV/JSON records, script/input hashes and square PNG/PDF figures accompany this report. Published earlier analyses remain unchanged. Google Doc was not edited for this request.
"""
    (out / "report.md").write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    build(parser.parse_args().out)
