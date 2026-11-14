"""Offline loss-term decomposition and a two-page Week 5 explanatory inset."""
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


def decompose(rows):
    """Ratio of mean weighted loss terms, not the mean of batch percentages."""
    real, synthetic, real_sums, synthetic_sums = [], [], [], []
    for row in rows:
        nr, ns = int(row["real_scored_targets"]), int(row["synthetic_scored_targets"])
        n = nr + ns
        lr = float(row["real_source_mean_nll"]) if nr else 0.
        ls = float(row["synthetic_source_mean_nll"]) if ns else 0.
        real.append(nr * lr / n)
        synthetic.append(ns * ls / n)
        real_sums.append(nr * lr)
        synthetic_sums.append(ns * ls)
    mean = lambda values: math.fsum(values) / len(values)
    r, s = mean(real), mean(synthetic)
    rn = mean([float(x["real_gradient_norm"]) for x in rows])
    sn = mean([float(x["synthetic_gradient_norm"]) for x in rows])
    return {"updates": len(rows), "mean_real_loss_term": r, "mean_synthetic_loss_term": s,
            "mean_total_loss": r + s, "real_loss_share_percent": 100 * r / (r + s),
            "synthetic_loss_share_percent": 100 * s / (r + s),
            "token_weighted_real_loss_share_percent": 100 * math.fsum(real_sums) / (math.fsum(real_sums) + math.fsum(synthetic_sums)),
            "mean_real_gradient_norm": rn, "mean_synthetic_gradient_norm": sn,
            "ratio_of_mean_gradient_norms_percent": 100 * rn / sn,
            "mean_real_conditional_delta_nll": mean([float(x["real_conditional_delta_nll"]) for x in rows]),
            "mean_synthetic_conditional_delta_nll": mean([float(x["synthetic_conditional_delta_nll"]) for x in rows])}


def build(out):
    out = Path(out)
    if out.exists():
        raise FileExistsError("Choose a new analysis destination")
    manifest = read_json(SOURCE / "publication_manifest.json")
    for name in ("per_update.csv", "cycle_summary.csv"):
        if sha256(SOURCE / name) != manifest["files"][name]["sha256"]:
            raise RuntimeError("Published input changed: " + name)
    rows = [r for r in csv.DictReader((SOURCE / "per_update.csv").open(encoding="utf-8")) if r["arm"] == "mixed99"]
    if [int(r["step"]) for r in rows] != list(range(1, 791)):
        raise RuntimeError("Expected all 790 ordered 99% updates")
    result = []
    for cycle in range(1, 11):
        values = [r for r in rows if int(r["cycle"]) == cycle]
        if len(values) != 79 or sum(int(r["real_pairs"]) for r in values) != 100 or sum(int(r["synthetic_pairs"]) for r in values) != 9900:
            raise RuntimeError("Source counts changed")
        if [int(r["real_pairs"]) + int(r["synthetic_pairs"]) for r in values] != [128] * 78 + [16]:
            raise RuntimeError("Actual batch sizes changed")
        for r in values:
            if any(int(r[source + "_scored_targets"]) != 64 * int(r[source + "_pairs"]) for source in ("real", "synthetic")):
                raise RuntimeError("Target weighting differs")
        result.append({"cycle": cycle, **decompose(values)})
    late = [r for r in rows if int(r["cycle"]) >= 6]
    summary = {"cycles": result, "cycles_6_to_10": decompose(late),
               "cycles_6_to_10_full128_only": decompose([r for r in late if int(r["real_pairs"]) + int(r["synthetic_pairs"]) == 128])}
    for r in result:
        if abs(r["real_loss_share_percent"] + r["synthetic_loss_share_percent"] - 100) > 1e-12:
            raise RuntimeError("Loss shares fail partition check")
    out.mkdir(parents=True)
    atomic_json(out / "summary.json", summary)
    with (out / "cycle_loss_shares.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0])); writer.writeheader(); writer.writerows(result)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(1, 11)
    real = np.array([r["real_loss_share_percent"] for r in result])
    synthetic = 100 - real
    fig, ax = plt.subplots(figsize=(8, 8)); ax.set_box_aspect(1)
    ax.bar(x, real, .72, color="#087F8C", label="Real loss contribution")
    ax.bar(x, synthetic, .72, bottom=real, color="#E6B3C3", label="Synthetic loss contribution")
    for xx, rr, ss in zip(x, real, synthetic):
        ax.text(xx, rr / 2, f"{rr:.1f}%", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        ax.text(xx, rr + ss / 2, f"{ss:.1f}%", ha="center", va="center", color="#492234", fontsize=10)
    ax.set(xlabel="Training cycle", ylabel="Share of mean batch training loss (%)", xticks=x, ylim=(0, 100), xlim=(.4, 10.6),
           title="99% synthetic run: who contributes the training loss?\nReal examples remain 1% in every cycle")
    ax.grid(axis="y", alpha=.18); ax.set_axisbelow(True)
    fig.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, .025), fontsize=11)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(out / "99_training_loss_shares.png", dpi=180)
    fig.savefig(out / "99_training_loss_shares.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)

    blocks = [
        {"type": "h2", "text": "5A. One percent of examples supplies a growing share of loss", "page": True},
        {"type": "bullet", "text": "The real targets remain exactly 1% of examples, but supply 6.5% of mean batch training loss in cycle 1, 24.0% in cycle 3, 43.3% in cycle 6 and 49.1% in cycle 10. Their loss share grows as synthetic targets become easier and real continuations become harder."},
        {"type": "bullet", "text": "Across cycles 6–10, the real share is 44.6%; restricting to full 128-example batches gives 44.0%. This is broader than the final-batch effect. Loss share describes the training objective; the separate gradient and held-out interventions establish the corrective effect."},
        {"type": "image", "path": str((out / "99_training_loss_shares.png").resolve())},
        {"type": "caption", "text": "Figure W5.1a. Contributions to training loss in the 99% arm. For each cycle, average the real and synthetic batch-weighted terms over all 79 updates, then divide each by their sum. Each bar totals 100%; it is not a percentage of examples or of the final gradient."},
        {"type": "h2", "text": "5B. From loss contributions to backpropagation", "page": True},
        {"type": "equation", "text": "L_B = (Σreal ℓᵢ)/N_B + (Σsynthetic ℓᵢ)/N_B = L_R + L_S"},
        {"type": "equation", "text": "g_R = ∇θ L_R;   g_S = ∇θ L_S;   Δθ = −0.01(g_R + g_S)"},
        {"type": "bullet", "text": "Each ℓᵢ averages that example's 64 continuation-token losses. N_B is the actual effective batch size: 128, or 16 at the tail. Microbatches of 32 accumulate into this same token-weighted objective. Real and synthetic terms retain this common denominator."},
        {"type": "bullet", "text": "Backpropagation differentiates the two loss terms. It does not use loss values directly as gradient magnitudes: 44.6% of loss does not mean 44.6% of the parameter update. Gradients are vectors whose directions can oppose."},
        {"type": "bullet", "text": "Early real influence is relatively weak and synthetic training drives deterioration. Later, in cycles 6–10, mean weighted real-gradient norm is 53.3% as large as synthetic. Including real improves held-out probe NLL by 0.0447/update on average; including synthetic worsens it by 0.0437. This direct evidence supports a moving near-balance; nonlinear effects and the small tail batch prevent a simple scalar cancellation or proof of permanent stability."},
        {"type": "caption", "text": "Table W5.4a. Actual mean batch loss terms, in nats per continuation token. Real share = mean(L_R) / [mean(L_R) + mean(L_S)]. The final column compares mean gradient norms, not shares of a scalar total; all proportions are already included."},
        {"type": "table", "id": "loss_shares", "widths": [45, 90, 105, 95, 115],
         "rows": [["Cycle", "Real loss term", "Synthetic loss term", "Real loss share", "Mean norm ratio R/S"],
                  *[[str(r["cycle"]), f"{r['mean_real_loss_term']:.4f}", f"{r['mean_synthetic_loss_term']:.4f}", f"{r['real_loss_share_percent']:.1f}%", f"{r['ratio_of_mean_gradient_norms_percent']:.1f}%"] for r in result]]},
    ]
    atomic_json(out / "blocks.json", {"run_id": RUN, "blocks": blocks})
    lines = []
    for b in blocks:
        if b["type"] == "h2": lines.append("## " + b["text"])
        elif b["type"] == "bullet": lines.append("- " + b["text"])
        elif b["type"] == "image": lines.append("![Training-loss shares](99_training_loss_shares.png)")
        elif b["type"] == "table":
            rr = b["rows"]; lines += ["| " + " | ".join(rr[0]) + " |", "| " + " | ".join(["---"] * len(rr[0])) + " |", *["| " + " | ".join(r) + " |" for r in rr[1:]]]
        else: lines.append(b["text"])
        lines.append("")
    (out / "section.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    atomic_json(out / "analysis_manifest.json", {"run_id": RUN, "script_sha256": sha256(__file__),
        "inputs": {str(SOURCE / n): sha256(SOURCE / n) for n in ("per_update.csv", "cycle_summary.csv", "publication_manifest.json")},
        "python": platform.python_version(), "numpy": np.__version__, "averaging": "Ratio of mean weighted batch loss terms; not mean of individual batch ratios.",
        "checks": ["Published input hashes", "790 ordered updates", "100 real/9900 synthetic per cycle", "78 full batches and one 16-example tail", "64 scored targets/example", "Loss shares partition 100%"],
        "training_or_inference": False, "output_hashes": {p.name: sha256(p) for p in out.iterdir() if p.is_file()}})
    print({"out": str(out), "late_real_loss_share_percent": summary["cycles_6_to_10"]["real_loss_share_percent"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    build(parser.parse_args().out)
