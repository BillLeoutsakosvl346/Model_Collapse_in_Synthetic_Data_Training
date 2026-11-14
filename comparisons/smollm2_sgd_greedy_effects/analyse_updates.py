"""Reproduce cycle-level SGD source effects from pinned, sealed update journals.

Offline analysis only: no model loading, inference, provisioning or source edits.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
from pathlib import Path

import numpy as np

from model_collapse.analysis.sgd_effects import _collect_updates
from model_collapse.common import atomic_json, read_json, sha256

RUN = "20261114T001352Z-full-completion64-mixtures"
ARMS = ("real", "mixed99", "synthetic")
COLORS = {"real": "#087F8C", "synthetic": "#BD365E", "mixed99": "#7651AF"}
FLOOR = 1e-6


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def interval(values, counts, documents, repetitions=2000):
    """Token-weighted mean and paired document-bootstrap interval.

    values is one aggregate NLL-sum change per probe example. Resampling is
    over documents, never updates or training seeds.
    """
    docs = np.unique(documents)
    numerator = np.array([values[documents == d].sum() for d in docs])
    denominator = np.array([counts[documents == d].sum() for d in docs])
    sampled = np.random.default_rng(42).integers(0, len(docs), (repetitions, len(docs)))
    draws = numerator[sampled].sum(1) / denominator[sampled].sum(1)
    return float(values.sum() / counts.sum()), np.quantile(draws, [.025, .975]).tolist()


def sign_counts(values):
    a = np.asarray(values)
    return {"helpful": int((a < -FLOOR).sum()), "harmful": int((a > FLOOR).sum()),
            "unresolved": int((np.abs(a) <= FLOOR).sum())}


def analyse(workspace, out):
    workspace, out = Path(workspace), Path(out)
    if out.exists():
        raise FileExistsError("Choose a new output directory; published evidence is immutable")
    recovery = read_json(workspace / "update_recovery.json")
    # Recheck every selectively restored file against its immutable source receipt.
    source_hashes = {}
    for name, source in recovery["sources"].items():
        source_root = workspace / (RUN if name == "parent" else RUN + "-" + name)
        for relative, info in source["files"].items():
            actual = sha256(source_root / relative)
            if actual != info["sha256"]:
                raise RuntimeError(f"Recovered evidence changed: {name}/{relative}")
            source_hashes[name + "/" + relative] = actual
    root = workspace / RUN
    roots = {a: workspace / (RUN + "-" + a) for a in ARMS}
    rows, layers, _, verified = _collect_updates(root, roots)
    if len(rows) != 2370:
        raise RuntimeError("Expected all 2,370 sealed updates")
    cycles, tails, trajectory, bootstrap, checks = [], [], [], {}, {}
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        if [r["step"] for r in arm_rows] != list(range(1, 791)):
            raise RuntimeError("Missing or unordered updates")
        continuity = max(abs(a["mixed_probe_nll"] - b["no_update_probe_nll"])
                         for a, b in zip(arm_rows, arm_rows[1:]))
        if continuity > 1e-12:
            raise RuntimeError("Probe scores do not telescope across updates/cycles")
        checks[arm] = {"updates": len(arm_rows), "probe_continuity_max_error": continuity,
                      "reconstruction_max_relative_error": max(r["gradient_reconstruction_relative_error"] for r in arm_rows),
                      "actual_total_probe_nll_change": sum(r["mixed_delta_nll"] for r in arm_rows),
                      "real_effect_sign_counts": sign_counts([r["real_conditional_delta_nll"] for r in arm_rows]),
                      "synthetic_effect_sign_counts": sign_counts([r["synthetic_conditional_delta_nll"] for r in arm_rows])}
        for cycle in range(1, 11):
            stage = roots[arm] / f"seed_42/gen_{cycle:02d}"
            seal = read_json(stage / "diagnostics_manifest.json")
            records = [json.loads(line) for line in (stage / seal["records_path"]).read_text(encoding="utf-8").splitlines()]
            values = [r for r in arm_rows if r["cycle"] == cycle]
            with np.load(stage / seal["scores_path"], allow_pickle=False) as arrays:
                arrays = {k: np.array(arrays[k], copy=True) for k in arrays.files}
            counts, docs = arrays["counts"], arrays["documents"]
            # These symmetric local allocations split the nonlinear interaction
            # equally. They add exactly, but are not alternative training paths.
            real_phi = .5 * (arrays["real"] - arrays["no_update"] + arrays["mixed"] - arrays["synthetic"])
            syn_phi = .5 * (arrays["synthetic"] - arrays["no_update"] + arrays["mixed"] - arrays["real"])
            if not np.allclose(real_phi + syn_phi, arrays["mixed"] - arrays["no_update"], atol=1e-12, rtol=0):
                raise RuntimeError("Local effect allocation does not add to the actual change")
            effect_arrays = {
                "real_conditional": arrays["mixed"] - arrays["synthetic"],
                "synthetic_conditional": arrays["mixed"] - arrays["real"],
                "real_isolated": arrays["real"] - arrays["no_update"],
                "synthetic_isolated": arrays["synthetic"] - arrays["no_update"],
                "actual": arrays["mixed"] - arrays["no_update"],
                "real_symmetric_allocation": real_phi, "synthetic_symmetric_allocation": syn_phi,
            }
            real_norm = np.array([r["real_gradient_norm"] for r in values])
            syn_norm = np.array([r["synthetic_gradient_norm"] for r in values])
            present = real_norm > 0
            both = (real_norm > 0) & (syn_norm > 0)
            ratios = real_norm[both] / syn_norm[both]
            cosines = [r["source_cosine"] for r in values if r["source_cosine"] is not None]
            row = {"arm": arm, "cycle": cycle, "updates": len(values),
                   "real_present_updates": int(present.sum()),
                   "real_pairs": sum(r["real_pairs"] for r in values),
                   "synthetic_pairs": sum(r["synthetic_pairs"] for r in values),
                   "real_norm_mean_all_updates": float(real_norm.mean()),
                   "synthetic_norm_mean_all_updates": float(syn_norm.mean()),
                   "ratio_median_when_both_present": float(np.median(ratios)) if len(ratios) else None,
                   "ratio_q25_when_both_present": float(np.quantile(ratios, .25)) if len(ratios) else None,
                   "ratio_q75_when_both_present": float(np.quantile(ratios, .75)) if len(ratios) else None,
                   "cosine_median_when_both_present": float(np.median(cosines)) if cosines else None,
                   "cosine_q25_when_both_present": float(np.quantile(cosines, .25)) if cosines else None,
                   "cosine_q75_when_both_present": float(np.quantile(cosines, .75)) if cosines else None,
                   "probe_ppl_before_cycle": values[0]["no_update_probe_perplexity"],
                   "probe_ppl_after_cycle": values[-1]["mixed_probe_perplexity"]}
            for name, effect in effect_arrays.items():
                total, ci = interval(effect.sum(0), counts, docs)
                row[name + "_sum_nll"] = total
                row[name + "_mean_nll"] = total / len(values)
                bootstrap[f"{arm}/cycle_{cycle}/{name}"] = {"sum_delta_nll": total,
                    "sum_document_bootstrap_95": ci,
                    "mean_document_bootstrap_95": [v / len(values) for v in ci]}
            for source in ("real", "synthetic"):
                row.update({source + "_conditional_" + k: v for k, v in
                    sign_counts([r[source + "_conditional_delta_nll"] for r in values]).items()})
                token_count = sum(r[source + "_scored_targets"] for r in values)
                row[source + "_optimization_mean_nll"] = (sum(
                    r[source + "_source_mean_nll"] * r[source + "_scored_targets"]
                    for r in values if r[source + "_scored_targets"]
                ) / token_count if token_count else None)
            if abs(row["actual_sum_nll"] - math.log(row["probe_ppl_after_cycle"] / row["probe_ppl_before_cycle"])) > 1e-12:
                raise RuntimeError("Cycle NLL change fails endpoint check")
            cycles.append(row)
            final = values[-1]
            tails.append({"arm": arm, "cycle": cycle, "step": final["step"],
                "batch_pairs": final["real_pairs"] + final["synthetic_pairs"],
                "real_pairs": final["real_pairs"], "synthetic_pairs": final["synthetic_pairs"],
                "first78_actual_delta_nll": sum(r["mixed_delta_nll"] for r in values[:-1]),
                "last_actual_delta_nll": final["mixed_delta_nll"],
                "last_real_conditional_delta_nll": final["real_conditional_delta_nll"],
                "last_synthetic_conditional_delta_nll": final["synthetic_conditional_delta_nll"],
                **{a + "_ppl": final[a + "_probe_perplexity"] for a in ("no_update", "real", "synthetic", "mixed")}})
            for metric_cycle in ([0] if cycle == 1 else []) + [cycle]:
                metric = read_json(roots[arm] / f"seed_42/gen_{metric_cycle:02d}/metrics.json")
                trajectory.append({"arm": arm, "cycle": metric_cycle, "test_nll": metric["test"]["loss"],
                                   "test_perplexity": metric["test"]["perplexity"]})
    out.mkdir(parents=True)
    write_csv(out / "per_update.csv", rows)
    write_csv(out / "cycle_summary.csv", cycles)
    layer_rows = []
    for arm in ARMS:
        for cycle in range(1, 11):
            for layer in sorted({r["layer"] for r in layers}):
                group = [r for r in layers if r["arm"] == arm and r["cycle"] == cycle and r["layer"] == layer]
                both = [r for r in group if min(r["real_norm"], r["synthetic_norm"]) > 0]
                layer_rows.append({"arm": arm, "cycle": cycle, "layer": layer,
                    "real_norm_mean": float(np.mean([r["real_norm"] for r in group])),
                    "synthetic_norm_mean": float(np.mean([r["synthetic_norm"] for r in group])),
                    "median_real_synthetic_ratio_when_both": float(np.median([r["real_norm"] / r["synthetic_norm"] for r in both])) if both else None,
                    "median_cosine_when_both": float(np.median([r["real_synthetic_cosine"] for r in both])) if both else None})
    write_csv(out / "layer_cycle_summary.csv", layer_rows)
    write_csv(out / "tail_batches.csv", tails)
    write_csv(out / "test_trajectory.csv", trajectory)
    atomic_json(out / "bootstrap_intervals.json", bootstrap)
    atomic_json(out / "verification.json", checks)
    atomic_json(out / "source_recovery.json", recovery)
    draw(out, rows, cycles, tails, trajectory, bootstrap)
    write_report(out, rows, cycles, tails, checks)
    atomic_json(out / "analysis_manifest.json", {
        "run_id": RUN, "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "analysis_script_sha256": sha256(__file__), "recovery_script_sha256": sha256(Path(__file__).with_name("recover_updates.py")),
        "python": platform.python_version(), "numpy": np.__version__,
        "source_files": source_hashes, "diagnostic_inputs": verified,
        "inference_or_training": False,
        "scope": "2,370 sealed logical update records and per-example probe arrays; no bulk model/optimizer restore.",
        "uncertainty": "2,000 paired document-bootstrap draws, seed 42; evaluation sampling only, not seed replication.",
        "output_files": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()},
    })
    print(json.dumps({"out": str(out), "updates": len(rows), "checks": checks}, indent=2))


def draw(out, rows, cycles, tails, trajectory, bootstrap):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12,
                         "legend.fontsize": 10, "savefig.facecolor": "white"})
    mixed = [r for r in cycles if r["arm"] == "mixed99"]
    x = np.arange(1, 11)

    def figure(title, ylabel, xlabel="Training cycle"):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_box_aspect(1)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=.18)
        return fig, ax

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / (name + ".png"), dpi=180)
        fig.savefig(out / (name + ".pdf"), metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)

    fig, ax = figure("99% synthetic: how each source helps or hurts\nLocal effects on the same 128 held-out real examples",
                     "Mean immediate NLL change per update")
    for source, label in (("real", "Real component (1% of epoch examples)"),
                          ("synthetic", "Synthetic component (99% of epoch examples)")):
        ys = [r[source + "_conditional_mean_nll"] for r in mixed]
        cis = [bootstrap[f"mixed99/cycle_{c}/{source}_conditional"]["mean_document_bootstrap_95"] for c in x]
        ax.plot(x, ys, "o-", color=COLORS[source], label=label, lw=2)
        ax.fill_between(x, [a[0] for a in cis], [a[1] for a in cis], color=COLORS[source], alpha=.13)
    ax.axhline(0, color="#334155", lw=1)
    ax.text(.04, .97, "Above zero: worse predictions", va="top", transform=ax.transAxes, color=COLORS["synthetic"])
    ax.text(.04, .03, "Below zero: better predictions", transform=ax.transAxes, color=COLORS["real"])
    ax.set_xticks(x)
    ax.legend(loc="center left", bbox_to_anchor=(.01, .56), fontsize=9)
    fig.text(.5, .01, "Conditional effects; averaged over all 79 updates/cycle. Bands: document-bootstrap 95% intervals.", ha="center", fontsize=8)
    fig.subplots_adjust(bottom=.12)
    save(fig, "99_source_help_hurt")

    fig, ax = figure("99% synthetic: actual weighted gradient strengths\nSource proportions are already included", "Mean gradient L2 norm per update")
    for source, label in (("real", "Weighted real gradient"), ("synthetic", "Weighted synthetic gradient")):
        ax.plot(x, [r[source + "_norm_mean_all_updates"] for r in mixed], "o-", color=COLORS[source], label=label, lw=2)
    ax.set_xticks(x); ax.set_ylim(bottom=0); ax.legend()
    fig.text(.5, .015, "All updates included, including zero real contributions. Norm size alone does not indicate benefit.", ha="center", fontsize=8)
    save(fig, "99_weighted_gradient_norms")

    fig, ax = figure("99% synthetic: real and synthetic directions oppose\nBatches containing both sources only", "Cosine alignment of source gradients")
    y = [r["cosine_median_when_both_present"] for r in mixed]
    ax.plot(x, y, "o-", color=COLORS["mixed99"], lw=2, label="Median alignment")
    ax.fill_between(x, [r["cosine_q25_when_both_present"] for r in mixed],
                    [r["cosine_q75_when_both_present"] for r in mixed], color=COLORS["mixed99"], alpha=.15, label="Middle 50% of batches")
    ax.axhline(0, color="#334155", lw=1); ax.set(ylim=(-1, 1), xticks=x); ax.legend()
    save(fig, "99_gradient_alignment")

    fig, ax = figure("99% synthetic: sharp corrections at small final batches\nActual per-update held-out probe perplexity", "128-pair probe perplexity", "Training cycle (fractional = progress within cycle)")
    rr = [r for r in rows if r["arm"] == "mixed99"]
    ax.plot([0] + [r["step"] / 79 for r in rr], [rr[0]["no_update_probe_perplexity"]] + [r["mixed_probe_perplexity"] for r in rr],
            color=COLORS["mixed99"], lw=1, label="Actual 99% training path")
    for c in x:
        ax.axvline(c, color="#64748b", alpha=.12, lw=.7)
    mt = [r for r in tails if r["arm"] == "mixed99" and r["real_pairs"]]
    ax.scatter([r["cycle"] for r in mt], [r["mixed_ppl"] for r in mt], color=COLORS["real"], s=55,
               zorder=4, label="Final batch: 1 real + 15 synthetic")
    for t in mt:
        if t["cycle"] not in (6, 10):
            continue
        c = t["cycle"]
        ax.scatter(c, t["synthetic_ppl"], color=COLORS["synthetic"], marker="x", s=70, linewidths=2, zorder=5,
                   label="Same final update, omit real component" if c == 6 else None)
        ax.annotate("", xy=(c, t["mixed_ppl"]), xytext=(c, t["synthetic_ppl"]),
                    arrowprops={"arrowstyle": "->", "color": COLORS["real"], "lw": 1.7})
        ax.annotate(f"{t['synthetic_ppl']:.0f} → {t['mixed_ppl']:.0f}", (c, (t["synthetic_ppl"] + t["mixed_ppl"]) / 2),
                    xytext=(-73 if c == 10 else 6, 0), textcoords="offset points", fontsize=10,
                    bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none", "pad": 2})
    ax.set(xticks=range(0, 11), xlim=(0, 10.35)); ax.legend(loc="upper left", fontsize=9)
    fig.text(.5, .015, "Counterfactual crosses change this update only; they are not alternative full trajectories or full-test PPL.", ha="center", fontsize=8)
    save(fig, "99_tail_batch_corrections")

    fig, ax = figure("99% synthetic: accumulated local effects by cycle\nSymmetric allocation of nonlinear interactions", "Sum of local NLL effects within cycle")
    for name, color, label in (("real", COLORS["real"], "Real allocation"), ("synthetic", COLORS["synthetic"], "Synthetic allocation")):
        ax.plot(x, [r[name + "_symmetric_allocation_sum_nll"] for r in mixed], "o-", color=color, lw=2, label=label)
    ax.plot(x, [r["actual_sum_nll"] for r in mixed], "s--", color="#334155", label="Actual net probe-NLL change")
    ax.axhline(0, color="#334155", lw=.8); ax.set_xticks(x); ax.legend(loc="center left")
    fig.text(.5, .015, "Real + synthetic allocations = net change. Local attribution along the observed path, not alternate training.", ha="center", fontsize=8)
    save(fig, "99_additive_local_allocation")

    fig, ax = figure("Full fixed-test evaluation: three SGD/greedy conditions", "Corpus perplexity (log scale)")
    for arm, label in (("real", "0% synthetic"), ("mixed99", "99% synthetic"), ("synthetic", "100% synthetic")):
        data = [r for r in trajectory if r["arm"] == arm]
        ax.plot([r["cycle"] for r in data], [r["test_perplexity"] for r in data], "o-", color=COLORS[arm], label=label, lw=2)
    ax.set(yscale="log", xticks=range(11)); ax.legend()
    save(fig, "full_test_perplexity")


def write_report(out, rows, cycles, tails, checks):
    mixed = [r for r in cycles if r["arm"] == "mixed99"]
    mt = {r["cycle"]: r for r in tails if r["arm"] == "mixed99"}
    real_allocation = sum(r["real_symmetric_allocation_sum_nll"] for r in mixed)
    syn_allocation = sum(r["synthetic_symmetric_allocation_sum_nll"] for r in mixed)
    text = """# Per-update real/synthetic effects under SGD and greedy decoding

Completed run `20261114T001352Z-full-completion64-mixtures`; offline analysis on 14 November 2026. No new inference, training or paid resources. The selected diagnostic journals, per-example scores and seals were recovered from immutable private backups and checked against their indexed hashes. This does not restore full model/optimizer checkpoints.

## Main findings

- In the 99% condition, all **563 updates containing real examples** have a beneficial real conditional effect on the fixed real-validation probe; **227 updates have no real examples**. Synthetic conditional effects worsen this probe in **778/790 updates**, improve it in 12, and are never numerically unresolved at the declared 1e-6 NLL floor.
- The real correction strengthens as the model deteriorates. Mean weighted real gradient norm rises from **0.100** in cycle 1 to **0.297** in cycle 10; the synthetic norm falls from **0.877** to **0.485**. These include the actual batch proportions and real-absent zero contributions.
- Among batches containing both sources, median real/synthetic weighted-norm ratio rises from **0.165** to **0.703**. Median source cosine becomes more negative (**-0.111 to -0.363**): the sources increasingly oppose, although they are far from exactly opposite. Norms and alignment alone do not establish benefit; the actual held-out interventions do.
- The final 16-example batch is important. In cycles 6 and 10 it contains one real example, giving that example weight **1/16**, versus **1/128** in a full batch. Recorded counterfactuals show large immediate corrections. Dataset composition remains exactly 100 real / 9,900 synthetic per cycle; this is a batch-weighting and endpoint-timing effect, not a changed dataset ratio.
- The full-test 99% trajectory still deteriorates strongly: **17.958 → 164.413**, versus **1,408.607** at 100% synthetic and **17.726** at 0%. The small real component restrains deterioration; these data do not establish stable or good performance.

## What was measured

Each condition starts independently from pretrained SmolLM2-135M, processes ten fresh 10,000-pair FineWeb chunks, and uses one terminal epoch per chunk. The 64 real prompt tokens remain visible; loss scores only 64 continuation tokens. The 99% run replaces 9,900 targets with its own preceding checkpoint's greedy completions. Plain FP32 SGD uses constant **0.01**, no momentum, weight decay, clipping or scheduler. Microbatch 32 / effective batch 128 gives 78 full updates and one 16-example update each cycle.

At each actual update, real and synthetic gradients were measured directly from the same training forward graphs and normalized by **all scored tokens in that effective batch**. The authoritative combined gradient trained the model. Four scratch evaluations used exactly the same initial weights: unchanged (L0), real component only (LR), synthetic component only (LS), and actual mixed update (LM). All use the same 128 real-validation pairs / 128 documents / 8,192 continuation targets. No held-out gradients train the model.

- **Real conditional effect = LM - LS:** how predictions change when the real component is included alongside the existing synthetic component.
- **Synthetic conditional effect = LM - LR:** analogous inclusion effect of synthetic data.
- Negative NLL change means better probabilities for unseen real continuations; positive means worse. These are conditional predictive effects, not a universal text-quality score.
- All component magnitudes retain their original batch weights. A real-only shadow in the 99% arm does not renormalize one real example into a whole real batch.
- Conditional effects are nonlinear and do not generally add. Their interaction is `LM - LR - LS + L0`.

![Immediate effects](99_source_help_hurt.png)

Figure 1. Mean local conditional effects over all 79 actual updates per cycle, including zero real contributions. Bands resample the same probe documents (2,000 paired bootstrap draws, seed 42). They describe evaluation-sample variability, not training-seed uncertainty or independent update replicates.

## Numerical cycle summary

Table 1. Conditional NLL effects per update and actual probe movement. All values concern the 128-pair probe; full-test perplexity is reported separately.

| Cycle | Updates with real / 79 | Mean real effect | Mean synthetic effect | Probe PPL before → after |
| ---: | ---: | ---: | ---: | ---: |
"""
    for r in mixed:
        text += f"| {r['cycle']} | {r['real_present_updates']} | {r['real_conditional_mean_nll']:+.6f} | {r['synthetic_conditional_mean_nll']:+.6f} | {r['probe_ppl_before_cycle']:.2f} → {r['probe_ppl_after_cycle']:.2f} |\n"
    text += """
![Weighted gradient norms](99_weighted_gradient_norms.png)

Figure 2. Arithmetic mean L2 norms, already weighted by the actual source share in each batch. A component's size says how strongly it changes parameters, not whether that change helps. Later real targets are much more corrective even though they remain just 1% of examples.

![Direction](99_gradient_alignment.png)

Figure 3. Median cosine and the middle 50% of actual both-source batches. Empty real sources are undefined and excluded. Negative cosine means the two parameter-update directions oppose; it does not on its own identify which improves validation.

## Why the late-cycle dips are sharp

The final batch contains one real example in cycles 1, 6 and 10, and none in the other cycles. SGD averages within each batch, so each example in the 16-row tail has eight times the weight of an example in a 128-row update. This conventional tail handling was retained from the original protocol. It affects all conditions' tail weights, but the source imbalance matters particularly when one corrective real example lands there.

Table 2. Direct final-update counterfactuals at cycles 6 and 10. The synthetic-only shadow omits the real component without renormalizing synthetic weights. These are probe values, not counterfactual full-test evaluations.

| Cycle | Real / total in tail | PPL before last update | After synthetic component only | After actual combined update | Net NLL change in preceding 78 updates | Net NLL change in last update |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    for c in (6, 10):
        t = mt[c]
        text += f"| {c} | {t['real_pairs']} / 16 | {t['no_update_ppl']:.2f} | {t['synthetic_ppl']:.2f} | {t['mixed_ppl']:.2f} | {t['first78_actual_delta_nll']:+.6f} | {t['last_actual_delta_nll']:+.6f} |\n"
    text += """
![Tail corrections](99_tail_batch_corrections.png)

Figure 4. Per-update probe trajectory, with measured one-update synthetic-only shadows at the two large drops. Cycle 6's first 78 updates worsen probe loss overall, followed by one larger improvement at the tail. Cycle 10's first 78 already slightly improve it, with most of the cycle's improvement again occurring at the tail. The full-test endpoint dips at the same cycles, making this a strong mechanistic explanation for their timing; the full test was not evaluated before/after each shadow.

## Additive local accounting, with a stated convention

To summarize net effects exactly, split each update's nonlinear interaction equally:

`real allocation = 0.5 × [(LR - L0) + (LM - LS)]`

`synthetic allocation = 0.5 × [(LS - L0) + (LM - LR)]`

These two allocations sum to `LM - L0`. This symmetric average-marginal convention is one attribution rule, not a unique causal decomposition. Summing over the actual path is meaningful because consecutive saved probe states agree exactly. It does not simulate a trajectory without one source.

"""
    text += f"Across 790 updates, the real allocation is **{real_allocation:+.6f} NLL**, synthetic **{syn_allocation:+.6f}**, and net **{real_allocation + syn_allocation:+.6f}**. Under this convention the real component offsets about **{-100 * real_allocation / syn_allocation:.1f}%** of the accumulated local synthetic harm. The large opposing totals describe repeated local corrections along an evolving model, rather than achievable isolated multi-epoch endpoints.\n\n"
    text += """![Additive effects](99_additive_local_allocation.png)

Figure 5. Local allocations per cycle plus their exact net. Late cycles approach a moving balance, with sharp tail corrections; this does not prove permanent stabilization.

![Full test comparison](full_test_perplexity.png)

Figure 6. Full held-out real-test evaluation, distinct from the miniature intervention probe. All three arms use the same constant-LR SGD recipe and greedy probes.

## Interpretation and limits

Real targets remain a corrective signal when greedy synthetic targets increasingly reinforce the model's own preferred predictions. As predictions drift away from real continuations, those real targets can create large gradients despite their small count. In the 99% arm, token-weighted real training-target NLL grows from **3.215** in cycle 1 to **5.333** in cycle 10, while synthetic-target NLL falls from **0.508** to **0.060**. These average pre-update training losses concern fresh chunks and changing synthetic targets, not a fixed evaluation set. They support growing disagreement with real text and increasing confidence on the model's own completions.

The measured growth in weighted real norm, negative source alignment and consistently beneficial real interventions supports the corrective-signal explanation under this recipe. The synthetic influence is usually harmful on the already-degraded 99% path; this is not a claim that every synthetic update in every condition harms real text. In the separate 100% path, 254/790 individual updates improve the probe, although its epoch endpoints worsen dramatically.

The decreasing learning rate and Adam momentum are absent here, so they cannot explain these particular corrections. Fresh real-only training stays near baseline; its individual batch effects are small and have both signs. This is consistent with small random movements near an already capable real-text model, while corrective real examples have a much larger effect after synthetic deterioration.

This is one seed, one corpus, one model, greedy fixed-length generation, constant LR 0.01 selected using a short real-only screen, and 128 probe documents. The probe's repeated evaluations are correlated. No multiple-seed or learning-rate sensitivity experiment is performed. Bootstrap intervals do not quantify training randomness. The calibration receipt predates full execution; the probe was not used to select production checkpoints or tune source updates. Very small effects below 1e-6 NLL are classified unresolved; absent sources are exact zero.

All 2,370 journal records pass source-count/row-order audits, component reconstruction, exact mixed-shadow equality, RNG/mode preservation, seal checks and contiguous step checks. Probe scores telescope exactly across all stages. The maximum mixed-source relative gradient-reconstruction error is below 2.65e-7. Only indexed diagnostic evidence and small metadata were restored in this analysis; full model/optimizer restoration remains a separate recovery obligation. No historical bytes or cloud resources were changed.

## Reproduction and records

From the repository root (private source access is required only for recovery):

```powershell
$env:PYTHONPATH = (Get-Location).Path
python comparisons/smollm2_sgd_greedy_effects/recover_updates.py
python comparisons/smollm2_sgd_greedy_effects/analyse_updates.py --out NEW_OUTPUT_DIRECTORY
```

`per_update.csv` retains 2,370 flattened records; `cycle_summary.csv` has all 30 cycle summaries; `tail_batches.csv` identifies all 30 tails. `bootstrap_intervals.json`, `verification.json`, `source_recovery.json` and `analysis_manifest.json` record uncertainty, immutable source revisions, file hashes, analysis version/environment and recovery scope. Per-example arrays and full journals remain in ignored recovery storage/private backups.
"""
    (out / "report.md").write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=".state/sgd-update-analysis")
    parser.add_argument("--out", default=f"reports/{RUN}/generated/update-gradients-20261114")
    args = parser.parse_args()
    analyse(args.workspace, args.out)
