"""Offline, source-sealed reports of actual-update SGD interventions."""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import numpy as np

from ..common import atomic_json, read_json, sha256


def source_summary(rows):
    """Separate scalar-loss shares, vector geometry and local held-out effects."""
    result = []
    for arm, cycle in sorted({(r["arm"], r["cycle"]) for r in rows}):
        epoch = [r for r in rows if r["arm"] == arm and r["cycle"] == cycle]
        for population in ("all_updates", "full_batches", "tail_batches"):
            values = [r for r in epoch if population == "all_updates" or
                      (r["real_pairs"] + r["synthetic_pairs"] == 128) == (population == "full_batches")]
            if not values:
                continue
            real = math.fsum(r["real_weighted_training_nll"] for r in values)
            synthetic = math.fsum(r["synthetic_weighted_training_nll"] for r in values)
            tokens = math.fsum(r["total_scored_targets"] for r in values)
            real_sum = math.fsum(r["real_weighted_training_nll"] * r["total_scored_targets"] for r in values)
            synthetic_sum = math.fsum(r["synthetic_weighted_training_nll"] * r["total_scored_targets"] for r in values)
            orthogonal = np.mean([math.hypot(r["real_gradient_norm"], r["synthetic_gradient_norm"]) for r in values])
            total = np.mean([r["total_gradient_norm"] for r in values])
            cosines = [r["source_cosine"] for r in values if r["source_cosine"] is not None]
            result.append({"arm": arm, "cycle": cycle, "population": population, "updates": len(values),
                "real_pairs": sum(r["real_pairs"] for r in values),
                "synthetic_pairs": sum(r["synthetic_pairs"] for r in values),
                "mean_weighted_real_nll": real / len(values), "mean_weighted_synthetic_nll": synthetic / len(values),
                "real_share_of_mean_batch_loss": real / (real + synthetic) if real + synthetic else None,
                "token_weighted_real_nll": real_sum / tokens,
                "token_weighted_synthetic_nll": synthetic_sum / tokens,
                "real_share_of_corpus_loss": real_sum / (real_sum + synthetic_sum) if real_sum + synthetic_sum else None,
                "mean_real_gradient_norm": float(np.mean([r["real_gradient_norm"] for r in values])),
                "mean_synthetic_gradient_norm": float(np.mean([r["synthetic_gradient_norm"] for r in values])),
                "mean_combined_gradient_norm": float(total), "mean_orthogonal_reference_norm": float(orthogonal),
                "cancellation_vs_orthogonal_reference": float(1 - total / orthogonal) if orthogonal else None,
                "mean_source_cosine": float(np.mean(cosines)) if cosines else None,
                "opposing_updates": sum(v < 0 for v in cosines), "both_source_updates": len(cosines),
                **{key: float(np.mean([r[key] for r in values])) for key in
                   ("real_conditional_delta_nll", "synthetic_conditional_delta_nll", "mixed_delta_nll", "interaction_nll")}})
    return result


def effect_summary(scores, records):
    """Document-bootstrap uncertainty of mean local effects along one epoch.

    Documents, not optimizer updates, are resampled. These means describe local
    interventions along the observed path; they are not alternative trajectories.
    """
    from ..science.gradients import paired_change
    counts = np.asarray(scores["counts"]); documents = np.asarray(scores["documents"])
    comparisons = {"real_conditional": ("synthetic", "mixed"),
                   "synthetic_conditional": ("real", "mixed"),
                   "actual_mixed": ("no_update", "mixed")}
    result = {"updates": len(records), "probe_pairs": len(counts), "probe_documents": len(np.unique(documents)),
              "real_present_updates": sum(r["real_pairs"] > 0 for r in records),
              "real_pairs": sum(r["real_pairs"] for r in records),
              "synthetic_pairs": sum(r["synthetic_pairs"] for r in records)}
    for name, (before, after) in comparisons.items():
        value = paired_change(np.asarray(scores[before]).mean(0), np.asarray(scores[after]).mean(0), counts, documents)
        result[name + "_mean_delta_nll"] = value["delta_nll"]
        result[name + "_document_bootstrap_95"] = value["delta_nll_document_bootstrap_95"]
    present = np.asarray([r["real_pairs"] > 0 for r in records])
    result["real_conditional_mean_when_present"] = None
    if present.any():
        result["real_conditional_mean_when_present"] = float(
            (np.asarray(scores["mixed"])[present] - np.asarray(scores["synthetic"])[present]).sum()
            / (int(counts.sum()) * int(present.sum())))
    return result


def _collect_updates(root, roots=None):
    from ..workflow.completion import child_roots
    from ..science.sgd_effects import verify_seal, ACTIONS
    root = Path(root); c = read_json(root / "config.json")
    roots = child_roots(root, c) if roots is None else roots
    rows, layers, summaries, inputs = [], [], [], {}
    expected_probe_identity = None
    expected_counts = expected_documents = None
    for arm, child in roots.items():
        for cycle in range(1, c["generations"] + 1):
            stage = Path(child) / f"seed_42/gen_{cycle:02d}"
            if not (stage / "stage_completion.json").exists():
                continue
            training = read_json(stage / "training.json")
            seal = verify_seal(stage, expected_weights=training["selected_weights_sha256"])
            if expected_probe_identity is None:
                expected_probe_identity = seal["probe_identity"]
            elif seal["probe_identity"] != expected_probe_identity:
                raise RuntimeError("SGD update probes differ across completed stages")
            records_path, scores_path = stage / seal["records_path"], stage / seal["scores_path"]
            records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line]
            for path in (stage / "diagnostics_manifest.json", stage / "training.json", records_path, scores_path):
                inputs[arm + "/" + path.relative_to(child).as_posix()] = sha256(path)
            with np.load(scores_path, allow_pickle=False) as arrays:
                score_arrays = {key: arrays[key] for key in (*ACTIONS, "counts", "documents")}
                if expected_counts is None:
                    expected_counts = np.array(arrays["counts"], copy=True)
                    expected_documents = np.array(arrays["documents"], copy=True)
                elif not (np.array_equal(arrays["counts"], expected_counts)
                          and np.array_equal(arrays["documents"], expected_documents)):
                    raise RuntimeError("SGD probe targets/documents changed across completed stages")
                summaries.append({"arm": arm, "cycle": cycle, **effect_summary(score_arrays, records)})
            for record in records:
                g = record["gradient"]
                item = {"arm": arm, "cycle": cycle, "step": record["step"],
                        "attempt_id": seal["attempt_id"], "probe_identity": seal["probe_identity"],
                        "learning_rate": record["learning_rate"], "real_pairs": record["real_pairs"],
                        "synthetic_pairs": record["synthetic_pairs"],
                        "real_scored_targets": record["real_scored_targets"],
                        "synthetic_scored_targets": record["synthetic_scored_targets"],
                        "total_scored_targets": record["total_scored_targets"],
                        "real_weighted_training_nll": record["real_weighted_training_nll"],
                        "synthetic_weighted_training_nll": record["synthetic_weighted_training_nll"],
                        "real_gradient_norm": g["real_norm"], "synthetic_gradient_norm": g["synthetic_norm"],
                        "total_gradient_norm": g["total_norm"], "source_cosine": g["real_synthetic_cosine"],
                        "gradient_reconstruction_relative_error": g["reconstruction_relative_error"],
                        "update_l2_norm": record["update_l2_norm"],
                        "real_source_mean_nll": record["real_source_mean_nll"],
                        "synthetic_source_mean_nll": record["synthetic_source_mean_nll"],
                        "unique_probe_evaluations": record["unique_probe_evaluations"],
                        "diagnostic_seconds": record.get("diagnostic_seconds"), **record["effects"]}
                for name, value in record["states"].items():
                    item[name + "_probe_nll"] = value["loss"]
                    item[name + "_probe_perplexity"] = value["perplexity"]
                item.update({"unresolved_" + name: value for name, value in record["numerically_unresolved"].items()})
                rows.append(item)
                layers.extend({"arm": arm, "cycle": cycle, "step": record["step"], "layer": name, **value}
                              for name, value in g["layers"].items())
    return rows, layers, summaries, inputs


def _csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def report(root, out=None, *, for_backup=False, reference=None):
    """Extend the standard completion report without mutating its source records."""
    from .completion import report as completion_report
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    base = completion_report(root, out, for_backup=for_backup, reference=reference)
    destination = Path(base["out"])
    c = read_json(root / "config.json")
    rows, layers, summaries, inputs = _collect_updates(root)
    if reference is not None:
        extra, extra_layers, extra_summaries, extra_inputs = _collect_updates(reference)
        if rows and extra and rows[0]["probe_identity"] != extra[0]["probe_identity"]:
            raise RuntimeError("Reference SGD diagnostic probe differs")
        if {r["arm"] for r in rows} & {r["arm"] for r in extra}:
            raise RuntimeError("Reference SGD arm identifiers overlap")
        rows += extra; layers += extra_layers; summaries += extra_summaries
        inputs.update({Path(reference).name + ":" + k: v for k, v in extra_inputs.items()})
    if not rows:
        raise RuntimeError("No sealed SGD update diagnostics are available")
    atomic_json(destination / "update_effects.json", rows)
    atomic_json(destination / "epoch_local_effects.json", summaries)
    _csv(destination / "update_effects.csv", rows)
    _csv(destination / "layer_gradients.csv", layers)
    _csv(destination / "epoch_local_effects.csv", summaries)
    sources = source_summary(rows)
    atomic_json(destination / "epoch_source_summary.json", sources)
    _csv(destination / "epoch_source_summary.csv", sources)
    colors = {"real": "#147d81", "synthetic": "#ad355c", "mixed": "#8b5fbf"}
    figures = []

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(destination / (name + ".png"), dpi=180)
        fig.savefig(destination / (name + ".pdf"), metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig); figures.append(name + ".png")

    for mixed_arm in dict.fromkeys(r["arm"] for r in rows if r["real_pairs"] and r["synthetic_pairs"]):
        mixed = sorted((r for r in rows if r["arm"] == mixed_arm), key=lambda r: r["step"])
        fraction = next(spec["synthetic_fraction"] for spec in c["completion_arms"] if spec["id"] == mixed_arm) if mixed_arm in {spec["id"] for spec in c["completion_arms"]} else .99
        condition_label = f"{100 * fraction:g}% condition"
        suffix = "" if mixed_arm == "mixed99" else "_" + mixed_arm
        fig, ax = plt.subplots(figsize=(7.5, 7.5)); ax.set_box_aspect(1)
        for prefix, label in (("real", "Weighted real component"), ("synthetic", "Weighted synthetic component"), ("total", "Actual combined gradient")):
            ax.plot([r["step"] for r in mixed], [r[prefix + "_gradient_norm"] for r in mixed],
                    color=colors.get(prefix, "#475569"), lw=1, label=label)
        ax.set(xlabel="Optimizer update", ylabel="Gradient L2 norm", title=f"{condition_label}: actual batch-weighted gradient strengths")
        ax.grid(alpha=.2); ax.legend(); finish(fig, "sgd_weighted_source_gradients" + suffix)
        fig, ax = plt.subplots(figsize=(7.5, 7.5)); ax.set_box_aspect(1)
        ax.plot([r["step"] for r in mixed], [np.nan if r["source_cosine"] is None else r["source_cosine"] for r in mixed],
                color=colors["mixed"], lw=1)
        ax.axhline(0, color="black", lw=1); ax.set(ylim=(-1, 1), xlabel="Optimizer update", ylabel="Real/synthetic gradient cosine",
                   title=f"{condition_label}: source-gradient alignment\nEmpty-source alignments are undefined")
        ax.grid(alpha=.2); finish(fig, "sgd_source_alignment" + suffix)
        fig, ax = plt.subplots(figsize=(7.5, 7.5)); ax.set_box_aspect(1)
        for key, label, color in (("real_conditional_delta_nll", "Including real: mixed minus synthetic-only", colors["real"]),
                                  ("synthetic_conditional_delta_nll", "Including synthetic: mixed minus real-only", colors["synthetic"])):
            ax.plot([r["step"] for r in mixed], [r[key] for r in mixed], lw=.9, color=color, label=label)
        ax.axhline(0, color="black", lw=1)
        ax.set(xlabel="Optimizer update", ylabel="Conditional real-validation NLL change",
               title=f"{condition_label}: immediate effect of each source\nBelow zero improves predictions on the fixed real probe")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3)); ax.grid(alpha=.2); ax.legend(fontsize=8)
        finish(fig, "sgd_conditional_source_effects" + suffix)
    for arm in dict.fromkeys(r["arm"] for r in rows):
        values = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["step"])
        fig, ax = plt.subplots(figsize=(7.5, 7.5)); ax.set_box_aspect(1)
        for prefix, label in (("real", "Weighted real component"), ("synthetic", "Weighted synthetic component"), ("mixed", "Actual combined update")):
            ax.plot([r["step"] for r in values], [r[prefix + "_only_delta_nll" if prefix != "mixed" else "mixed_delta_nll"] for r in values],
                    color=colors[prefix], lw=.9, label=label)
        ax.axhline(0, color="black", lw=1); ax.grid(alpha=.2)
        ax.set(xlabel="Optimizer update", ylabel="Change in real-validation NLL",
               title=f"{arm}: one-step source interventions\nAll shadows start from the same actual pre-update weights")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3)); ax.legend(fontsize=8)
        finish(fig, "sgd_" + arm + "_local_update_effects")
    for mixed_arm in dict.fromkeys(r["arm"] for r in rows if r["real_pairs"] and r["synthetic_pairs"]):
        mixed_layers = [r for r in layers if r["arm"] == mixed_arm]
        suffix = "" if mixed_arm == "mixed99" else "_" + mixed_arm
        def order(value):
            match = re.search(r"layers\.(\d+)", value)
            return (1, int(match[1])) if match else (0, value)
        names = sorted({r["layer"] for r in mixed_layers}, key=order)
        steps = sorted({r["step"] for r in mixed_layers}); matrix = np.full((len(names), len(steps)), np.nan)
        ni = {name: i for i, name in enumerate(names)}; si = {step: i for i, step in enumerate(steps)}
        for r in mixed_layers:
            if min(r["real_norm"], r["synthetic_norm"]) > 0:
                matrix[ni[r["layer"]], si[r["step"]]] = math.log10(r["real_norm"] / r["synthetic_norm"])
        fig, ax = plt.subplots(figsize=(7.5, 7.5)); ax.set_box_aspect(1)
        image = ax.imshow(matrix, aspect="auto", interpolation="nearest", origin="lower", cmap="coolwarm", vmin=-2, vmax=2,
                          extent=(steps[0] - .5, steps[-1] + .5, -.5, len(names) - .5))
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
        ax.set(xlabel="Optimizer update", ylabel="Parameter layer/group", title=f"{mixed_arm}: weighted real/synthetic gradient-norm ratio\nLog10 scale; empty-source cells are unmeasured")
        fig.colorbar(image, ax=ax, label="log10(real norm / synthetic norm)")
        finish(fig, "sgd_layer_source_ratio" + suffix)
    fig, ax = plt.subplots(figsize=(7.5, 7.5)); ax.set_box_aspect(1)
    for arm in dict.fromkeys(r["arm"] for r in summaries):
        values = sorted((r for r in summaries if r["arm"] == arm), key=lambda r: r["cycle"])
        x = [r["cycle"] for r in values]
        ax.plot(x, [r["real_conditional_mean_delta_nll"] for r in values], "o-", label=arm)
        ax.vlines(x, [r["real_conditional_document_bootstrap_95"][0] for r in values],
                  [r["real_conditional_document_bootstrap_95"][1] for r in values], alpha=.35)
    ax.axhline(0, color="black", lw=1); ax.grid(alpha=.2); ax.legend()
    ax.set(xlabel="Training cycle", ylabel="Mean local mixed-minus-synthetic NLL",
           title="Contribution of real data along the actual update path\nIntervals resample probe documents, not training updates")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3)); finish(fig, "sgd_epoch_real_effect")

    for arm in dict.fromkeys(r["arm"] for r in sources):
        values = [r for r in sources if r["arm"] == arm and r["population"] == "all_updates"]
        x = [r["cycle"] for r in values]
        fig, ax = plt.subplots(figsize=(7.5, 7.5)); ax.set_box_aspect(1)
        for key, label, color in (("mean_weighted_real_nll", "Real weighted loss", colors["real"]),
                                  ("mean_weighted_synthetic_nll", "Synthetic weighted loss", colors["synthetic"])):
            ax.plot(x, [r[key] for r in values], "o-", label=label, color=color)
        ax.set(xlabel="Training cycle", ylabel="Mean batch-weighted training NLL", title=f"{arm}: scalar training-loss contributions")
        ax.grid(alpha=.2); ax.legend(); finish(fig, "sgd_" + arm + "_source_losses")
        fig, ax = plt.subplots(figsize=(7.5, 7.5)); ax.set_box_aspect(1)
        for key, label, color in (("mean_real_gradient_norm", "Weighted real", colors["real"]),
                                  ("mean_synthetic_gradient_norm", "Weighted synthetic", colors["synthetic"]),
                                  ("mean_combined_gradient_norm", "Actual combined", "#475569"),
                                  ("mean_orthogonal_reference_norm", "Perpendicular reference", "#aaa")):
            ax.plot(x, [r[key] for r in values], "o-", label=label, color=color)
        ax.set(xlabel="Training cycle", ylabel="Mean gradient L2 norm", title=f"{arm}: actual gradients and perpendicular reference")
        ax.grid(alpha=.2); ax.legend(); finish(fig, "sgd_" + arm + "_cycle_gradient_geometry")
        fig, ax = plt.subplots(figsize=(7.5, 7.5)); ax.set_box_aspect(1)
        for population, style in (("all_updates", "o-"), ("full_batches", "s--"), ("tail_batches", "^:")):
            subset = [r for r in sources if r["arm"] == arm and r["population"] == population]
            ax.plot([r["cycle"] for r in subset], [100 * r["real_share_of_mean_batch_loss"] for r in subset],
                    style, label=population.replace("_", " "))
        ax.set(xlabel="Training cycle", ylabel="Real share of mean batch loss (%)", ylim=(0, 100),
               title=f"{arm}: real loss contribution\nRatio of source loss sums; not mean per-batch percentages")
        ax.grid(alpha=.2); ax.legend(); finish(fig, "sgd_" + arm + "_real_loss_share")
        fig, ax = plt.subplots(figsize=(7.5, 7.5)); ax.set_box_aspect(1)
        ax.plot(x, [100 * r["cancellation_vs_orthogonal_reference"] for r in values], "o-", color="#475569")
        ax.axhline(0, color="black", lw=1)
        ax.set(xlabel="Training cycle", ylabel="Reduction versus perpendicular reference (%)",
               title=f"{arm}: gradient cancellation\nPositive values indicate opposition, not held-out improvement")
        ax.grid(alpha=.2); finish(fig, "sgd_" + arm + "_cycle_cancellation")

    text = (destination / "report.md").read_text(encoding="utf-8")
    text += ("\n## Actual-update SGD diagnostics\n\n"
             "Training uses FP32 plain SGD at one calibrated constant rate, without momentum, weight decay, scheduler or gradient clipping. "
             "Generation and the fixed output-statistic probe use greedy decoding. Historical AdamW/five-beam results differ in several settings and are not optimizer-only controls.\n\n"
             "Real and synthetic gradients are directly measured from the same training forward graphs and normalized by the complete effective batch's scored-target count. "
             "Only the ordinary combined SGD update advances training. A reusable scratch model measures weighted-real, weighted-synthetic and combined counterfactuals on the fixed real-validation probe. "
             "The latter never supplies an update. Empty-source states share exact measurements with their no-op or combined counterparts.\n\n"
             "A negative mixed-minus-synthetic NLL means including that batch's real component improved real-probe predictions locally. "
             "Gradient norm alone does not establish quality. Separate source loss changes need not add: the interaction column records that nonlinearity. "
             "Small effects flagged numerically unresolved are retained.\n\n"
             "Epoch means and intervals describe local interventions along the actual observed training path. "
             "They are not trajectories produced by deleting a source for an entire epoch, and their sums do not establish long-term causal influence. "
             "Probe documents are resampled together; repeated optimizer updates are not independent seed replicates.\n\n"
             "| Arm | Cycle | Updates | Real-present updates | Mean real conditional NLL change | Document-bootstrap 95% interval |\n"
             "|---|---:|---:|---:|---:|---|\n")
    for row in summaries:
        interval = row["real_conditional_document_bootstrap_95"]
        text += f"| {row['arm']} | {row['cycle']} | {row['updates']} | {row['real_present_updates']} | {row['real_conditional_mean_delta_nll']:.8g} | [{interval[0]:.8g}, {interval[1]:.8g}] |\n"
    text += "\nOnly sealed completed attempts contribute to these tables; replayed abandoned attempts remain excluded.\n\n"
    for name in figures:
        text += f"![{name}]({name})\n\n"
    (destination / "report.md").write_text(text, encoding="utf-8")
    manifest = read_json(destination / "analysis_manifest.json")
    manifest["inputs"].update(inputs)
    manifest["sgd_analysis_script_sha256"] = sha256(__file__)
    manifest["source_summary_populations"] = ["all_updates", "full_batches", "tail_batches"]
    manifest["sgd_effect_scope"] = "actual_batch_weighted_local_sgd_interventions_128_pair_fp32_validation_probe"
    manifest["outputs"] = {p.name: sha256(p) for p in destination.iterdir() if p.is_file() and p.name != "analysis_manifest.json"}
    atomic_json(destination / "analysis_manifest.json", manifest)
    return {**base, "updates": len(rows), "effect_figures": figures}
