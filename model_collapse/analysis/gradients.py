"""Offline reporting of checkpoint-gradient diagnostics."""
from __future__ import annotations

import csv
import json
import platform
from pathlib import Path

from ..common import atomic_json, read_json, sha256


def report(root, out=None, *, for_backup=False):
    from .reporting import _output_directory, _analysis_identity
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    if read_json(root / "config.json", {}).get("phase") == "layer_norms":
        out = _output_directory(root, out, root.name, for_backup=for_backup)
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for path in sorted((root / "layers").glob("*.json")):
            item = read_json(path)
            rows.extend({"arm":item["arm"],"cycle":item["cycle"],"parameter":name,
                         "real_norm":item["real_parameter_norms"][name],"synthetic_norm":value}
                        for name,value in item["synthetic_geometry"]["parameters"].items())
        with (out/"per_parameter_norms.csv").open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=["arm","cycle","parameter","real_norm","synthetic_norm"])
            writer.writeheader();writer.writerows(rows)
        (out/"report.md").write_text("# Synthetic per-parameter gradient norms\n\nRead-only supplement to the primary diagnostic. Each repeated global synthetic norm and loss must match the original measurement.\n\nSee per_parameter_norms.csv.\n",encoding="utf-8")
        return {"report":str(out/"report.md"),"parameter_rows":len(rows)}
    out = _output_directory(root, out, root.name, for_backup=for_backup)
    out.mkdir(parents=True, exist_ok=True)
    gradients, updates, inputs = [], [], {}
    for path in sorted((root / "cases").glob("*/gradients.json")):
        gradients.append(read_json(path)); inputs[path.relative_to(root).as_posix()] = sha256(path)
        other = path.with_name("interventions.json")
        if other.exists():
            updates.extend(read_json(other)); inputs[other.relative_to(root).as_posix()] = sha256(other)
    if not gradients:
        raise RuntimeError("No gradient measurements available")
    for name in ("config.json", "resources.json", "source_recovery.json", "probe_manifest.json", "acceptance.json", "environment.json"):
        if (root / name).exists():
            inputs[name] = sha256(root / name)
    atomic_json(out / "gradients.json", gradients)
    atomic_json(out / "updates.json", updates)
    flat = [{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in updates]
    if flat:
        with (out / "updates.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    figures = []
    def finish(fig, name):
        fig.tight_layout(); fig.savefig(out / (name + ".png"), dpi=180)
        fig.savefig(out / (name + ".pdf"), metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig); figures.append(name + ".png")
    mixed = sorted((r for r in gradients if r["arm"] == "mixed99"), key=lambda r: r["cycle"])
    if mixed:
        fig, ax = plt.subplots(figsize=(7.5, 7.5)); ax.set_box_aspect(1)
        for key, label, color in [("weighted_real_norm", "1% × real mean gradient", "#147d81"),
                                  ("weighted_synthetic_norm", "99% × synthetic mean gradient", "#ad355c"),
                                  ("mixed_norm", "Resulting mixed gradient", "#475569")]:
            ax.plot([r["cycle"] for r in mixed], [r[key] for r in mixed], "o-", label=label, color=color)
        ax.set(xlabel="Before training cycle", ylabel="Gradient L2 norm", title="Actual 99% mixture: weighted gradient strengths")
        ax.grid(alpha=.2); ax.legend(); finish(fig, "01_weighted_gradient_strength")
    fig, ax = plt.subplots(figsize=(7.5, 7.5)); ax.set_box_aspect(1)
    for arm, label in [("mixed99", "99%: identical-prompt real vs synthetic"), ("synthetic", "100%: identical-prompt real vs synthetic")]:
        rows = sorted((r for r in gradients if r["arm"] == arm), key=lambda r: r["cycle"])
        ax.plot([r["cycle"] for r in rows], [r["matched_real_vs_synthetic"]["cosine"] for r in rows], "o-", label=label)
    ax.axhline(0, color="gray", lw=1); ax.set(ylim=(-1, 1), xlabel="Before training cycle", ylabel="Cosine alignment",
                                             title="Do real and synthetic targets request opposing changes?")
    ax.grid(alpha=.2); ax.legend(fontsize=9); finish(fig, "02_gradient_alignment")
    colors = {"optimizer_history": "#718096", "real_only": "#147d81", "synthetic_only": "#ad355c", "mixed": "#8b5fbf"}
    for arm, label in [("mixed99", "99% synthetic checkpoint"), ("synthetic", "100% synthetic checkpoint")]:
        for rate in ("historical", "common"):
            fig, ax = plt.subplots(figsize=(7.5, 7.5)); ax.set_box_aspect(1)
            for action, color in colors.items():
                rows = sorted((r for r in updates if r["arm"] == arm and r["learning_rate_policy"] == rate and r["action"] == action), key=lambda r:r["cycle"])
                if not rows: continue
                ax.plot([r["cycle"] for r in rows], [r["delta_nll"] for r in rows], "o-", color=color, label=action.replace("_", " "))
                ax.vlines([r["cycle"] for r in rows], [r["delta_nll_document_bootstrap_95"][0] for r in rows],
                          [r["delta_nll_document_bootstrap_95"][1] for r in rows], color=color, alpha=.4)
            ax.axhline(0, color="black", lw=1); ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
            ax.set(xlabel="Before training cycle", ylabel="Change in real-validation NLL (nats / target)",
                   title=f"{label}: {rate} learning rate\nOne aggregate-gradient AdamW update")
            ax.grid(alpha=.2); ax.legend(fontsize=9); finish(fig, f"03_{arm}_{rate}_update_effects")
    lines = ["# SmolLM2 checkpoint-gradient diagnostic", "",
             "Dropout-disabled FP32 gradients and isolated complete-source-average AdamW updates. These are not historical minibatch replays.", "",
             "## Measurements", "", "| Arm | Before cycle | Real mean norm | Synthetic mean norm | Same-prompt cosine | Mixture reconstruction error |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in gradients:
        cos = row["matched_real_vs_synthetic"]["cosine"]
        lines.append(f"| {row['arm']} | {row['cycle']} | {row['real_vs_synthetic']['norm']:.6g} | {row['real_vs_synthetic']['other_norm']:.6g} | {cos if cos is not None else 'undefined'} | {row['mixture_relative_error']} |")
    lines += ["", "## Interpretation and limits", "",
              "- Negative validation-loss changes mean improved predictions on the fixed real validation probe; positive changes mean deterioration.",
              "- Gradient norm measures requested movement before AdamW; it does not establish beneficial direction or a percentage of causal influence.",
              "- The optimizer-history control supplies explicit zero gradients, preserving AdamW momentum and weight-decay effects.",
              "- Update records also compare each intervention directly with the optimizer-history control, isolating its incremental data-gradient effect under the same prior state.",
              "- Real-only gradients use the 100 actual real rows for the 99% case, and counterfactual original real targets on all rows for the 100% case. Same-prompt comparisons are recorded separately.",
              "- Source-mean gradient norms depend on sample size and cancellation; the actual 100-versus-9,900-row decomposition and equal-row matched comparison answer different questions.",
              "- Both recorded historical learning rates and a common 3e-6 rate are measured. No held-out gradient updates weights.",
              "- Intervals use paired validation-document resampling; checkpoints and conditions are not independent training-seed replicates.",
              "- This measures immediate local effects, not long-term equilibrium, irreversible collapse or permanent stabilization.", ""]
    for name in figures:
        lines += [f"![{name}]({name})", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    atomic_json(out / "analysis_manifest.json", {"inputs": inputs, "analysis": _analysis_identity(),
                "environment": {"python": platform.python_version(), "matplotlib": matplotlib.__version__},
                "outputs": {p.name: sha256(p) for p in out.iterdir() if p.is_file()}})
    return {"report": str(out / "report.md"), "cases": len(gradients), "updates": len(updates), "figures": figures}
