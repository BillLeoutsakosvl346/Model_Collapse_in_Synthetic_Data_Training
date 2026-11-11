"""Offline four-condition analysis; never trains or contacts a cloud service."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model_collapse.analysis.completion import _collect, comparison_contract
from model_collapse.common import read_json, sha256
from scripts.analysis.publication import new_destination

PROJECT = Path(__file__).resolve().parents[2]
LABELS = {"fresh_real": "Fresh real", "fresh_synthetic": "Fresh synthetic",
          "repeated_real": "Repeated real", "repeated_synthetic": "Repeated synthetic"}
COLORS = {"fresh_real": "#167d8d", "fresh_synthetic": "#bd3940",
          "repeated_real": "#d58a00", "repeated_synthetic": "#7651a8"}


def build(root, reference, out):
    root, reference = Path(root).resolve(), Path(reference).resolve()
    for source in (root, reference):
        if not read_json(source / "completion.json", {}).get("complete"):
            raise RuntimeError("Both comparisons must be complete")
    c = read_json(root / "config.json")
    if c.get("training_chunk_policy") != "repeat_fixed" or c.get("fixed_chunk") != 1:
        raise RuntimeError("New comparison must repeat the first chunk")
    previous = read_json(reference / "config.json")
    if previous.get("training_chunk_policy", "fresh_per_cycle") != "fresh_per_cycle" or previous.get("fixed_chunk") is not None:
        raise RuntimeError("Historical reference must use fresh-per-cycle chunks")
    contract = comparison_contract(root)
    if contract != comparison_contract(reference):
        raise RuntimeError("Historical controls differ in bank, hardware or scientific settings")
    rows, inputs, examples = [], {}, {}
    for source, prefix in ((reference, "fresh"), (root, "repeated")):
        actual, _, decoded = _collect(source)
        if len(actual) != 22:
            raise RuntimeError("Each completed comparison requires 22 checkpoint rows")
        for row in actual:
            row = dict(row)
            row["condition"] = prefix + "_" + row["arm"]
            if row["condition"] not in LABELS or row["global_step"] != row["cycle"] * 79:
                raise RuntimeError("Unexpected condition or optimizer trajectory")
            if row["synthetic_fraction"] != (0.0 if row["arm"] == "real" else 1.0):
                raise RuntimeError("Four-condition comparison requires real/synthetic endpoint arms")
            if row["cycle"] == 0 and row["weights_sha256"] != contract["baseline_weights_sha256"]:
                raise RuntimeError("Child baseline differs from the common pretrained tensors")
            if row["test_targets"] != 457664:
                raise RuntimeError("Primary test token count differs")
            rows.append(row)
        examples[prefix] = decoded
        for name in ("config.json", "completion.json", "children.json", "manifest.json", "environment.json",
                     "child_snapshots.json", "artifact_index.json", "recovery_verified.json", "seed_42/gen_00/metrics.json"):
            p = source / name
            if p.exists():
                inputs[p.relative_to(PROJECT).as_posix()] = sha256(p)
        for child in read_json(source / "children.json").values():
            folder = source.parent / child
            for stage in sorted(folder.glob("seed_42/gen_*")):
                for name in ("metrics.json", "training_history.json", "probe_examples.json", "training.json",
                             "training_input/manifest.json", "stage_completion.json"):
                    p = stage / name
                    if p.exists():
                        inputs[p.relative_to(PROJECT).as_posix()] = sha256(p)
            for name in ("config.json", "artifact_index.json", "recovery_snapshot.json", "backup_snapshot.json"):
                p = folder / name
                if p.exists():
                    inputs[p.relative_to(PROJECT).as_posix()] = sha256(p)
    for key in LABELS:
        values = [r for r in rows if r["condition"] == key]
        if sorted(r["cycle"] for r in values) != list(range(11)):
            raise RuntimeError("Missing/duplicate checkpoint in four-condition comparison")
    out = new_destination(root, out)
    out.mkdir(parents=True)
    (out / "metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (out / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / "examples.json").write_text(json.dumps(examples, indent=2) + "\n", encoding="utf-8")
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12})
    figures = []
    def finish(fig, name, title):
        fig.tight_layout()
        for extension in ("png", "pdf"):
            metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
            fig.savefig(out / f"{name}.{extension}", dpi=200, metadata=metadata)
        plt.close(fig)
        figures.append({"name": name, "title": title})
    def trajectory(name, field, title, ylabel, keys=tuple(LABELS), percentage=False):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_box_aspect(1)
        for key in keys:
            values = sorted((r for r in rows if r["condition"] == key), key=lambda r:r["cycle"])
            multiplier = 100 if percentage else 1
            ax.plot([r["cycle"] for r in values], [multiplier*r[field] for r in values],
                    marker="s" if key.endswith("synthetic") else "o",
                    linestyle="--" if key.startswith("fresh") else "-",
                    color=COLORS[key], label=LABELS[key], linewidth=2)
        ax.set(xlabel="Epoch / training cycle (79 updates each)", ylabel=ylabel, title=title, xticks=range(11))
        ax.grid(alpha=.22)
        ax.legend(fontsize=10)
        finish(fig, name, title)
    trajectory("01_test_perplexity", "test_perplexity", "Fixed real-test continuation perplexity", "Corpus perplexity")
    trajectory("02_validation_perplexity", "validation_perplexity", "Fixed real-validation continuation perplexity", "Corpus perplexity")
    trajectory("03_real_only_perplexity", "test_perplexity", "Real targets: fresh versus repeated data", "Test perplexity",
               ("fresh_real", "repeated_real"))
    trajectory("04_pooled_entropy", "token_entropy_nats", "Token diversity across the fixed probe", "Pooled token entropy (nats)")
    trajectory("05_within_entropy", "mean_within_entropy_nats", "Token diversity inside individual outputs", "Mean within-output entropy (nats)")
    trajectory("06_distinct2", "distinct_2", "Unique two-token phrases across the probe", "Pooled distinct-2 (%)", percentage=True)
    trajectory("07_distinct3", "distinct_3", "Unique three-token phrases across the probe", "Pooled distinct-3 (%)", percentage=True)
    trajectory("08_repeated4", "mean_within_repeated_4gram_rate", "Repetition inside individual outputs", "Mean within-output repeated 4-grams (%)", percentage=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_box_aspect(1)
    # Only the real arm trains on this real anchor; its gap has the usual
    # train-versus-held-out interpretation. Both arms' anchor values remain in CSV.
    for key in ("repeated_real",):
        values = sorted((r for r in rows if r["condition"] == key), key=lambda r:r["cycle"])
        for field, style, description in (("fixed_real_anchor_loss", "--", "fixed real chunk 1"), ("test_loss", "-", "held-out test")):
            ax.plot([r["cycle"] for r in values], [r[field] for r in values], style, color=COLORS[key], marker="o",
                    label=LABELS[key] + ": " + description)
    ax.set(xlabel="Epoch", ylabel="Conditional NLL (nats / target)", title="Repeated real: training and held-out conditional loss", xticks=range(11))
    ax.grid(alpha=.22)
    ax.legend(fontsize=9)
    finish(fig, "09_real_anchor_and_test", "Real first-chunk versus held-out loss")
    trajectory("10_real_anchor_gap", "real_anchor_generalization_gap", "Held-out minus real first-chunk loss", "NLL difference (nats / target)",
               ("repeated_real",))
    trajectory("14_fixed_real_answers", "fixed_real_anchor_loss", "Original real answers on the repeated prompts", "Conditional NLL on real chunk-1 targets (nats)",
               ("repeated_real", "repeated_synthetic"))
    corpora = c["secondary_corpora"]
    final = {r["condition"]:r for r in rows if r["cycle"] == 10}
    baseline = {r["condition"]:r for r in rows if r["cycle"] == 0}
    contrasts = {"scope": "Descriptive cycle-10 differences; one seed per condition, no significance test", "metrics": {}}
    for field in ("test_loss", "test_perplexity", "token_entropy_nats", "mean_within_entropy_nats", "distinct_2", "distinct_3", "mean_within_repeated_4gram_rate"):
        real_reuse = final["repeated_real"][field] - final["fresh_real"][field]
        synthetic_reuse = final["repeated_synthetic"][field] - final["fresh_synthetic"][field]
        contrasts["metrics"][field] = {"reuse_effect_with_real_targets": real_reuse,
            "reuse_effect_with_synthetic_targets": synthetic_reuse,
            "difference_of_reuse_effects": synthetic_reuse - real_reuse,
            "synthetic_effect_with_fresh_prompts": final["fresh_synthetic"][field] - final["fresh_real"][field],
            "synthetic_effect_with_repeated_prompts": final["repeated_synthetic"][field] - final["repeated_real"][field]}
    (out / "endpoint_contrasts.json").write_text(json.dumps(contrasts, indent=2) + "\n", encoding="utf-8")
    matrix = np.array([[final[key][f"secondary_{name}_perplexity"] / baseline[key][f"secondary_{name}_perplexity"] for name in corpora] for key in LABELS])
    fig, ax = plt.subplots(figsize=(8, 8))
    heat = ax.imshow(np.log2(matrix), cmap="RdBu_r", vmin=-1, vmax=max(1, float(np.log2(matrix).max())), aspect="auto")
    for i in range(4):
        for j in range(len(corpora)):
            ax.text(j, i, f"{matrix[i,j]:.2f}×", ha="center", va="center", fontsize=12,
                    color="white" if abs(np.log2(matrix[i,j])) > 1.4 else "black")
    ax.set(xticks=range(len(corpora)), xticklabels=[x.replace("_", "\n") for x in corpora],
           yticks=range(4), yticklabels=list(LABELS.values()), title="Cycle-10 full-text perplexity / pretrained baseline")
    fig.colorbar(heat, ax=ax, label="log₂ perplexity ratio", shrink=.65)
    finish(fig, "11_secondary_corpora", "Secondary full-text corpus evaluation")
    trajectory("12_learning_rate", "learning_rate_end", "Preserved global cosine learning-rate schedule", "Learning rate")
    trajectory("13_test_nll_p95", "test_nll_p95", "Harder held-out passages: 95th-percentile NLL", "P95 example NLL (nats / target)")
    (out / "figure_manifest.json").write_text(json.dumps(figures, indent=2) + "\n", encoding="utf-8")
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    except subprocess.CalledProcessError:
        revision = None
    record = {"inputs": inputs, "comparison_contract": contract, "script_sha256": sha256(__file__),
              "analysis_revision": revision, "python": platform.python_version(), "numpy": np.__version__,
              "matplotlib": matplotlib.__version__, "scope": "44 saved checkpoint rows; no new training or inference",
              "training_environments": {p.name:read_json(p / "environment.json", {}) for p in (root, reference)},
              "outputs": {p.name:sha256(p) for p in out.iterdir() if p.is_file()}}
    (out / "analysis_manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out":str(out), "rows":len(rows), "final":final}, indent=2))
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--reference", default="runs/20261106T193918Z-full-completion64")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    build(a.root, a.reference, a.out)
