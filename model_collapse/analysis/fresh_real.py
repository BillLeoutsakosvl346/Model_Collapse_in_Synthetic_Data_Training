"""Offline reports for the token-budgeted FineWeb control."""
from __future__ import annotations

import csv
import importlib.metadata
import platform
from pathlib import Path

from ..common import atomic_json, read_json, sha256
from .reporting import _output_directory


def report(root, out=None, *, for_backup=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    c = read_json(root / "config.json")
    destination = _output_directory(root, out, root.name, for_backup=for_backup)
    destination.mkdir(parents=True, exist_ok=True)
    inputs = {"config.json": sha256(root / "config.json"), "data/manifest.json": sha256(root / "data/manifest.json")}
    rows = []
    for cycle in range(c["generations"] + 1):
        path = root / f"seed_{c['seed']}" / f"gen_{cycle:02d}" / "metrics.json"
        m = read_json(path)
        if m is None:
            continue
        inputs[path.relative_to(root).as_posix()] = sha256(path)
        history_path = path.parent / "training_history.json"
        history = read_json(history_path, [{}])[-1]
        if history_path.exists():
            inputs[history_path.relative_to(root).as_posix()] = sha256(history_path)
        row = {"cycle": cycle, "cumulative_source_tokens": cycle * c["source_tokens_per_cycle"],
               "validation_loss": m["validation"]["loss"], "validation_perplexity": m["validation"]["perplexity"],
               "test_loss": m["test"]["loss"], "test_perplexity": m["test"]["perplexity"],
               "training_loss": m.get("training_evaluation", {}).get("loss"),
               "optimization_loss": history.get("train_loss"), "learning_rate": history.get("learning_rate"),
               "learning_rate_end": history.get("learning_rate_end"), "global_step": history.get("global_step", 0),
               "train_seconds": history.get("train_seconds", 0),
               "source_tokens": m.get("data", {}).get("source_text_tokens", 0),
               "scored_training_tokens": m.get("data", {}).get("scored_tokens", 0),
               "training_documents": m.get("data", {}).get("documents", 0),
               "validation_scored_tokens": m["validation"]["scored_tokens"], "test_scored_tokens": m["test"]["scored_tokens"]}
        row["generalization_gap"] = row["validation_loss"] - row["training_loss"] if row["training_loss"] is not None else None
        rows.append(row)
    if not rows:
        raise RuntimeError("No FineWeb measurements available")
    with (destination / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(destination / "metrics.json", rows)
    baseline = rows[0]["validation_perplexity"]
    complete = len(rows) == c["generations"] + 1 and read_json(root / "completion.json", {}).get("complete") is True
    summary = {"run_id": root.name, "kind": c["kind"], "completed_cycles": len(rows) - 1,
               "complete": complete, "initial_validation_perplexity": baseline,
               "final_validation_perplexity": rows[-1]["validation_perplexity"],
               "maximum_validation_ppl_relative_change": max(row["validation_perplexity"] / baseline - 1 for row in rows),
               "stability_threshold": .05, "stability_observed": bool(complete and c["kind"] == "full" and
                   all(row["validation_perplexity"] <= baseline * 1.05 for row in rows)),
               "interpretation": "One seed, descriptive trajectory. Smoke validates the pipeline; scientific stability is assessed on the full pilot."}
    atomic_json(destination / "summary.json", summary)
    families = {
        "loss": [("training_loss", "Post-epoch training"), ("validation_loss", "Validation"), ("test_loss", "Test")],
        "perplexity": [("validation_perplexity", "Validation"), ("test_perplexity", "Test")],
        "generalization_gap": [("generalization_gap", "Validation minus training NLL")],
        "learning_rate": [("learning_rate", "Epoch start"), ("learning_rate_end", "Epoch end")],
        "runtime": [("train_seconds", "Training seconds")],
    }
    for name, series in families.items():
        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        ax.set_box_aspect(1)
        for key, label in series:
            selected = [row for row in rows if row.get(key) is not None]
            ax.plot([row["cycle"] for row in selected], [row[key] for row in selected], marker="o", label=label)
        ax.set_xlabel("Training cycle / cumulative epoch")
        ax.set_ylabel("NLL (nats per scored token)" if name == "loss" else name.replace("_", " ").capitalize())
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(destination / f"{name}.png", dpi=180)
        plt.close(fig)
    text = f"# FineWeb-Edu real-only {'pipeline validation' if c['kind'] == 'smoke' else 'pilot'}\n\n"
    text += (f"OPT-125M, seed {c['seed']}, one epoch per cycle, variable sequences up to {c['max_sequence_length']} tokens. "
             "Model, AdamW and global cosine scheduler continue across cycles. Data come from disjoint sampled documents.\n\n")
    text += f"Completed cycles {len(rows) - 1}. Initial validation PPL {baseline:.4f}. Latest validation PPL {rows[-1]['validation_perplexity']:.4f}.\n\n"
    text += "| Cycle | Source tokens | Validation PPL | Test PPL |\n|---|---|---|---|\n"
    for row in rows:
        text += f"| {row['cycle']} | {row['source_tokens']} | {row['validation_perplexity']:.4f} | {row['test_perplexity']:.4f} |\n"
    for name in families:
        text += f"\n![{name.replace('_', ' ')}]({name}.png)\n"
    text += "\nTraining corpora change each cycle. Post-epoch training NLL uses evaluation mode. Padding is excluded from all scored-token counts.\n"
    (destination / "report.md").write_text(text, encoding="utf-8")
    outputs = {p.name: sha256(p) for p in destination.iterdir() if p.is_file()}
    atomic_json(destination / "analysis_manifest.json", {"inputs": inputs, "outputs": outputs,
                "analysis_sha256": sha256(Path(__file__)), "python": platform.python_version(),
                "runtime_sha256": read_json(root / "manifest.json", {}).get("code_sha256"),
                "training_environment": read_json(root / "environment.json", {}),
                "packages": {name: importlib.metadata.version(name) for name in ("numpy", "matplotlib")}})
    return {"out": str(destination), "summary": summary}
