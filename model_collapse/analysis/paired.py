"""Offline paired reports; execution remains in workflow/paired.py."""
from pathlib import Path

from ..common import atomic_json, read_json, sha256
from ..workflow.selection import verify_selection
from ..workflow.paired import child_roots


def report_pair(root, out=None, for_backup=False):
    import csv
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..analysis.reporting import Evidence, _analysis_identity, _output_directory, _rows, report as arm_report
    root = Path(root)
    out = _output_directory(root, out, root.name, for_backup=for_backup)
    out.mkdir(parents=True, exist_ok=True)
    rows, inputs = [], {}
    for arm, child in child_roots(root).items():
        evidence = Evidence(child)
        child_config = evidence.read(child / "config.json")
        seed_root = child / f"seed_{child_config['seed']}"
        metrics = [evidence.read(p) for p in sorted(seed_root.glob("gen_*/metrics.json"))]
        if not metrics:
            continue
        for m in metrics:
            if m["generation"]:
                history = evidence.read(seed_root / f"gen_{m['generation']:02d}/training_history.json")
                verify_selection(m["training"], history, child_config)
        arm_rows, imports, refs = _rows(metrics, seed_root, evidence, child_config)
        for row in arm_rows:
            row["arm"] = arm
            reference = evidence.read(seed_root / f"gen_{row['generation']:02d}/reference_sequences.json")
            if reference and reference.get("sequence_mean_nll"):
                import numpy as np
                row["frozen_s0_block_nll_p95"] = float(np.percentile(reference["sequence_mean_nll"], 95))
            rows.append(row)
        inputs.update({f"{child.name}/{rel}": meta for rel, meta in evidence.files.items()})
        # New derived files live under the parent output; sealed child artifacts remain untouched.
        arm_report(child, out / arm)
    if not rows:
        raise RuntimeError("No paired metrics to report")
    roots = child_roots(root)
    for g in (0, 1):
        rel = f"seed_42/gen_{g:02d}/metrics.json"
        if all((child / rel).exists() for child in roots.values()):
            if sha256(roots["continued"] / rel) != sha256(roots["reset"] / rel):
                raise RuntimeError("Cannot report mismatched shared-stage measurements")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    keys = ("test_perplexity", "validation_perplexity", "distinct_2", "distinct_3", "repeated_4gram_rate", "token_entropy_nats")
    for ax, key in zip(axes.flat, keys):
        for arm in ("continued", "reset"):
            chosen = [r for r in rows if r["arm"] == arm]
            ax.plot([r["generation"] for r in chosen], [r.get(key, math.nan) for r in chosen], marker="o", label=arm + " (S0/S1 shared)")
        ax.set_title(key.replace("_", " "))
        ax.set_xlabel("Synthetic training cycle")
        ax.grid(alpha=.2)
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)
    text = ("# Matched final-epoch comparison\n\n"
            "S0 and the newly trained S1 are shared. Every synthetic stage executes and retains epoch 5; validation is diagnostic. "
            "Continued weights carry five updates per cycle; reset always initializes from S0.\n\n"
            "![Paired metrics](comparison.png)\n\n"
            "Complete metric tables and existing graph families: [continued](continued/report.md), [reset](reset/report.md). "
            "The CSV includes corpus NLL/PPL, diversity, duplicates, entropy, repetition, frozen-S0 scores, token counts, selected epochs and timing.\n\n"
            "Smoke results validate the pipeline and do not establish model collapse. Full results use one seed and are descriptive. "
            "Token entropy measures pooled token frequencies; it is not predictive entropy or within-output repetition.\n")
    (out / "report.md").write_text(text, encoding="utf-8")
    atomic_json(out / "analysis_manifest.json", {"inputs": inputs, "analysis": _analysis_identity(),
                "pair_config_sha256": sha256(root / "config.json"),
                "outputs": {p.relative_to(out).as_posix(): sha256(p) for p in out.rglob("*")
                            if p.is_file() and p != out / "analysis_manifest.json"}})
    return {"report": str(out / "report.md"), "metrics_rows": len(rows)}
