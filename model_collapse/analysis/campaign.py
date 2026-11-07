"""Offline campaign reports; execution remains in workflow/mixed_campaign.py."""
from pathlib import Path

from ..common import atomic_json, read_json, sha256
from ..workflow.selection import verify_selection
from ..workflow.mixed_campaign import job_roots, JOB_ORDER


def report_campaign(root, out=None, for_backup=False):
    """Report into a new parent destination without rewriting sealed child evidence."""
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
    for name, child in job_roots(root).items():
        if not (child / "config.json").is_file():
            continue
        evidence = Evidence(child)
        c = evidence.read(child / "config.json")
        seed_root = child / f"seed_{c['seed']}"
        metrics = [evidence.read(p) for p in sorted(seed_root.glob("gen_*/metrics.json"))]
        if not metrics:
            continue
        for m in metrics:
            if m["generation"]:
                verify_selection(m["training"], evidence.read(seed_root / f"gen_{m['generation']:02d}/training_history.json"), c)
        values, _, _ = _rows(metrics, seed_root, evidence, c)
        for row in values:
            row.update(job=name, synthetic_fraction=c["synthetic_fraction"], initialization=c["initialization"])
            rows.append(row)
        inputs.update({f"{child.name}/{rel}": meta for rel, meta in evidence.files.items()})
        arm_report(child, out / name)
    if not rows:
        raise RuntimeError("No campaign metrics to report")
    for rel in ("config.json", "manifest.json", "continued20_seal.json", "shared20_mixture.json", "children.json", "child_snapshots.json"):
        path = root / rel
        if path.is_file():
            inputs[f"{root.name}/{rel}"] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    for path in sorted((root / "job_seals").glob("*.json")):
        inputs[f"{root.name}/{path.relative_to(root).as_posix()}"] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, key in zip(axes.flat, ("test_perplexity", "validation_perplexity", "distinct_2", "distinct_3", "repeated_4gram_rate", "token_entropy_nats")):
        for name in JOB_ORDER:
            for origin, style in (("current_run", "o-"), ("imported_shared_stage", "s--")):
                selected = [row for row in rows if row["job"] == name and row.get("stage_origin") == origin]
                if not selected:
                    continue
                # Historical full-data prefix measurements are not a tiny smoke baseline.
                if read_json(root / "config.json")["kind"] == "smoke" and name == "reset50" and origin == "imported_shared_stage":
                    continue
                ax.plot([row["generation"] for row in selected], [row.get(key, math.nan) for row in selected], style,
                        label=name + (" (imported source)" if origin == "imported_shared_stage" else ""))
        ax.set_title(key.replace("_", " "))
        ax.set_xlabel("Synthetic training cycle")
        ax.grid(alpha=.2)
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)
    (out / "report.md").write_text(
        "# Mixed-fraction final-epoch campaign\n\n"
        "Three sequential logical jobs share one GPU session. Reset50 imports the historical 50/50 S0/S1/D2; "
        "continued20 and reset20 share their newly trained S1, generated D2 and exact cycle-2 mixture. "
        "Every new stage executes five epochs and retains epoch 5. Imported work is not new execution.\n\n"
        "Imported points retain their original evaluation scope and execution identity. In smoke reports, reset50's "
        "full-data S0/S1 points remain in the CSV but are omitted from the combined plot; the newly executed reset50 "
        "points use the small active evaluation split. They must not be read as one matched evaluation trajectory.\n\n"
        "![Campaign metrics](comparison.png)\n\n"
        "Per-job tables and graph families: [50/50 reset](reset50/report.md), "
        "[20/80 continued](continued20/report.md), [20/80 reset](reset20/report.md).\n\n"
        "Smoke trajectories validate the pipeline; full trajectories describe one seed. Pooled diversity, within-output "
        "repetition, formatting sensitivity and real-text prediction are distinct measurements.\n", encoding="utf-8")
    atomic_json(out / "analysis_manifest.json", {"inputs": inputs, "analysis": _analysis_identity(),
                "renderer": {"path": "model_collapse/analysis/campaign.py", "sha256": sha256(Path(__file__))},
                "source_runtime_sha256": read_json(root / "manifest.json", {}).get("code_sha256"),
                "campaign_config_sha256": sha256(root / "config.json"),
                "outputs": {p.relative_to(out).as_posix(): sha256(p) for p in out.rglob("*")
                            if p.is_file() and p != out / "analysis_manifest.json"}})
    return {"report": str(out / "report.md"), "metrics_rows": len(rows)}
