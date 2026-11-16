"""Compose the Week 5 whole-epoch source-ablation section from published evidence."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from model_collapse.common import PROJECT, atomic_json, read_json, sha256, verify_files


RUN = "20261116T195118Z-epoch-ablation"


def compose(out):
    out = Path(out)
    if out.exists():
        raise FileExistsError("Choose a new authored destination")
    source = PROJECT / "reports" / RUN / "generated/verified-20261116"
    mean = source.parent / "with-pointwise-mean-20261116"
    receipt_path = PROJECT / "docs/validation/2026-11-16_epoch-ablation-completion-and-analysis.json"
    receipt = read_json(receipt_path)
    for folder in (source, mean):
        verify_files(folder, read_json(folder / "analysis_manifest.json")["outputs"])
    read = lambda p: list(csv.DictReader(p.open(encoding="utf-8", newline="")))
    endpoints = {(int(r["cycle"]), r["source"]): r for r in read(source / "endpoints.csv")}
    summary = {(int(r["cycle"]), r["source"]): r for r in read(source / "branch_summary.csv")}
    averages = {int(r["cycle"]): r for r in read(mean / "pointwise_average.csv") if int(r["local_step"]) == 79}
    replay = {r["cycle"]: r for r in receipt["historical_replays"]}
    blocks = []
    def add(kind, text="", **kwargs): blocks.append({"type": kind, "text": text, **kwargs})
    def heading(text, page=False): add("h2", text, page=page)
    def bullets(*items):
        for text in items: add("bullet", text)
    def table(name, caption, rows, widths):
        add("caption", caption, before_table=True)
        add("table", id=name, rows=rows, widths=widths)
    add("h1", "Week 5 continued: Whole-epoch source ablations", page=True)
    add("body", "Do real targets continue to correct recursive synthetic training over a whole epoch? Earlier diagnostics removed one source for a single update. Here, independent model copies keep only real losses, only synthetic losses, or both throughout an entire historical epoch.")
    heading("1. Matched methodology")
    bullets(
        "Fork the completed 99%-synthetic SmolLM2-135M SGD run from M2, M5 and M9, immediately before cycles 3, 6 and 10. Each copy starts from the same weights and recorded training RNG for its case, then evolves independently.",
        "Restore the exact saved training inputs: 100 original real continuations and 9,900 synthetic continuations, paired with their original prompts. No synthetic targets are regenerated, and no new real examples are sampled.",
        "Keep SGD at constant learning rate 0.01, FP32 with TF32 off, seed 42, microbatch 32, and the historical order of 78 full 128-example updates plus a final 16-example update. There is no momentum, weight decay, scheduler or clipping.",
        "Inputs remain BOS + 64 prompt tokens + 64 continuation tokens; loss scores only the continuation. Both-source training keeps the whole loss. Real-only or synthetic-only removes the other terms while retaining each original batch denominator. Source-empty batches produce zero movement.",
        "Measure the same fixed 128-pair real-validation probe after every update, plus complete validation/test collections before and after each branch. Held-out gradients never train the model. Nine branches follow 711 scheduled update positions in total.")
    table("forks", "Table W5C.1. Historical fork points. Each row creates three independent one-epoch branches: both, real-only and synthetic-only.",
          [["Cycle", "Initializer", "Exact saved target collection"],
           ["3", "M2", "Cycle 3: 100 real + 9,900 synthetic"],
           ["6", "M5", "Cycle 6: 100 real + 9,900 synthetic"],
           ["10", "M9", "Cycle 10: 100 real + 9,900 synthetic"]], [45, 75, 320])
    heading("2. Endpoint results", page=True)
    bullets(
        "Lower perplexity means the model assigns higher probability to unseen real continuations. All three copies share the same starting score within a case.",
        "The figures use the 128-pair probe, scoring 8,192 continuation tokens after every update. Complete validation (7,447 pairs) and test (7,303 pairs) are separate, fixed epoch-end measurements. Both evaluations give the same ordering: real-only best, mixed intermediate, synthetic-only worst.")
    probe_rows = [["Cycle", "Shared start", "Both", "Real-only", "Synthetic-only"]]
    test_rows = [["Cycle", "Both", "Real-only", "Synthetic-only"]]
    for cycle in (3, 6, 10):
        probe_rows.append([str(cycle), f"{float(summary[cycle,'both']['start_probe_ppl']):.2f}"] +
                          [f"{float(endpoints[cycle,s]['probe_ppl_after']):.2f}" for s in ("both", "real", "synthetic")])
        test_rows.append([str(cycle)] + [f"{float(endpoints[cycle,s]['test_ppl']):.2f}" for s in ("both", "real", "synthetic")])
    table("probe", "Table W5C.2. Fixed-probe perplexity at the start and end of each one-epoch fork. These values match the plotted curves.", probe_rows, [45, 85, 85, 105, 120])
    table("test", "Table W5C.3. Complete held-out test perplexity after each branch, scoring 467,392 continuation tokens. These values are independent of the smaller diagnostic probe.", test_rows, [45, 115, 140, 140])
    for cycle, number, title, observation in (
        (3, 3, "Early deterioration is reduced by real targets", "At M2, both-source training still worsens probe perplexity overall, from 63.33 to 100.94. Removing the real correction raises it further to 117.13; keeping only the small real contribution lowers it to 37.97."),
        (6, 4, "The middle-epoch correction depends on real targets", "From M5, the real-only branch improves from 192.06 to 52.91, while synthetic-only rises to 299.77. The combined branch oscillates and ends at 148.38. Cycle 6 was deliberately chosen because the historical curve showed a correction."),
        (10, 5, "The corrective real signal remains at the late checkpoint", "From M9, real-only lowers probe perplexity from 221.43 to 56.26, while synthetic-only rises to 325.44. Both sources finish at 170.45. This is one extra epoch from M9, not a counterfactual replay of all ten cycles."),
    ):
        heading(f"{number}. Cycle {cycle}: {title}", page=True)
        bullets(observation,
                "Purple is the both-source replay; dashed grey retains real losses; dotted grey retains synthetic losses. Orange is the equal-weight arithmetic mean of the two grey scores at each matching update. The black star marks the original historical mixed endpoint.")
        add("image", path=str((mean / f"cycle_{cycle:02d}_probe_perplexity_with_average.png").resolve()))
        add("caption", f"Figure W5C.{ {3:1,6:2,10:3}[cycle] }. Cycle {cycle} from identical M{cycle-1} weights. Fractional x positions show progress within the epoch. Original source weights, prompts, targets, masks and update positions are preserved.")
    heading("6. What the orange average measures", page=True)
    add("body", "PPL_mean(t) = [PPL_real-only(t) + PPL_synthetic-only(t)] / 2")
    bullets(
        "This is a 50/50 average of the two plotted perplexities at the same update position, including their shared baseline. It averages the scores directly; averaging NLL first and then exponentiating would give a different quantity.",
        "The real-only and synthetic-only branches already retain their original source weighting. The orange curve is a visual reference, not an additional training condition, an averaged gradient or a mixed probability model.",
        "After the first update, the independent branches have different weights, so later gradients are evaluated at different locations. Neither their losses nor their perplexities must average to the purple trajectory. Above or below the orange curve does not by itself identify a gradient interaction.")
    rows = [["Cycle", "Orange mean", "Purple actual", "Actual - mean"]]
    for cycle in (3, 6, 10):
        r = averages[cycle]
        rows.append([str(cycle), f"{float(r['arithmetic_mean_ppl']):.2f}", f"{float(r['both_ppl']):.2f}", f"{float(r['both_minus_arithmetic_mean']):+.2f}"])
    table("average", "Table W5C.4. Pointwise-average comparison at each epoch endpoint, using fixed-probe perplexity. Purple ends above the mean in cycle 3 and below it in cycles 6 and 10.", rows, [45, 120, 120, 155])
    heading("7. Interpretation and limitations", page=True)
    bullets(
        "Measured source-removal effect: retaining only the small real contribution substantially improves unseen-real prediction at all three checkpoints. Removing it lets the saved synthetic targets drive further deterioration. Complete test endpoints confirm that this ordering is not confined to the 128-pair probe.",
        "Cycles 6 and 10 retain a sharp final mixed correction. Their final batch contains one real and 15 synthetic examples: 6.25% real in that update, versus 1% across the epoch. With the original denominator, each tail example has eight times the averaging weight of an example in a full batch. This contributes to endpoint sensitivity.",
        "The real-only curves also improve before the tail, and real-only batches with no real examples stay exactly flat. Real targets provide a continuing corrective signal; the correction is not confined to the last example.",
        "Whole-epoch ablations include feedback along diverging model trajectories. They cannot be added like the two gradient components measured at the same weights in the earlier one-update diagnostics. Three selected cases and one seed do not establish permanent stabilization, a universal safe ratio or a validated predictor.")
    heading("8. Replay fidelity and evidence")
    bullets(
        "Every both-source replay passed the predeclared 1e-5 tolerances before interpreting its ablations. Bitwise tensor equality was not achieved, but relative tensor errors stayed below 6.3e-9 and maximum per-update probe-NLL discrepancies below 1.75e-6.",
        "All 711 update records, source counts, original row schedules, denominators, empty-source movement and branch seals were checked. The analysis recovered 176 small files and regenerated CSVs identical to those saved remotely. This retrieval did not cold-restore all nine terminal checkpoints; recovery storage is retained.")
    rows = [["Cycle", "Relative tensor error", "Max probe NLL difference", "Max full-val/test NLL difference"]]
    for cycle in (3, 6, 10):
        r = replay[cycle]
        rows.append([str(cycle), f"{r['relative_l2']:.2e}", f"{r['maximum_probe_nll_discrepancy']:.2e}", f"{max(r['validation_nll_discrepancy'],r['test_nll_discrepancy']):.2e}"])
    table("replay", "Table W5C.5. Numerical discrepancies of both-source replays versus their historical epochs. Every listed discrepancy is below the 1e-5 acceptance limit.", rows, [45, 105, 140, 150])
    add("body", "Execution record: all nine branches completed on the matching RTX PRO 4000 Blackwell. Compute stopped normally on 16 November 2026 at 20:39 UTC, after 47.9 paid minutes including acceptance. Run: " + RUN + ".")
    out.mkdir(parents=True)
    atomic_json(out / "content.json", {"run_id": RUN, "blocks": blocks, "image_size_pt": 430})
    lines = []
    for b in blocks:
        if b["type"] in ("h1", "h2"):
            lines.append(("# " if b["type"] == "h1" else "## ") + b["text"])
        elif b["type"] == "bullet": lines.append("- " + b["text"])
        elif b["type"] == "table":
            lines += ["| " + " | ".join(b["rows"][0]) + " |", "| " + " | ".join("---" for _ in b["rows"][0]) + " |"]
            lines += ["| " + " | ".join(r) + " |" for r in b["rows"][1:]]
        elif b["type"] == "image":
            relative = Path(b["path"]).relative_to(mean)
            lines.append(f"![Branch comparison](../with-pointwise-mean-20261116/{relative.as_posix()})")
        else: lines.append(b["text"])
        lines.append("")
    (out / "section.md").write_text("\n".join(lines), encoding="utf-8")
    atomic_json(out / "authorship_manifest.json", {
        "inputs": {str(p.relative_to(PROJECT)): sha256(p) for p in [source / "endpoints.csv", source / "branch_summary.csv", mean / "pointwise_average.csv", receipt_path]},
        "script_sha256": sha256(__file__), "outputs": {p.name: sha256(p) for p in out.iterdir() if p.is_file()},
        "new_training_or_inference": False,
    })
    return {"out": str(out), "blocks": len(blocks), "tables": 5, "figures": 3}


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    print(json.dumps(compose(parser.parse_args().out), indent=2))
