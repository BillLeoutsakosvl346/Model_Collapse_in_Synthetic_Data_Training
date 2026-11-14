"""Compose Week 5's authored Google Docs blocks from verified saved records."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from model_collapse.common import atomic_json, read_json

RUN = "20261114T001352Z-full-completion64-mixtures"
BASE = Path(f"reports/{RUN}/generated/update-gradients-20261114")


def compose(out):
    out = Path(out)
    if out.exists():
        raise FileExistsError("Choose a new authored output directory")
    out.mkdir(parents=True)
    cycles = list(csv.DictReader((BASE / "cycle_summary.csv").open(encoding="utf-8")))
    mixed = {int(r["cycle"]): r for r in cycles if r["arm"] == "mixed99"}
    tails = {int(r["cycle"]): r for r in csv.DictReader((BASE / "tail_batches.csv").open(encoding="utf-8")) if r["arm"] == "mixed99"}
    source = Path(f"reports/{RUN}/generated/status-20261114/records")
    metrics = {a: {c: read_json(source / (RUN + "-" + a) / f"seed_42/gen_{c:02d}/metrics.json") for c in range(11)}
               for a in ("real", "mixed99", "synthetic")}
    blocks = []

    def add(kind, text="", **kw):
        blocks.append({"type": kind, "text": text, **kw})

    def heading(text, page=True):
        add("h2", text, page=page)

    def bullets(*texts):
        for text in texts:
            add("bullet", text)

    def table(key, caption, rows, widths):
        add("caption", caption)
        add("table", id=key, rows=rows, widths=widths)

    def image(name, caption):
        add("image", path=str((BASE / (name + ".png")).resolve()))
        add("caption", caption)

    add("h1", "Week 5: Why can a small amount of real data restrain recursive deterioration?", page=True)
    heading("1. Question and principal findings", False)
    add("body", "The earlier fraction sweeps showed much less deterioration at 99% synthetic targets than at 100%. We investigate the mechanism in two stages: local interventions on saved AdamW checkpoints, followed by new SGD runs that measure real and synthetic contributions at every training update.")
    bullets(
        "Real examples provide a measurable corrective signal. In the new 99% run, including the real component improves the fixed real-validation probe in all 563 updates that contain real examples; 227 updates contain none.",
        "Synthetic inclusion worsens that same probe in 778 of 790 updates in the 99% arm. The real correction strengthens as the model becomes worse at real text.",
        "The 99% full-test curve rises strongly and then oscillates. Small final batches explain sharp corrections at cycles 6 and 10. This supports a moving competition between corrections and deterioration, rather than permanent stabilization.")
    table("conditions", "Table W5.1. New conditions. Synthetic refers to the continuation targets; all prompts remain real. Each arm starts independently from pretrained SmolLM2.",
          [["Synthetic fraction", "Real pairs/cycle", "Generated pairs/cycle"], ["0%", "10,000", "0"], ["99%", "100", "9,900"], ["100%", "0", "10,000"]], [130, 160, 160])

    heading("2. First diagnostic: the historical AdamW checkpoints")
    add("body", "Before changing the training recipe, we restored five distinct checkpoints from the earlier SmolLM2 99%/100% sweep: shared M0 and each arm's M4/M9, immediately before cycles 1/5/10. Exact saved targets and optimizer states were reused; no recursive trajectory or replacement synthetic targets were generated.")
    bullets(
        "FP32, dropout-disabled aggregate gradients separated the actual 100 real and 9,900 synthetic rows. At M4, the weighted real norm was 0.330 versus 0.569 for synthetic: a ratio of 58%, despite 1% of examples.",
        "At M4, one synthetic-only AdamW intervention increased real-probe NLL by 0.001818, while the original 1%/99% mixed intervention decreased it by 0.001022. Both began from identical weights and inherited optimizer state.",
        "Later raw source gradients were positively aligned, so a simple cancellation story was insufficient. Clearing only AdamW's inherited first moment reversed the synthetic intervention's effect in a supplementary control. Optimizer history affects the weight displacement.",
        "These are one-step aggregate-gradient interventions on a 1,024-pair validation probe, not historical minibatch replays. They motivated a simpler SGD experiment rather than establishing the complete cause of the earlier plateau.")
    table("adam", "Table W5.2. Historical 99% diagnostic. Weighted norms precede AdamW; the numerical ratio is not a percentage of useful influence.",
          [["Before cycle", "Real/synthetic weighted norm ratio", "Mixed probe ΔNLL"], ["1 (M0)", "0.00293", "+0.0001719"], ["5 (M4)", "0.580", "−0.001022"], ["10 (M9)", "0.461", "−0.003313"]], [105, 225, 120])

    heading("3. New experiment: constant-rate SGD and greedy decoding")
    bullets(
        "Reuse pinned SmolLM2-135M base and its frozen FineWeb-Edu sample-10BT native-tokenizer bank. Ten fresh, document-separated chunks contain 10,000 prompt/continuation pairs each. Both validation and test stay fixed and outside experimental training.",
        "At cycle k, each arm's preceding checkpoint generates greedy continuations for its selected synthetic rows. Scatter them into their original prompt positions, retain the original real targets elsewhere, freeze the mixture, and train that same checkpoint for one epoch. Weights continue across cycles; there is no reset.",
        "Use BOS + 64 real prompt tokens + 64 target tokens. Attention sees the prompt; loss scores only the 64 continuation tokens. Each cycle has 640,000 scored targets, and each arm has 6.4 million target-token exposures.",
        "A bounded real-only learning-rate screen selected 0.01 as the best tested rate. Test data and synthetic outcomes did not rank candidates; calibration weights were discarded. All three production arms restarted pretrained M0.")
    table("recipe", "Table W5.3. Shared SGD recipe. This changes several settings relative to the historical AdamW/five-beam sweep, so it is not an optimizer-only comparison.",
          [["Setting", "Value"], ["Optimizer / learning rate", "Plain SGD / constant 0.01"], ["Momentum / weight decay", "0 / 0"], ["Schedule / warm-up / clipping", "None / none / none"], ["Arithmetic", "FP32; TF32 disabled"], ["Synthetic generation", "Greedy; exactly 64 tokens; repetition penalty 1.0"], ["Microbatch / effective batch", "32 / 128; final batch 16"], ["Cycles / epochs / updates per arm", "10 / 10 / 790"], ["Seed / checkpoint policy", "42 / terminal epoch"]], [180, 270])

    heading("4. Primary result: held-out real-test perplexity")
    bullets(
        "Shared test PPL is 17.958. After ten cycles, real-only is 17.726, 99% synthetic is 164.413 and 100% synthetic is 1,408.607. The 1% real component substantially restrains deterioration, while the 99% condition remains far worse than baseline.",
        "The 99% endpoints are irregular, with drops at cycles 6 and 10. The 100% endpoints rise at every cycle. Constant learning rate removes LR decay as an explanation for these particular corrections.")
    image("full_test_perplexity", "Figure W5.1. Token-weighted corpus perplexity on all 7,303 fixed real-test pairs / 467,392 continuation targets. Logarithmic y-axis; connecting lines join measured cycle endpoints. One seed per condition.")

    heading("5. Measuring each source's immediate effect")
    add("body", "For each actual batch, directly measure the real and synthetic loss gradients at the same starting weights. Normalize both by the whole batch's scored-target count, so their sum reconstructs the ordinary combined gradient. The actual update remains authoritative.")
    add("equation", "g = gR + gS;       θnext = θ − 0.01(gR + gS)")
    bullets(
        "On a scratch model, evaluate unchanged weights (L0), the real component alone (LR), synthetic alone (LS), and the actual mixed update (LM). Each intervention starts from identical weights and retains the original batch weighting.",
        "The same fixed 128 real-validation pairs, from 128 documents, supply 8,192 scored tokens. No held-out gradients train the model. The per-update probe is separate from complete epoch validation/test evaluation.",
        "Real conditional effect = LM − LS; synthetic conditional effect = LM − LR. Negative ΔNLL improves real-text prediction; positive ΔNLL worsens it. Omitting one component does not renormalize the other.",
        "Curvature means conditional effects need not add. The interaction is LM − LR − LS + L0. Document-bootstrap intervals measure evaluation-sample variability, not uncertainty across training seeds.")
    table("effects", "Table W5.4. Mean conditional probe ΔNLL per actual update in the 99% arm. Means include all 79 updates, including zero contributions from absent real sources.",
          [["Cycle", "Updates containing real", "Real effect", "Synthetic effect"],
           *[[str(c), f"{mixed[c]['real_present_updates']} / 79", f"{float(mixed[c]['real_conditional_mean_nll']):+.6f}", f"{float(mixed[c]['synthetic_conditional_mean_nll']):+.6f}"] for c in (1, 5, 6, 10)]], [60, 150, 120, 120])

    heading("6. Real corrections strengthen over training")
    bullets(
        "Including real data helps the probe in all 563 real-present updates. Synthetic inclusion hurts in 778/790 and helps in 12; this statement concerns the 99% trajectory, not every possible synthetic update.",
        "The average real correction grows from −0.00225 NLL/update in cycle 1 to −0.04939 in cycle 10. The corresponding synthetic effect grows from +0.00930 to +0.04549. Late corrections can therefore compete with the synthetic pressure.")
    image("99_source_help_hurt", "Figure W5.2. Conditional local effects along the observed 99% training path. Below zero helps held-out real prediction. Bands are 95% paired document-bootstrap intervals, with 2,000 resamples; updates are not independent replicates.")

    heading("7. Why 1% of examples can matter much more than 1%")
    bullets(
        "Actual weighted real-gradient norm averages 0.100 in cycle 1 and 0.297 in cycle 10; synthetic averages 0.877 and 0.485. Among batches containing both sources, the median weighted real/synthetic ratio grows from 0.165 to 0.703.",
        "Real-target training NLL rises from 3.215 to 5.333, while synthetic-target NLL falls from 0.508 to 0.060. The degraded model finds real continuations surprising but is increasingly confident on its own targets. These are fresh-chunk pre-update training averages, not fixed-set evaluations.",
        "Source directions also increasingly oppose: median cosine moves from −0.111 to −0.363 in both-source batches. Parameter-gradient size depends on errors, network derivatives and cancellation across examples; norm size alone does not establish usefulness.")
    image("99_weighted_gradient_norms", "Figure W5.3. Mean gradient norms over all 79 updates/cycle, including real-absent zero contributions. Actual batch proportions are already included. The direct validation interventions, rather than norms alone, establish the measured benefit.")

    heading("8. What causes the sharp cycle-6 and cycle-10 drops?")
    bullets(
        "Each cycle has 78 batches of 128 and a final batch of 16. In cycles 1, 6 and 10, that final batch contains one real example. Its averaging weight is 1/16 rather than 1/128: eight times the weight of one example in a full batch.",
        "The late final updates strongly improve the real probe. This makes the cycle endpoint sensitive to the random placement of scarce real examples, even though every cycle still contains exactly 100 real and 9,900 synthetic targets.")
    image("99_tail_batch_corrections", "Figure W5.4. Actual per-update 99% probe perplexity. Crosses show measured synthetic-only shadows for the last update at cycles 6/10; arrows show the correction when real is included. These are 128-pair probe values, not full-test counterfactuals.")

    heading("9. Direct numerical evidence for the final-batch effect")
    bullets(
        "At cycle 6, the preceding 78 updates worsen probe NLL overall by +0.09146. The final update improves it by −0.34946, causing a net improving cycle. Its synthetic-only shadow would worsen the probe; the one real example supplies the correction.",
        "At cycle 10, the preceding 78 updates already improve NLL slightly (−0.02186), followed by −0.23980 at the tail. Thus most of that cycle's improvement also occurs at the final update.",
        "The complete test has dips at the same cycle endpoints. The probe provides direct one-step evidence for their timing; the full test was not evaluated before and after each shadow. We cannot assign the exact full-test drop to one component.")
    table("tails", "Table W5.5. Final update in the 99% arm. Both shadows use the same starting weights and original batch-weighted components.",
          [["Cycle", "Real / total", "PPL before", "Synthetic only", "Actual mixed"],
           *[[str(c), "1 / 16", f"{float(tails[c]['no_update_ppl']):.2f}", f"{float(tails[c]['synthetic_ppl']):.2f}", f"{float(tails[c]['mixed_ppl']):.2f}"] for c in (6, 10)]], [50, 85, 100, 110, 105])
    table("endpoints", "Table W5.6. Complete test PPL. These values use all fixed test pairs and differ from the miniature probe above.",
          [["Cycle", "0% synthetic", "99% synthetic", "100% synthetic"],
           *[[str(c), *[f"{metrics[a][c]['test']['perplexity']:.2f}" for a in ("real", "mixed99", "synthetic")]] for c in (0, 1, 5, 6, 9, 10)]], [60, 130, 130, 130])

    heading("10. Why can the trajectory appear to stabilize?")
    bullets(
        "The corrective real signal strengthens as the model drifts away from real text. In late cycles, it can nearly balance synthetic deterioration, or temporarily outweigh it. A plateau can therefore emerge from changing update effects even with constant LR. The 100% arm has no real-target gradient component to supply this correction.",
        "For additive accounting, average each source's isolated and conditional effects: real = ½[(LR−L0)+(LM−LS)]; synthetic = ½[(LS−L0)+(LM−LR)]. This splits the nonlinear interaction equally; it is a stated attribution convention.",
        "Across 790 updates the allocations are −24.884 real and +27.114 synthetic, giving net +2.231 probe NLL. The real allocation offsets about 91.8% of the local synthetic allocation under this convention. These sums follow the actual evolving path; they do not simulate removing a source for ten cycles.")
    image("99_additive_local_allocation", "Figure W5.5. Per-cycle sums of symmetric local allocations, which add exactly to the observed net probe-NLL change. A moving near-balance and tail corrections explain the apparent flattening; ten cycles do not establish a limiting equilibrium.")

    heading("11. Generated-text diagnostics and metric scope")
    bullets(
        "The usual generation metrics use 256 fixed real-validation prompts and 64-token greedy outputs at each checkpoint. This is a different population from the 128-pair teacher-forced update probe.",
        "Pooled token entropy measures vocabulary-frequency spread across all outputs; within-output entropy measures local token variety. Neither is predictive-distribution entropy. Distinct-2/3 measure unique token phrases divided by all phrases within outputs, pooled across the fixed sample.",
        "Repeated four-grams inside an output detect loops. These statistics describe text form, while held-out PPL measures the probability of real continuations. They need not rank conditions identically, and a single repetitive example cannot establish a population rate.",
        "At cycle 10, the 99% and 100% arms have almost equal local entropy (about 2.64 nats) and repeated-four-gram rates (about 58.8%), despite an 8.6-fold test-PPL gap. Similar greedy text form can coexist with very different probabilities for held-out real continuations; these output statistics do not imply equal predictive ability.")
    labels = {"real": "0%", "mixed99": "99%", "synthetic": "100%"}
    table("outputs", "Table W5.7. Saved fixed-greedy-probe metrics at baseline and cycle 10. H is in nats; D2/D3 and local repeated four-grams are percentages.",
          [["Checkpoint", "Pooled H", "Local H", "D2", "D3", "Local repeat-4"],
           *[[label, f"{m['probe']['token_entropy_nats']:.3f}", f"{m['probe']['mean_within_entropy_nats']:.3f}", f"{100*m['probe']['distinct_2']:.2f}", f"{100*m['probe']['distinct_3']:.2f}", f"{100*m['probe']['mean_within_repeated_4gram_rate']:.2f}"] for label, m in [("M0", metrics['real'][0]), *[(labels[a]+" M10", metrics[a][10]) for a in labels]]]], [80, 70, 70, 65, 65, 100])
    table("losses", "Table W5.8. Complete held-out likelihood at terminal checkpoints; NLL is nats per scored continuation token.",
          [["Synthetic", "Validation PPL", "Test NLL", "Median example NLL", "P95 example NLL"],
           *[[labels[a], f"{metrics[a][10]['validation']['perplexity']:.2f}", f"{metrics[a][10]['test']['loss']:.4f}", f"{metrics[a][10]['test']['median_example_nll']:.4f}", f"{metrics[a][10]['test']['p95_example_nll']:.4f}"] for a in labels]], [65, 95, 80, 110, 100])

    heading("12. Conclusions, limitations and the next control")
    bullets(
        "Measured: the rare real targets have a beneficial immediate effect and grow more influential along the 99% SGD path. Synthetic targets usually worsen real-probe predictions there. This is direct evidence for a corrective real-data mechanism under this recipe.",
        "Interpretation: as synthetic training moves the model away from real continuations, those continuations ask for increasingly strong corrections. Their sparse but large updates can approach a balance with synthetic updates. The smaller tail batch amplifies particular corrections and affects cycle endpoints.",
        "Stabilization remains provisional. The 99% endpoint is still about 9.2× its initial full-test PPL, with large within-cycle fluctuations. One seed and ten cycles cannot establish a permanent plateau, universal real-data threshold or irreversible collapse.",
        "The earlier AdamW result also involves optimizer history, LR decay and five-beam targets. The new recipe changes optimizer, schedule, regularization, precision and decoding together. It shows that LR decay and Adam momentum are not necessary for these SGD corrections, but does not identify an optimizer-only causal effect.",
        "A targeted next control would give the 16-row tail the same per-example update weight as full batches, then repeat several training-order seeds. More cycles and LR sensitivity would test whether the apparent balance persists. These follow-up runs have not been performed.",
        "All 2,370 update journals and per-example scores were verified from pinned backups. Component reconstruction error is at most 2.65e-7 relative, mixed shadow weights match actual updates, and probe states telescope exactly. Bootstrap intervals describe probe-document variability only. Full model/optimizer cold restoration was not repeated during this offline analysis.")
    table("execution", "Table W5.9. Execution and recovery scope. Historical resources remain preserved; no new GPU work was used to prepare this report.",
          [["Item", "Recorded outcome"], ["New SGD training", "30 terminal epochs; 2,370 updates"], ["Hardware", "RTX PRO 4000 Blackwell"], ["Automatic compute shutdown", "14 Nov 2026, 07:30:26 UTC; provider verified"], ["Full Pod lifetime / provider bill", "7 h 16 min / USD 4.166; storage/setup additional"], ["Offline analysis recovery", "All selected update journals, scores and seals"], ["New report compute charge", "USD 0"]], [190, 260])

    atomic_json(out / "blocks.json", {"run_id": RUN, "analysis": str(BASE.resolve()), "blocks": blocks})
    lines = []
    for b in blocks:
        kind, text = b["type"], b.get("text", "")
        if kind in ("h1", "h2"):
            lines.append(("# " if kind == "h1" else "## ") + text)
        elif kind == "bullet":
            lines.append("- " + text)
        elif kind == "image":
            lines.append("![Research figure](../update-gradients-20261114/" + Path(b["path"]).name + ")")
        elif kind == "table":
            rr = b["rows"]
            lines += ["| " + " | ".join(rr[0]) + " |", "| " + " | ".join(["---"] * len(rr[0])) + " |",
                      *["| " + " | ".join(row) + " |" for row in rr[1:]]]
        else:
            lines.append(text)
        lines.append("")
    (out / "section.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps({"blocks": len(blocks), "figures": sum(b["type"] == "image" for b in blocks),
                      "tables": sum(b["type"] == "table" for b in blocks), "out": str(out)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    compose(parser.parse_args().out)
