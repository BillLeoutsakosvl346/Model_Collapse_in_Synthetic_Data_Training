"""Author a continuation of Week 5 from verified 50/80 SGD source diagnostics."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from model_collapse.common import atomic_json, read_json


def compose(source, out):
    source, out = Path(source), Path(out)
    if out.exists():
        raise FileExistsError("Choose a new authored destination")
    out.mkdir(parents=True)
    read = lambda name: list(csv.DictReader((source / name).open(encoding="utf-8")))
    summary = read_json(source / "summary.json")
    effects = {(r["arm"], int(r["cycle"])): r for r in read("cycle_effects.csv")}
    sources = {(r["arm"], int(r["cycle"])): r for r in summary["source_summaries"] if r["population"] == "all_updates"}
    phases = {(r["arm"], r["phase"], r["population"]): r for r in summary["phases"]}
    endpoints = {r["arm"]: r for r in summary["endpoints"]}
    outputs = {r["arm"]: r for r in read("output_metrics.csv") if int(r["cycle"]) == 10}
    baseline = float(next(r for r in read("test_trajectory.csv") if r["arm"] == "real" and int(r["cycle"]) == 0)["test_perplexity"])
    labels = {"real": "0%", "mixed50": "50%", "mixed80": "80%", "mixed99": "99%", "synthetic": "100%"}
    blocks = []
    def add(kind, text="", **kw): blocks.append({"type": kind, "text": text, **kw})
    def heading(text): add("h2", text, page=True)
    def bullets(*texts):
        for text in texts: add("bullet", text)
    def table(key, text, rows, widths):
        add("caption", text, before_table=True)
        add("table", id=key, rows=rows, widths=widths)
    def image(name, caption):
        add("image", path=str((source / (name + ".png")).resolve()))
        add("caption", caption)

    add("h1", "Week 5 continued: Balancing real and synthetic updates", page=True)
    add("body", "The matched 50% and 80% synthetic conditions remain close to pretrained real-test performance. Direct source diagnostics show strong, continuing real corrections opposing synthetic updates. Evaluation can remain nearly stable while both gradients and parameter updates remain substantial.")
    bullets("Each arm independently starts the same pretrained SmolLM2-135M and continues its own weights through ten different, document-separated 10,000-pair FineWeb-Edu chunks. One terminal epoch is retained per cycle; there is no reset.",
        "Reuse the native-tokenizer bank, fixed evaluation collections, seed 42, plain SGD at constant 0.01, FP32 with TF32 off, and greedy exactly-64-token generation. There is no momentum, decay, clipping or scheduler. Critical computation files match the earlier SGD controls.",
        "Inputs contain BOS + 64 real prompt tokens + 64 targets. Loss scores only targets. Each arm generates its own selected continuations using its preceding checkpoint, retains real targets elsewhere, freezes the mixture and then trains. The nested row selections are deterministic.",
        "Every cycle scores 640,000 targets. Microbatches of 32 accumulate into 78 updates of 128 examples and a final update of 16. Each arm has 790 updates. Held-out document identities and evaluation arrays never enter training.")
    table("endpoints", "Table W5B.1. Matched cycle-10 outcomes. The 0%, 99% and 100% SGD controls are preserved historical runs; 50% and 80% are new. All share initial test PPL 17.958.",
        [["Synthetic", "Real / cycle", "Synthetic / cycle", "Final test PPL", "Change vs M0"],
         *[[labels[a], f"{10000-round(10000*f):,}", f"{round(10000*f):,}", f"{endpoints[a]['test_perplexity']:.3f}",
            f"{100*(endpoints[a]['test_perplexity']/baseline-1):+.1f}%"]
           for a,f in (("real",0),("mixed50",.5),("mixed80",.8),("mixed99",.99),("synthetic",1))]], [65,95,110,90,90])

    heading("1. The new conditions avoid the large likelihood deterioration")
    bullets("Final real-test PPL is 18.586 at 50% synthetic and 19.707 at 80%, compared with 17.726 for real-only, 164.413 at 99% and 1,408.607 at 100%. These are full held-out continuation evaluations, not the small update probe.",
        "Both new mixtures are worse than real-only, but greatly outperform the high-synthetic controls. All conditions have constant learning rate; LR decay cannot explain their different trajectories. This remains one seed per condition.")
    image("01_test_perplexity_all", "Figure W5B.1. Full real-test corpus PPL on 7,303 fixed pairs / 467,392 scored tokens. The logarithmic scale reveals all five conditions. Lines connect recorded cycle endpoints; the shared baseline is reused evidence.")

    heading("2. A closer view shows an overshoot and partial recovery")
    bullets("At 50%, test PPL remains between 18.586 and 19.295 after adaptation, ending 3.5% above M0. At 80%, it peaks at 24.175 in cycle 2, then ends 9.7% above M0. The 80% trajectory improves after its early deterioration rather than becoming perfectly flat.",
        "The late 80% endpoints still fluctuate. Its final drop is partly amplified by the small final batch, examined below. Ten cycles show a bounded-looking observed trajectory; they do not establish a permanent equilibrium or a universal safe mixture fraction.")
    image("02_test_perplexity_zoom", "Figure W5B.2. The same complete-test metric on a linear, expanded scale for 0%, 50% and 80% synthetic. The first point after baseline represents one complete epoch on a fresh 10,000-pair chunk.")

    heading("3. Real examples supply much more loss than their row fraction")
    add("equation", "L_B = L_R + L_S;  L_R = (Σreal ℓᵢ)/N_B;  L_S = (Σsynthetic ℓᵢ)/N_B")
    bullets("Each ℓᵢ is mean NLL over that example's 64 continuation tokens. N_B is 128, or 16 for the tail. Source terms already contain their actual batch proportions; we do not multiply their gradients by the mixture percentage a second time.",
        "At 50% synthetic, real examples supply 82.4% of mean batch loss in cycle 1 and 93.3% in cycle 10. At 80% synthetic, the 20% real examples supply 58.4% initially and 87.5% by cycle 10. Synthetic targets become easy for the model to predict while real targets remain comparatively surprising.",
        "These percentages describe scalar training loss. They are not percentages of the gradient, parameter change or predictive benefit. All-batch and corpus-token weighting agree closely here, and the full-batch-only results retain the same pattern.")
    image("03_real_loss_shares", "Figure W5B.3. Real share = mean(L_R) / [mean(L_R) + mean(L_S)], averaging all 79 updates/cycle. This is a ratio of source loss sums, not the mean of per-batch percentages. The 99% reference contains only 1% real examples.")

    heading("4. Per-cycle loss accounting")
    bullets("The following tables show the actual weighted terms entering backpropagation. For example, at 80% synthetic in cycle 10, mean real loss is 0.6120 and synthetic loss is 0.0871: real supplies 87.5% of their sum. Corresponding within-source corpus NLLs are 3.045 real versus 0.109 synthetic.",
        "Different chunks are used each cycle. These pre-update training averages follow changing weights and synthetic targets; they are not evaluations on one fixed training collection.")
    for arm in ("mixed50","mixed80"):
        table("losses_"+arm, "Table W5B."+("2" if arm=="mixed50" else "3")+f". {labels[arm]} synthetic: mean batch-weighted source loss, in nats per target token. Counts remain fixed while loss shares evolve.",
            [["Cycle","Real loss term","Synthetic loss term","Real share"],
             *[[str(c),f"{sources[arm,c]['mean_weighted_real_nll']:.4f}",f"{sources[arm,c]['mean_weighted_synthetic_nll']:.4f}",f"{100*sources[arm,c]['real_share_of_mean_batch_loss']:.1f}%"] for c in range(1,11)]], [50,130,140,130])

    heading("5. Backpropagation measures vectors rather than loss shares")
    add("equation", "g_R = ∇θ L_R;   g_S = ∇θ L_S;   θ_next = θ − 0.01(g_R + g_S)")
    bullets("Source gradients are directly differentiated from the same forward graphs at unchanged weights. Their sum reconstructs the ordinary batch gradient. Only that combined update advances training; the diagnostics do not supply an additional update.",
        "In the 50% condition at cycle 10, mean weighted real and synthetic norms are 1.096 and 0.936; the actual combined norm is 0.709. Thus 93.3% of loss does not mean 93.3% of an update. Direction matters, and the nonzero combined gradient shows continued learning.")
    image("04_mixed50_gradients", "Figure W5B.4. Means of 79 per-update gradient norms. The grey reference is mean √(‖g_R‖²+‖g_S‖²), as if the measured component vectors were perpendicular. It is not calculated from their epoch-mean vectors.")

    heading("6. At 80% synthetic the minority real gradient is comparably strong")
    bullets("Despite supplying 20% of examples, the weighted real norm grows from 0.729 to 1.371; synthetic grows from 0.985 to 1.321. By cycle 10 they have similar strengths. The actual combined norm is 0.907, considerably shorter than either component.",
        "Synthetic scalar loss falls even though its gradient norm rises late. Loss values do not determine parameter derivatives: model sensitivity, changing targets and cancellation among examples also matter. These records establish the divergence, but do not isolate which factor causes it.",
        "At constant LR 0.01, cycle-10 combined norms imply typical update lengths around 0.0071 (50%) and 0.0091 (80%). Evaluation stability therefore does not mean that updates or parameter movement have stopped.")
    image("04_mixed80_gradients", "Figure W5B.5. Actual batch weighting is already included. Norms quantify movement in parameter space, not whether that movement improves real-text predictions. The held-out interventions below measure that separately.")

    heading("7. The sources oppose much more strongly than at 99%")
    bullets("Both new conditions have opposing source gradients in all 790 updates. Median cosine at cycle 10 is −0.809 at 50% and −0.899 at 80%, compared with −0.363 at 99% among batches containing both sources. Negative cosine means partial opposition.",
        "At cycle 10, opposition reduces mean combined length by 51.3% (50%) and 54.0% (80%) relative to the perpendicular reference. The corresponding 99% reduction is about 12.2%. Cancellation is substantial but incomplete, and its magnitude alone does not establish held-out benefit.")
    image("07_gradient_cancellation", "Figure W5B.6. Reduction = 100 × [1 − mean(‖g_R+g_S‖) / mean(√(‖g_R‖²+‖g_S‖²))]. Positive values indicate opposition. Every source proportion is already included; this is not a percentage of synthetic-data damage removed.")

    heading("8. At 50% real updates help and synthetic updates hurt locally")
    bullets("From the same actual starting weights, a scratch model scores unchanged weights (L0), real-only change (LR), synthetic-only change (LS) and the actual mixed change (LM) on 128 fixed real-validation pairs. Original source weights are retained; an omitted source is not renormalized.",
        "Real conditional effect is LM−LS; synthetic conditional effect is LM−LR. In all 790 updates, including real improves the probe and including synthetic worsens it. The isolated comparisons to unchanged weights have the same signs in this run.",
        "The two plotted effects use different counterfactual baselines and need not add to the actual change. They describe immediate effects along the observed path, not ten-cycle trajectories with one source removed.")
    image("05_mixed50_help_hurt", "Figure W5B.7. Mean conditional ΔNLL/update at 50% synthetic. Below zero improves the fixed real probe. Bands are 95% paired document-bootstrap intervals from 2,000 resamples; they do not measure training-seed uncertainty.")

    heading("9. At 80% the opposing predictive effects grow larger")
    bullets("Again, real inclusion helps and synthetic inclusion hurts in all 790 updates. By cycle 10, mean conditional effects are −0.114386 real and +0.054855 synthetic NLL/update, while the actual mixed change is only −0.001600.",
        "This is not an arithmetic contradiction: conditional effects include a nonlinear interaction. In cycle 10 that interaction averages −0.057932; actual change equals real conditional + synthetic conditional − interaction.",
        "The larger local effects indicate strong opposing pressures around the observed model, not a large net improvement each update. The next section gives an explicit additive accounting convention.")
    image("05_mixed80_help_hurt", "Figure W5B.8. Same 128 real pairs / 8,192 scored tokens, distinct from the complete epoch test. No held-out gradient trains the model. These one-step interventions retain the actual learning rate and batch weighting.")

    heading("10. Why prediction quality can stay nearly stable")
    add("equation", "I = LM − LR − LS + L0")
    add("equation", "φ_R = ½[(LR−L0)+(LM−LS)];  φ_S = ½[(LS−L0)+(LM−LR)]")
    add("equation", "φ_R + φ_S = LM − L0")
    bullets("Splitting the nonlinear interaction equally gives symmetric local allocations that add exactly to the actual probe change. This is a transparent attribution convention, not an experiment removing a source for ten epochs.",
        "Early synthetic deterioration slightly exceeds the correction. Later, the allocations almost balance: at 80%, cycle-10 means are −0.085420 real and +0.083820 synthetic, leaving −0.001600 net. At 50%, late average net change is only −0.000022 NLL/update.",
        "A plausible feedback explanation is that self-generated greedy targets encourage confidence in existing favourite continuations, while real targets restore probability to alternatives. The measured corrections support that explanation; a fixed point or complete causal account has not been established.")
    table("balance", "Table W5B.4. Mean symmetric allocations per actual update, including tails. Early means cycles 1–3; late means cycles 6–10. Their sum equals the net column exactly before rounding.",
        [["Synthetic","Window","Real allocation","Synthetic allocation","Actual net"],
         *[[labels[a],title,f"{r['real_conditional_delta_nll']-.5*r['interaction_nll']:+.6f}",f"{r['synthetic_conditional_delta_nll']-.5*r['interaction_nll']:+.6f}",f"{r['mixed_delta_nll']:+.6f}"]
           for a in ("mixed50","mixed80","mixed99") for title,key in (("1–3","early_cycles_1_to_3"),("6–10","late_cycles_6_to_10"))
           for r in [phases[a,key,"all_updates"]]]], [65,50,110,115,110])
    table("nonlinear", "Table W5B.5. Cycle-10 conditional effects and interaction. Actual net = real conditional + synthetic conditional − I; the conditional columns alone are not additive.",
        [["Synthetic","Real conditional","Synthetic conditional","Interaction I","Actual net"],
         *[[labels[a],f"{float(effects[a,10]['real_conditional_mean_delta_nll']):+.6f}",f"{float(effects[a,10]['synthetic_conditional_mean_delta_nll']):+.6f}",f"{sources[a,10]['interaction_nll']:+.6f}",f"{sources[a,10]['mixed_delta_nll']:+.6f}"] for a in ("mixed50","mixed80")]], [65,105,105,90,85])

    heading("11. The final small batch still affects cycle endpoints")
    bullets("The final 16 examples have eight times the per-example averaging weight of a full 128-row batch. At 80% in cycle 10, that tail contains 5 real and 11 synthetic examples; its probe change is −0.094650 NLL, about 75% of the cycle's −0.126389 total improvement.",
        "This is not the whole mechanism: late full batches also improve on average (−0.000572 NLL/update at 80%, −0.000102 at 50%). Over cycles 6–10, tails worsen the probe on average in both new runs. The strong opposing gradients and loss-share pattern persist when tails are excluded.",
        "Counts alone do not determine the sign. The 80% tails in cycles 5, 6 and 10 all contain five real examples, but their effects differ. Model state and example content matter alongside source proportions and batch size.")
    image("10_update_path_and_tails", "Figure W5B.9. Actual per-update probe PPL, with final 16-row updates marked. The final 80% jump is real, but these are 128-pair probe values, not complete-test values or full alternative trajectories.")

    heading("12. Likelihood preservation does not preserve every text statistic")
    bullets("On the fixed 256-prompt greedy-output probe, within-output entropy ends at 3.101 nats for real-only, 2.928 at 50% and 2.760 at 80%. Local repeated four-grams rise from 31.2% for real-only to 46.3% and 54.5%. Real targets restrain likelihood deterioration more strongly than they prevent all greedy-output repetition.",
        "Pooled entropy at 50% is higher than real-only (6.052 versus 5.959 nats), despite lower local variety. Different outputs can collectively use more vocabulary while each repeats internally. Neither quantity is the entropy of the model's next-token distribution.",
        "Distinct-2/3 count unique token phrases divided by phrase occurrences in this fixed sample, without crossing continuation boundaries. Their scope differs from within-output repetition and probability assigned to held-out real continuations.")
    image("08_output_entropy", "Figure W5B.10. Mean empirical token entropy inside each 64-token generated continuation, averaged over 256 fixed prompts. Lower values mean less local token variety; they do not directly measure predictive uncertainty or factual accuracy.")

    heading("13. Conclusions and remaining limitations")
    bullets("Measured: the 50% and 80% mixtures have large, opposing real/synthetic gradients. Real source updates consistently improve the fixed real probe, while synthetic source updates worsen it. Their near-balanced additive allocations are consistent with the modest observed likelihood drift.",
        "At cycle 10, 50% real examples supply 93.3% of mean batch loss; 20% real examples supply 87.5%. These are loss shares, not percentages of gradient or causal benefit. The separate vector and intervention measurements establish their corrective influence.",
        "Stabilization is an evaluation-level, finite-horizon observation. Parameter updates remain nonzero; the 80% curve overshoots then partly recovers, and endpoint values remain sensitive to tail batches. One seed, ten cycles and one fixed diagnostic probe cannot establish permanent stability or a universal threshold.",
        "All 3,950 sealed updates across five conditions were rechecked. Gradient reconstruction error is at most 3.13e-7 relative, shadow/actual updates match, and probe states telescope exactly. The 21 unique new output arrays were recomputed; historical output rows are preserved saved measurements.",
        "No new model training or inference was needed for this analysis. Full checkpoint cold recovery was not repeated. A tail-normalization control and repeated training-order seeds would test how robust the apparent balance is; those follow-ups have not been run.")
    table("output_metrics", "Table W5B.6. Cycle-10 fixed-greedy-probe outputs. H is empirical token entropy in nats; D2/D3 and repeated four-grams are percentages. Pooled and within-output scopes are distinct.",
        [["Synthetic","Pooled H","Local H","Distinct-2","Distinct-3","Local repeat-4"],
         *[[labels[a],f"{float(outputs[a]['token_entropy_nats']):.3f}",f"{float(outputs[a]['mean_within_entropy_nats']):.3f}",f"{100*float(outputs[a]['distinct_2']):.2f}",f"{100*float(outputs[a]['distinct_3']):.2f}",f"{100*float(outputs[a]['mean_within_repeated_4gram_rate']):.2f}"] for a in labels]], [65,70,70,75,75,95])
    table("scope", "Table W5B.7. Execution and analysis scope. Retained cloud recovery resources remain preserved.",
        [["Item","Outcome"],["New full training","20 epochs; 1,580 updates; two independent arms"],
         ["GPU completion","14 Nov 2026, 20:45:56 UTC; provider EXITED"],
         ["New full Pod lifetime","5 h 37 min; storage remains retained"],
         ["Diagnostic analysis","3,950 sealed updates including historical controls"],
         ["Recovery scope","Small diagnostics/probes verified against sealed backup indexes; no full bulk cold restore"],
         ["New analysis GPU compute","None"]], [155,295])

    atomic_json(out / "blocks.json", {"source": str(source.resolve()), "blocks": blocks})
    lines=[]
    for b in blocks:
        k=b['type'];t=b.get('text','')
        if k in ('h1','h2'):lines.append(('# ' if k=='h1' else '## ')+t)
        elif k=='bullet':lines.append('- '+t)
        elif k=='image':lines.append('![Research figure](../'+source.name+'/'+Path(b['path']).name+')')
        elif k=='table':
            rr=b['rows'];lines.extend(['| '+' | '.join(rr[0])+' |','| '+' | '.join(['---']*len(rr[0]))+' |',*['| '+' | '.join(r)+' |' for r in rr[1:]]])
        else:lines.append(t)
        lines.append('')
    (out/'section.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print({'blocks':len(blocks),'figures':sum(b['type']=='image' for b in blocks),'tables':sum(b['type']=='table' for b in blocks),'out':str(out)})


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();compose(a.source,a.out)
