"""Build an illustrated reading guide from immutable published experiment records.

No model downloads, training, network calls or cloud-resource mutations.
The JSON is a native-Google-Docs authoring plan; figures are standard Matplotlib.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model_collapse.common import PROJECT, atomic_json, sha256

W1 = "20261020T211121Z-full"
CONT = "20261026T192740Z-full-continued"
RESET = "20261027T211852Z-full-reset-s0"
BLUE, ORANGE, PURPLE, GREY = "#347b9b", "#cc6b27", "#755493", "#666666"


def build(out: Path):
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new or empty output directory; published evidence is preserved")
    out.mkdir(parents=True, exist_ok=True)
    used = {}

    def source(rel):
        p = PROJECT / rel
        used[rel] = {"sha256": sha256(p), "bytes": p.stat().st_size}
        return p

    def js(rel):
        return json.loads(source(rel).read_text(encoding="utf-8"))

    def rows(rel):
        with source(rel).open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    w = rows(f"reports/{W1}/metrics.csv")
    c = rows(f"reports/{CONT}/metrics.csv")
    pair = rows(f"reports/{RESET}/analysis/metrics.csv")
    a = [r for r in pair if r["arm"] == "reset_s0"]
    b = [r for r in pair if r["arm"] == "continued"]
    hist = rows(f"reports/{RESET}/analysis/training_history.csv")
    assert len(w) == len(c) == len(a) == len(b) == 11
    assert [float(r["test_perplexity"]) for r in c] == [float(r["test_perplexity"]) for r in b]
    for i in (0, 1):
        assert a[i]["test_perplexity"] == b[i]["test_perplexity"]
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.fontsize": 7.3, "savefig.facecolor": "white"})
    figures = {}

    def save(fig, name, formula=None, height=2.65, footer=.16):
        fig.set_size_inches(6.26, height)
        if name == "02_distinct":
            # Separate legends, axis labels and equation into explicit bands.
            fig.subplots_adjust(left=.085, right=.985, top=.80, bottom=.40, wspace=.55)
            for ax in fig.axes:
                ax.set_xlabel("Cycle", fontsize=7.5)
                box = ax.get_position()
                ax.legend(loc="upper center", bbox_transform=fig.transFigure,
                          bbox_to_anchor=((box.x0 + box.x1) / 2, .275),
                          fontsize=6.6, frameon=False)
            fig.text(.5, .015, formula, ha="center", va="bottom", fontsize=12)
        elif formula:
            fig.text(.5, .015, formula, ha="center", va="bottom", fontsize=12)
            fig.tight_layout(rect=(0, footer, 1, 1), pad=.6, w_pad=.9)
        else:
            fig.tight_layout(pad=.6, w_pad=.9)
        p = out / (name + ".png")
        fig.savefig(p, dpi=280)
        plt.close(fig)
        figures[name] = {"file": p.name, "sha256": sha256(p), "width_pt": 450.72,
                         "height_pt": height * 72}

    def vals(rr, key, scale=1):
        return [float(r[key]) * scale for r in rr]

    def decorate(ax, label, ylabel):
        ax.set(title=label, xlabel="Synthetic-training cycle", ylabel=ylabel,
               xticks=[0, 2, 4, 6, 8, 10])
        ax.grid(alpha=.2)

    def match(ax, key, scale=1):
        for rr, col, label in ((b, ORANGE, "Continue (B)"), (a, PURPLE, "Reset to S0 (A)")):
            ax.plot(range(1, 11), vals(rr, key, scale)[1:], "o-", color=col, ms=2.7, lw=1.3, label=label)
        ax.plot([0, 1], vals(b, key, scale)[:2], "s-", color=GREY, ms=3.2, label="Shared S0/S1", zorder=4)
        ax.axvline(1, color="#bbbbbb", linestyle=":", lw=.7)

    fig, ax = plt.subplots(figsize=(6.26, 1.9))
    ax.axis("off")
    for y, label, detail, col in (
        (.82, "Week 1: pretrained reset", "previous output data  ->  train original pretrained OPT", BLUE),
        (.49, "Week 2: continued weights", "previous output data  ->  train the preceding checkpoint", ORANGE),
        (.16, "Week 2: fixed-S0 reset", "previous output data  ->  train the fixed real-trained S0", PURPLE)):
        ax.text(.015, y+.09, label, color=col, fontsize=10.5, weight="bold", transform=ax.transAxes)
        ax.text(.015, y-.045, detail, fontsize=9, transform=ax.transAxes)
    save(fig, "00_protocols", height=1.9)

    fig, axes = plt.subplots(1, 3)
    for ax, rr, title, col in ((axes[0], w, "A  Week 1\nPretrained reset", BLUE),
                               (axes[1], c, "B  Week 2\nContinued weights", ORANGE)):
        ax.plot(range(11), vals(rr, "test_perplexity"), "o-", color=col, ms=2.7, label="Test")
        ax.plot(range(11), vals(rr, "validation_perplexity"), "--", color="#444444", lw=1, label="Validation")
        decorate(ax, title, "Corpus perplexity")
        ax.legend()
    match(axes[2], "test_perplexity")
    decorate(axes[2], "C  Shared start\nReset versus continued", "Test perplexity")
    axes[2].legend()
    for ax in axes:
        ax.set_ylim(30, 66)
    save(fig, "01_perplexity", r"$L=-\frac{1}{N}\sum_i\ln p_\theta(x_i\mid x_{<i}),\qquad \mathrm{PPL}=e^L$")

    fig, axes = plt.subplots(1, 3)
    for ax, rr, title in ((axes[0], w, "A  Week 1"), (axes[1], c, "B  Continued")):
        for n, col in ((1, BLUE), (2, ORANGE), (3, "#43855d")):
            ax.plot(range(11), vals(rr, f"distinct_{n}", 100), "o-", ms=2.5, color=col, label=f"Distinct-{n}")
        decorate(ax, title, "Unique n-grams (%)")
        ax.set_ylim(0, 58)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.34),
                  ncol=1, fontsize=6.6, frameon=False)
    match(axes[2], "distinct_2", 100)
    decorate(axes[2], "C  Shared-start\nDistinct-2", "Unique bigrams (%)")
    axes[2].legend(loc="upper center", bbox_to_anchor=(.5, -.34),
                   ncol=1, fontsize=6.6, frameon=False)
    save(fig, "02_distinct", r"$\mathrm{distinct}\!\!-\!n=\frac{U_n}{T_n},\qquad T_n=256(64-n+1)$",
         height=3.05, footer=.31)

    for name, key, ylabel, formula in (
        ("03_repetition", "repeated_4gram_rate", "Repeated 4-grams (%)", r"$R_4=\frac{T_4-U_4}{T_4},\qquad T_4=256\times61=15{,}616$"),
        ("04_duplicates", "duplicate_continuation_rate", "Duplicate outputs (%)", r"$D=1-\frac{\text{unique complete continuations}}{256}$"),
        ("05_entropy", "token_entropy_nats", "Empirical entropy (nats)", r"$H=-\sum_t \widehat p(t)\ln\widehat p(t),\quad \widehat p(t)=\frac{\mathrm{count}(t)}{16{,}384}$")):
        scale = 1 if key == "token_entropy_nats" else 100
        fig, axes = plt.subplots(1, 3)
        for ax, rr, title, col in ((axes[0], w, "A  Week 1", BLUE), (axes[1], c, "B  Continued", ORANGE)):
            ax.plot(range(11), vals(rr, key, scale), "o-", color=col, ms=2.7)
            decorate(ax, title, ylabel)
        match(axes[2], key, scale)
        decorate(axes[2], "C  Shared start", ylabel)
        axes[2].legend()
        if key == "duplicate_continuation_rate":
            for ax in axes:
                ax.set_ylim(-.25, 8)
        elif key == "repeated_4gram_rate":
            for ax in axes:
                ax.set_ylim(37, 62)
        else:
            for ax in axes:
                ax.set_ylim(4.8, 6.4)
        save(fig, name, formula)

    wref = [js(f"reports/{W1}/records/seed_42/gen_{i:02}/metrics.json")["frozen_g0_reference"]["loss"] for i in range(11)]
    fig, axes = plt.subplots(1, 3)
    axes[0].plot(range(11), wref, "o-", color=BLUE, ms=2.7)
    decorate(axes[0], "A  Week 1\nFrozen G0", "Mean output NLL")
    axes[1].plot(range(11), vals(c, "frozen_s0_nll"), "o-", color=ORANGE, ms=2.7)
    decorate(axes[1], "B  Continued\nFrozen S0", "Mean output NLL")
    match(axes[2], "frozen_s0_nll")
    decorate(axes[2], "C  Shared start\nFrozen S0", "Mean output NLL")
    axes[2].legend()
    for ax in axes:
        ax.set_ylim(1.3, 2.75)
    save(fig, "06_reference", r"$L_{\rm ref}(g)=-\frac{1}{256\times63}\sum_{j=1}^{256}\sum_{t=2}^{64}\ln p_{\rm ref}(y^{(g)}_{j,t}\mid y^{(g)}_{j,<t})$", height=2.95, footer=.25)

    seq = {i: np.asarray(js(f"reports/{CONT}/records/seed_42/gen_{i:02}/test_sequences.json")["sequence_mean_nll"]) for i in (0, 1, 10)}
    av = np.asarray(js(f"reports/{RESET}/records/seed_42/gen_10/test_sequences.json")["sequence_mean_nll"])
    assert all(len(v) == 4494 for v in [*seq.values(), av])
    fig, axes = plt.subplots(1, 2)
    bins = np.linspace(1, 8, 36)
    for i, col in ((0, GREY), (1, BLUE), (10, ORANGE)):
        axes[0].hist(seq[i], bins=bins, histtype="step", density=True, color=col, label=f"S{i}", lw=1.5)
    axes[0].set(title="A  Real-test loss histogram", xlabel="Mean block NLL", ylabel="Density", xlim=(1, 7))
    axes[0].legend()
    for arr, col, label in ((av, PURPLE, "Reset A10"), (seq[10], ORANGE, "Continued B10")):
        axes[1].plot(np.sort(arr), np.arange(1, 4495)/4494*100, color=col, label=label)
    axes[1].set(title="B  Cumulative final block losses", xlabel="Mean block NLL", ylabel="Test blocks at or below loss (%)", xlim=(1, 7))
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    save(fig, "07_distributions", r"$\ell_b=-\frac{1}{63}\sum_{t=2}^{64}\ln p_\theta(x_{b,t}\mid x_{b,<t}),\quad P95=Q_{0.95}(\ell_b)$", height=2.8)

    fig, axes = plt.subplots(1, 3)
    for i, col in ((1, BLUE), (5, "#43855d"), (10, ORANGE)):
        hh = js(f"reports/{CONT}/records/seed_42/gen_{i:02}/training_history.json")
        axes[0].plot([r["epoch"] for r in hh], [r["validation_loss"] for r in hh], "o-", ms=2.5, color=col, label=f"S{i}")
        best = min(hh, key=lambda r: r["validation_loss"])
        axes[0].scatter(best["epoch"], best["validation_loss"], marker="*", s=55, color=col, zorder=4)
    axes[0].set_title("A  Continued: selected cycles")
    axes[0].legend()
    for ax, cycle in zip(axes[1:], (2, 10)):
        for arm, label, col in (("continued", "B", ORANGE), ("reset_s0", "A", PURPLE)):
            hh = [r for r in hist if r["arm"] == arm and int(r["cycle"]) == cycle]
            assert len(hh) == 5
            ax.plot([int(r["epoch"]) for r in hh], [float(r["validation_loss"]) for r in hh], "o-", ms=2.5, color=col, label=f"{label}{cycle}")
            best = min(hh, key=lambda r: float(r["validation_loss"]))
            ax.scatter(int(best["epoch"]), float(best["validation_loss"]), marker="*", s=55, color=col, zorder=4)
        ax.set_title(f"{'B' if cycle == 2 else 'C'}  Matched cycle {cycle}")
        ax.legend()
    for ax in axes:
        ax.set(xlabel="Epoch within one cycle", ylabel="Real-validation NLL", xticks=range(1, 6))
        ax.grid(alpha=.2)
    save(fig, "08_selection", r"$e^*=\underset{e\in\{1,2,3,4,5\}}{\operatorname{argmin}}\ L_{\rm validation}(\theta_e)$")

    reset_new = [r for r in a if r["stage_origin"] == "executed"]
    train_h = [sum(vals(w, "train_seconds"))/3600, sum(vals(c, "train_seconds"))/3600,
               sum(vals(reset_new, "new_branch_training_seconds"))/3600]
    gen_h = [sum(vals(w, "generation_seconds"))/3600, sum(vals(c, "generation_seconds"))/3600,
             sum(vals(reset_new, "new_branch_generation_seconds"))/3600]
    e1 = js(f"reports/{W1}/execution_receipt.json")
    e2 = js(f"reports/{CONT}/execution_receipt.json")
    source(f"reports/{RESET}/report.md")
    wall = [e1["hours_from_provisioning_to_completion"], e2["hours_from_provisioning"], 5.62]
    debit = [e1["observed_balance_decrease_at_previous_status_check_usd"], e2["observed_account_debit_usd"], 3.36]
    fig, axes = plt.subplots(1, 3)
    labels = ["Week 1", "Continued", "Reset new"]
    xx = np.arange(3)
    axes[0].bar(xx, train_h, color=BLUE, label="Training records")
    axes[0].bar(xx, gen_h, bottom=train_h, color="#a7c4cf", label="Generation records")
    axes[0].plot(xx, wall, "kD", ms=4, label="Provision-to-finish")
    axes[0].set(title="A  Time accounting", ylabel="Hours", ylim=(0, 10))
    axes[0].legend(fontsize=6.5)
    axes[1].bar(xx, [55, 50, 45], color=[BLUE, ORANGE, PURPLE])
    axes[1].set(title="B  New epochs executed", ylabel="Epochs", ylim=(0, 65))
    axes[2].bar(xx, debit, color=[BLUE, ORANGE, PURPLE])
    axes[2].set(title="C  Observed account debit", ylabel="USD incl. storage", ylim=(0, 4.6))
    for i, value in enumerate(debit):
        axes[2].text(i, value+.1, f"${value:.2f}", ha="center", fontsize=8)
    for ax in axes:
        ax.set_xticks(xx, labels, rotation=16)
        ax.grid(axis="y", alpha=.2)
    save(fig, "09_resources", height=2.35)

    fig, axes = plt.subplots(1, 3)
    rr = [c[0], a[-1], b[-1]]
    for ax, key, scale, title, unit in zip(axes,
            ("test_perplexity", "repeated_4gram_rate", "token_entropy_nats"),
            (1, 100, 1), ("A  Real-text prediction", "B  Phrase reuse", "C  Token concentration"),
            ("Perplexity", "Repeated 4-grams (%)", "Entropy (nats)")):
        vv = vals(rr, key, scale)
        ax.bar(range(3), vv, color=[GREY, PURPLE, ORANGE])
        ax.set(title=title, ylabel=unit, xticks=range(3), xticklabels=["S0", "Reset A10", "Continue B10"])
        ax.tick_params(axis="x", labelrotation=18)
        for i, value in enumerate(vv):
            ax.text(i, value+.025*max(vv), f"{value:.2f}", ha="center", fontsize=8)
        ax.set_ylim(0, max(vv)*1.2)
        ax.grid(axis="y", alpha=.2)
    save(fig, "10_synthesis", height=2.3)

    pages = []

    def page(title, image, caption, items, subtitle=None):
        pages.append({"title": title, "image": image, "caption": caption,
                      "subtitle": subtitle, "items": items})

    def body(t): return {"kind": "body", "text": t}
    def bullet(t): return {"kind": "bullet", "text": t}
    def label(t): return {"kind": "label", "text": t}

    page("Understanding our model-collapse graphs", "00_protocols",
         "Reading guide to the completed OPT-125M / WikiText-2 experiments. No new training was run for this guide.", [
        body("Two questions run through the report: can a model still predict genuine text, and what happens to the text it generates? The figures measure both, because the answers can differ."),
        bullet("Cycle 0: the real-data-trained reference, called G0 in Week 1 and S0 in Week 2. It is different from untouched pretrained OPT."),
        bullet("Shared beginning: the Week 2 reset/continued arms reuse the exact same S0, S1 and D2. Grey means one shared measurement pair; the arms first differ at cycle 2."),
        bullet("Fair comparisons: Week 1 and continued training have different first-step initializers and used different GPU classes. The shared-start pair matches the beginning and recorded primary environment."),
        bullet("Fixed measurements: all models see the same real validation/test blocks. Output statistics use the first 256 fixed training prompts, each producing exactly 64 new tokens."),
        bullet("Tokens: words, parts of words, punctuation or special symbols. Output statistics use token IDs, including special tokens; a 64-token output is not necessarily 64 words."),
        body("Read each section in this order: inspect the graph, read its equation, then connect the intuitive example to our measured results. All logarithms are natural; probability-based losses and entropy use nats."),
    ], subtitle="Graph families, equations, intuition and measured results | 28 October 2026")

    page("1. Real-text loss and perplexity", "01_perplexity",
         "A/B: fixed real validation and test sets. C: matched test comparison. The shared prefix is reused; the untouched pretrained value 115.43 is omitted from these axes.", [
        body("Question: how much probability does the current model assign to the next token actually present in held-out real WikiText? Evaluation supplies the correct real prefix at every position."),
        bullet("Equation: x_i is an actual next token, p_theta is the current model and N is the number of scored targets. NLL L averages negative log-probability; perplexity is exp(L). Lower values mean better real-text prediction."),
        bullet("Intuition: assigning probability 0.1 to the correct token costs -ln(0.1) = 2.303 nats; probability 0.01 costs 4.605. Confident mistakes receive a larger penalty."),
        bullet("Perplexity scale: PPL 50 corresponds to a geometric mean correct-token probability of 1/50, or 2%. It is not an accuracy percentage. Loss and perplexity are two scales for the same measurement."),
        bullet("Our curves: Week 1 goes 33.88 -> 62.49 -> 52.83. Continued goes 33.89 -> 48.38, recovers to 42.72 at S3, then reaches 55.56. Final reset is 42.10, versus continued 55.56."),
        bullet("Validation versus test: validation chooses the lowest-loss checkpoint; test reports its performance. Neither split is used as training targets. Corpus PPL is exp(total NLL / scored targets), never an average of batch perplexities."),
        body("A plateau or dip does not contradict increased output repetition: predicting real next tokens and generating a whole continuation are different measurements. These trajectories alone do not identify the mechanism causing the dip."),
    ])

    page("2. Distinct-n: variety of token combinations", "02_distinct",
         "A/B show distinct-1/2/3 within each historical condition. C compares distinct-2 after the identical Week 2 beginning. Higher means more unique combinations per occurrence.", [
        body("Question: how many different individual tokens, pairs or triples occur in the fixed output sample? An n-gram is n consecutive tokens."),
        bullet("Equation: U_n is the number of unique n-gram types and T_n is their total number of occurrences. We form n-grams within each continuation, then pool counts across outputs."),
        bullet("Toy example: A B A B has distinct-1 = 2/4 = 50%; its pairs AB, BA, AB give distinct-2 = 2/3 = 66.7%; its triples ABA, BAB give distinct-3 = 2/2 = 100%."),
        bullet("Read the lines: compare each n with the same n across cycles. Longer sequences have more possible combinations, so distinct-3 naturally tends to exceed distinct-1."),
        bullet("Our results: Week 1 distinct-3 falls 42.83% -> 38.15%, although distinct-2 rises 29.23% -> 30.58%. Continued finishes above S0 on all three ratios; its distinct-2 still falls 19.6% from the S2 peak."),
        bullet("Matched final models: reset distinct-2 is 31.05%, versus continued 34.71%. Reset distinct-3 is 39.02%, versus S0 42.66%. Diversity changes are therefore mixed."),
        body("Interpretation limit: these ratios measure token-combination variety, not semantic variety or factual accuracy. Sample sizes are fixed because unique/total ratios depend on output length and sample size."),
    ])

    page("3. Repeated four-token phrases", "03_repetition",
         "The same 15,616 within-continuation four-gram positions are counted at every model. Higher values mean more occurrences beyond the first appearance of each four-gram.", [
        body("Question: how much of the generated sample reuses four-token combinations that have already appeared?"),
        bullet("Equation: T_4 counts all four-gram occurrences; U_4 counts unique four-gram types. Subtracting unique types leaves the excess occurrences after each first appearance."),
        bullet("Toy example: ten four-gram occurrences containing six unique types give (10 - 6)/10 = 40% repetition."),
        bullet("Scope: a phrase can repeat inside one output or across different outputs. No four-gram crosses an output boundary. Shared Wikipedia-style language can contribute even without a visible looping sentence."),
        bullet("Our results: Week 1 rises 48.87% -> 59.05%; continued rises 48.55% -> 52.36%. The matched reset model finishes at 57.66%, above continued's 52.36%."),
        bullet("Direction: rising repetition can indicate concentrated output, but neither every repeat nor every fluctuation establishes collapse. The initial real-trained model already has substantial phrase reuse."),
        body("A 57.66% rate does not mean 57.66% of sentences are identical. This is a four-token occurrence statistic. It equals one minus distinct-4 under the same counting rule; we report distinct-1/2/3 separately."),
    ])

    page("4. Exact duplicate continuations", "04_duplicates",
         "Each curve compares 256 complete 64-token outputs generated from the fixed prompt set. Exact equality is tested on token IDs.", [
        body("Question: do different prompts lead the model to produce exactly the same complete continuation? This is a stricter comparison than shared phrases."),
        bullet("Equation: subtract the number of unique continuation rows from 256 and divide by 256. The result counts extra copies after the first occurrence."),
        bullet("Toy example: ten outputs with eight unique rows contain two excess copies, so the duplicate rate is 20%. One pair among 256 contributes one extra copy: 1/256 = 0.39%."),
        bullet("Our results: Week 1 ends at 13/256 excess copies (5.08%). Continued ends at 7/256 (2.73%). The shared-start reset model ends at 15/256 (5.86%)."),
        bullet("Intuition: convergence to the same complete output across prompts suggests reduced responsiveness or output variety. In this experiment deterministic five-beam decoding is held fixed."),
        body("Interpretation limit: near-identical outputs differing by one token count as distinct. This metric can miss heavy repetition inside each output, so it should be read alongside repeated four-grams and examples."),
    ])

    page("5. Empirical token entropy", "05_entropy",
         "Entropy is calculated from observed token counts in the 16,384-token output probe. The vertical axis is nats, not a percentage.", [
        body("Question: is token usage spread across many tokens, or dominated by a small set? Entropy incorporates token frequencies as well as which token types appear."),
        bullet("Equation: p-hat(t) is token t's observed frequency divided by the total token count. H averages the surprisal -ln p-hat(t). Lower entropy means a more concentrated empirical distribution."),
        bullet("Toy example: two equally frequent tokens have H = ln(2) = 0.693 nats. If one token appears 99% of the time, H is only about 0.056 nats, despite having the same two-token vocabulary."),
        bullet("Difference from distinct-1: distinct-1 counts unique types relative to sample size. Entropy also notices whether a few of those types dominate almost all occurrences."),
        bullet("Our results: Week 1 moves 5.772 -> 5.675. Continued ends at 5.745, close to S0's 5.736. Reset falls to 5.125, showing stronger concentration in token use."),
        body("This is observed-output entropy, not the entropy of the model's predicted next-token probabilities. Its exponential is therefore not our real-test perplexity. High entropy can also come from incoherent text; it is not automatically a quality improvement."),
    ])

    page("6. Outputs scored by the frozen real-trained model", "06_reference",
         "Week 1 uses frozen G0; Week 2 uses frozen S0. Within each panel the reference weights are held fixed while the generated texts change.", [
        body("Question: how predictable are later outputs under the original real-data-trained reference model? Here the model doing the scoring stays fixed, while the text being scored changes."),
        bullet("Equation: y is a saved continuation from cycle g. The fixed reference scores targets 2-64 of each of 256 continuations: 16,128 targets. The original prompt is excluded."),
        bullet("Direction: a larger score means the generated continuations are less predictable under the reference; a smaller score means they are more predictable. Neither direction is a stand-alone quality verdict."),
        bullet("Our results: continued rises 1.446 -> 2.343 nats per target. Final reset scores 2.127. Reset output is more predictable under S0 than continued output, while also being more repetitive on several measures."),
        bullet("Compare with real-test PPL: real evaluation changes the scoring model but fixes the text. Frozen-reference scoring fixes the model but changes the text. These experiments answer different questions."),
        body("The saved reference-score histograms show the spread of these scores across continuations, instead of just their mean: rightward means less predictable. The score is cross-entropy-like and depends on the generated text's own uncertainty as well as mismatch; it is not KL divergence, factuality or prompt relevance."),
    ])

    page("7. Passage losses, distributions and P95", "07_distributions",
         "A: continued S0/S1/S10 real-test block losses, using common bins. B: final reset/continued cumulative losses on the same 4,494 blocks. The Week 1 histogram uses the same definition.", [
        body("Question: does an average hide which passages get harder? Each separately scored 64-token test block has 63 scored next-token targets. Its mean NLL is one observation in these plots."),
        bullet("Histogram A: the horizontal axis is block loss; the vertical axis is density. Area, rather than bar height alone, represents a fraction of passages. A rightward shift indicates higher losses."),
        bullet("Cumulative curve B: at any loss threshold, the height is the percentage of blocks at or below it. A curve further left generally indicates lower losses. A height of 80% means 80% of blocks have loss no greater than that x-value."),
        bullet("Median and P95: the median is the middle block loss. P95 is the 95th percentile: approximately 95% of blocks have loss at or below it. Final P95 is 4.629 for reset versus 4.998 for continued."),
        bullet("Paired result: reset has lower loss on 4,340/4,494 blocks (96.57%). Its mean test NLL is 3.740 versus 4.017, a 0.277-nat difference. The 96.57% is a descriptive passage fraction, not a confidence level."),
        body("Mean passage PPL averages exp(block loss); corpus PPL exponentiates the overall mean loss. They differ because exponentiation is nonlinear. Continued S10 has corpus PPL 55.56, but passage-PPL mean 67.50 and SD 44.79. This SD describes passage variation, not uncertainty across training runs. High-loss blocks do not by themselves identify rare-token loss."),
    ])

    page("8. Epochs, validation and checkpoint selection", "08_selection",
         "Horizontal axes now count epochs within a cycle, not recursive cycles. Stars identify the lowest real-validation-loss checkpoint among epochs 1-5.", [
        body("Question: which checkpoint should produce the next dataset, and what happens if synthetic training continues for more epochs? An epoch is one pass over that cycle's training dataset."),
        bullet("Selection equation: e-star is the epoch with the smallest fixed real validation NLL. Validation chooses the checkpoint; test data never make that choice. The unmodified starting checkpoint is not among the candidates."),
        bullet("Read the curves: in these cycles real validation loss rises after epoch 1, so the first epoch is selected. Further fitting of synthetic targets can worsen prediction on real text."),
        bullet("Executed versus inherited: continued training executed 50 epochs, but S10 inherits the ten selected synthetic epochs after S0. The reset branch executed 45 new epochs, but A10 starts from S0 and retains its own selected cycle-10 epoch. Its data still carry earlier recursive history."),
        bullet("Training loss: the mean negative log-probability on the current training targets. It can fall while real validation loss rises. Training datasets change between cycles, so their losses also reflect different target difficulty."),
        body("Fresh optimizer: both Week 2 arms restart AdamW and the learning-rate scheduler every cycle. Continued weights do not mean continued optimizer state. Epoch 1 was selected in every synthetic cycle of the continued and new reset arms."),
    ])

    page("9. Data size, runtime and resource statistics", "09_resources",
         "New work only: Week 1 trains G0-G10; continued imports S0; reset imports S0/S1 and runs A2-A10. Reset hours/debit are rounded report values. Debit includes storage and differing observation windows.", [
        body("Question: did the data budget stay constant, and where did the computation go? These are accounting measurements that make the experiment interpretable and reproducible."),
        bullet("Equal dataset size: every full training dataset has 37,946 blocks x 64 = 2,428,544 input tokens. There are 37,946 x 63 = 2,390,598 prediction targets per epoch. Prompts are excluded from synthetic training targets."),
        bullet("Fixed evaluation/probe sizes: validation scores 247,086 targets; test scores 283,122. The output probe has 256 x 64 = 16,384 tokens and 15,616 within-output four-gram positions. It is a fixed contiguous prompt slice, not a random full-corpus sample."),
        bullet("Runtime A: bars sum recorded new training/generation durations; diamonds show provisioning-to-completion time. Gaps include other work and accounting scope. Five-beam generation dominates these runs. The final model generates only a 256-output probe, not a full next training dataset."),
        bullet("Epochs B: 55, 50 and 45 new executed epochs respectively. Shared checkpoints and datasets retain their history but are not counted as newly executed work."),
        bullet("Cost C: about $2.23, $3.99 and $3.36 observed account debits. Hardware, work scopes and retained-storage windows differ; these are not itemized compute invoices or matched efficiency comparisons."),
        body("Throughput means scored training targets per second or generated continuations per second under a stated workload. Peak allocated GPU bytes measure the training process's allocator peak, not total machine VRAM. Forecast plots are benchmark-based estimates, separate from actual completed-run time and debit."),
    ])

    page("10. Reading the measurements together", "10_synthesis",
         "Same Week 2 reference and final models, viewed through three different measurements. Prediction, phrase reuse and token-frequency spread do not rank the models identically.", [
        bullet("Real-text prediction: reset A10 has lower PPL than continued B10, but both remain worse than S0. The advantage is broad across the fixed test blocks."),
        bullet("Generated output: reset has more repeated four-grams, more exact duplicates and lower entropy than continued. Its real-test advantage does not establish preserved output diversity."),
        bullet("Mixed metrics: continued ends above S0 on distinct-1/2/3 and near S0 in entropy, even though real-text PPL worsens. One example of a repeated sentence cannot establish a uniform vocabulary collapse."),
        bullet("Protocol scope: fixed real prompts, deterministic five-beam decoding, real-validation checkpoint selection and one seeded chain per condition. Passage variation does not supply independent training replicates."),
        body("Original report map: Week 1 A-F correspond to Sections 1, 2, 3, 4, 5 and 7 here. Standalone Week 2 A-F correspond to Sections 1, 2, 3/4, 5, 6 and 8. Shared-start A-F correspond to Sections 1, 2, 3, 4, 5 and 6. The final two panels correspond to Sections 7 and 8."),
        label("Sources and exact definitions"),
        {"kind": "link", "text": "Project report: methodology, tables and saved-text examples", "url": "https://docs.google.com/document/d/1WyQia9SBCOtT9_X4X56BpTihz-X9q9LWZFVapFEB4uk"},
        {"kind": "link", "text": "Reproducible metric definitions and implementation", "repo_path": "docs/metrics.md"},
        {"kind": "link", "text": "Week 1 / continued / matched-pair evidence and figures", "repo_path": "experiments/catalog.yaml"},
    ])

    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    for p in pages:
        for item in p["items"]:
            if "repo_path" in item:
                item["url"] = f"https://github.com/BillLeoutsakosvl346/Model_Collapse_in_Synthetic_Data_Training/blob/{revision}/{item.pop('repo_path')}"
    atomic_json(out / "guide.json", {"title": "Understanding our model-collapse graphs", "pages": pages, "figures": figures})
    atomic_json(out / "manifest.json", {
        "scope": "Illustrated explanation; replotting immutable records only, no new scientific experiment.",
        "analysis_revision": revision, "script_sha256": sha256(__file__), "inputs": used,
        "environment": {"python": platform.python_version(), **{k: importlib.metadata.version(k) for k in ("numpy", "matplotlib")}},
        "outputs": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()},
        "checks": {"eleven_models_per_condition": True, "exact_shared_s0_s1_measurements": True, "4494_blocks_per_distribution": True},
    })
    print(json.dumps({"pages_planned": len(pages), "figures": len(figures), "input_files": len(used),
                      "words": sum(len(i["text"].split()) for p in pages for i in p["items"]), "output": str(out)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    build(parser.parse_args().out.resolve())
