"""Build the authored Week 3 report and a native-Docs editing draft.

Requires the frozen selective audit and paired analysis; no network or training.
"""
from pathlib import Path
import csv
import hashlib
import json
import re

PROJECT = Path(__file__).resolve().parents[2]
PAIR = "20261030T180755Z-full-paired-final5"
ROOT = PROJECT / "reports" / PAIR
RESULTS = ROOT / "analysis/results"
AUDIT = ROOT / "analysis/audit"
PRIVATE = PROJECT / ".state/week3_doc"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build():
    summary = read(RESULTS / "summary.json")
    audit = read(AUDIT / "audit.json")
    rows = list(csv.DictReader((AUDIT / "metrics.csv").open(encoding="utf-8")))
    assert audit["checks_passed"] and audit["unique_stages"] == 19
    assert audit["week2_vs_week3_cycle1"]["five_epoch_training_validation_trajectory_exact"]
    columns = [next(r for r in rows if r["arm"] == arm and int(r["cycle"]) == cycle)
               for arm, cycle in [("continued", 0), ("continued", 1), ("reset", 10), ("continued", 10)]]

    def vals(key, fmt=".2f", scale=1):
        return [format(float(r[key]) * scale, fmt) for r in columns]

    entries = []

    def add(identifier, text, kind="body", page=False):
        entries.append({"id": identifier, "text": text, "kind": kind, "page": page})

    add("week3", "Week 3: Matched training with final-epoch checkpoints", "heading1", True)
    add("meta", f"OPT-125M / WikiText-2; seed 42. Paired experiment {PAIR}. Completed 31 October 2026.", "small")
    add("methods", "1. Objective and methodology", "heading2")
    add("objective", "We repeated the shared-start reset-versus-continue comparison, changing checkpoint selection: every synthetic stage now retains epoch 5. Historical Week 2 also executed five epochs, but selected epoch 1 using real validation loss. This is a checkpoint-policy comparison, not one executed epoch versus five.")
    add("same", "Shared beginning: reuse the exact historical real-trained S0 and its synthetic D1. Train S0 on D1 for five epochs to obtain a NEW shared S1, then generate NEW shared D2.", "bullet")
    add("branch", "First divergence: both cycle-2 models learn from the same D2. Reset A2 initializes from S0; continued B2 initializes from S1. From cycle 3 onward, each arm generates its own next dataset.", "bullet")
    add("carry", "Weight inheritance: B10 carries 50 selected synthetic epochs after S0. A10 carries its own five epochs from S0; earlier A-model updates survive through synthetic data, not inherited weights.", "bullet")
    add("protocol", "Recursive protocol", "heading3")
    add("loop", "Shared: S1 = train5(S0, historical D1); D2 = generate(S1)\nReset: A2 = train5(S0, D2); A(k+1) = train5(S0, generate(Ak))\nContinue: B2 = train5(S1, D2); B(k+1) = train5(Bk, generate(Bk))", "code")
    add("protocol_caption", "Table 12. Matched settings; retain the fifth checkpoint even when real validation loss worsens.", "caption")
    add("protocol_table", "[TABLE_protocol]", "table")
    add("budget", "Equal data: each dataset contains 37,946 continuation-only blocks, or 2,428,544 input tokens. Each epoch scores 2,390,598 shifted targets. Across 19 unique stages / 95 epochs: 230,711,680 input-token exposures and 227,106,810 target exposures.", "bullet")

    add("trajectories", "2. Results from the shared starting point", "heading2", True)
    add("metrics_image", "[FIGURE_metrics]", "image")
    add("metrics_caption", "Figure 6. Grey squares are the single shared S0/S1 pair; purple resets to S0 (A), orange continues weights (B). Branches diverge at cycle 2. Output statistics use the same first 256 prompts; formatting is included. Frozen-S0 scores omit the generation prompt. One seeded pair; no independent-run error bars. F shows mean/P95 across 256 continuations, each scored on 63 shifted targets.", "caption")
    add("key_heading", "Key findings", "heading3")
    add("ppl_path", "Real-test perplexity: common S0 -> S1 is 33.89 -> 54.46. Continued improves to 48.97 at B2, then rises to 107.37. Reset reaches 42.75 at A3 and ends at 44.50: 58.56% below continued, but 31.32% above S0.", "bullet")
    add("div_path", "Generated text: after an early diversity peak, continued develops lower entropy, more repeated four-grams and more duplicates than reset. These are raw token statistics; Section 5 identifies a strong newline effect.", "bullet")
    add("mixed", "Mixed baseline changes: continued distinct-2 is still 29.03%, slightly above S0's 28.81%, despite much worse PPL. Reset entropy and distinct-1/2/3 also remain above S0. Neither curve supports 'every metric deteriorates monotonically'.", "bullet")

    add("statistics", "3. Summary statistics and metric definitions", "heading2", True)
    add("metric_intro", "Prediction and generation answer different questions. Real-text NLL measures surprise on fixed WikiText targets; corpus PPL = exp(mean NLL). Output diversity measures the continuations produced for fixed real prompts, not predictive accuracy or factual correctness.")
    add("stats_caption", "Table 13. The same saved S0/S1 measurements and final models. Lower real-text loss/PPL is better; diversity statistics describe different, dependent aspects of output behaviour.", "caption")
    add("stats_table", "[TABLE_stats]", "table")
    add("definitions", "Definitions and interpretation", "heading3")
    add("real_def", "Real evaluation: fixed 4,494 test blocks / 283,122 shifted targets; validation uses 3,922 blocks / 247,086 targets. P95 is the 95th percentile of mean block NLL, not rare-token loss.", "bullet")
    add("pooled_def", "Pooled metrics: entropy uses token frequencies across 16,384 generated tokens. Distinct-2/3 divide unique windows by 16,128 / 15,872. Repeated-four-gram rate counts excess occurrences after the first across 15,616 windows, including phrases shared between outputs; it equals 1 - distinct-4.", "bullet")
    add("within_def", "Within-output metrics: compute entropy or repeated-four-gram rate within each 64-token continuation, then average. Duplicate counts are excess identical complete outputs after the first copy.", "bullet")

    add("selection", "4. Checkpoint selection and real-text deterioration", "heading2", True)
    add("selection_image", "[FIGURE_selection]", "image")
    add("selection_caption", "Figure 7. A: real-test block-loss distributions; B: continued-minus-reset loss for matched final test blocks. C/D: real-validation loss across epochs; stars retain epoch 5, hollow markers identify the minimum. Within-cycle curves and test blocks are not independent training replicates.", "caption")
    add("selection_findings", "What the diagnostics show", "heading3")
    add("coverage", "Coverage: reset has lower final NLL on 4,475/4,494 blocks (99.58%); the mean gap is 0.881 nats/target. Both remain worse than S0 on most blocks: 4,198 for reset and 4,465 for continued.", "bullet")
    add("overfit", "All 19 stages: the lowest real validation loss occurs at epoch 1, but epoch 5 is intentionally retained. In shared cycle 1, validation NLL rises from 3.892 to 4.007. Additional optimization on synthetic targets can reduce synthetic training loss while worsening prediction of real text.", "bullet")
    add("matched_old", "Cross-week control: historical and new cycle-1 train/validation losses match exactly at all five epochs with identical S0 and D1. Selecting epoch 5 instead of 1 creates the new S1 and subsequent data chain; historical S1/D2 were not reused.", "bullet")
    add("historic", "Historical context: final continued PPL is now 107.37 versus 55.56 under Week 2's selected-epoch-1 policy; reset is 44.50 versus 42.10. These are descriptive policy trajectories, not independent-seed significance tests.", "bullet")
    add("mechanism", "Possible explanation: continued weights repeatedly reinforce their own preferred synthetic distribution. Reset removes direct inheritance of earlier synthetic weight updates and may constrain real-text drift; its recursively generated data can still deteriorate. Neither mechanism is isolated by independent interventions here.", "bullet")

    add("formatting", "5. Why entropy and repetition need a formatting check", "heading2", True)
    add("formatting_image", "[FIGURE_formatting]", "image")
    add("formatting_caption", "Figure 8. Primary metrics include formatting. B/D dashed curves are post-hoc newline-exclusion diagnostics: entropy removes newline tokens and renormalizes; four-gram counts retain only original contiguous windows without a newline. Tokens are never joined into new windows. Denominators differ between arms, so dashed curves do not replace the raw comparison.", "caption")
    add("format_findings", "Measured findings", "heading3")
    add("newline_share", "Newline concentration: continued B10 has 6,327/16,384 newline tokens (38.62%), versus 459/16,384 (2.80%) for reset A10. Of 256 outputs, 111 continued outputs are at least half newlines and ten contain only newlines; reset has none in either category.", "bullet")
    add("raw_local", "Raw within-output behaviour: continued entropy is 2.051 versus reset 2.935 nats, and repeated-four-gram rate is 57.03% versus 43.44%. The aggregate gap therefore also appears within responses, but blank-line loops contribute to it.", "bullet")
    add("sensitivity", "Sensitivity: excluding newlines reverses pooled entropy to 6.132 continued versus 5.994 reset. Repeated-four-gram rates on retained original windows become 35.99% versus 50.27%, with 9,286 versus 14,573 windows. These describe unequal retained samples, not improved output quality after filtering.", "bullet")
    add("reference", "Frozen-S0 score: final mean NLL is almost equal (2.238 continued; 2.233 reset) despite very different real-test PPL. The ten all-newline outputs score only 0.465 on average under S0. Familiar or easy text can be degenerate; this score is not a factuality, relevance or quality judge.", "bullet")
    add("format_meaning", "Interpretation: the raw continued probe is strongly concentrated on blank-line output, not necessarily a smaller substantive vocabulary after formatting is removed. This is a measured output failure; why this training trajectory favours newlines remains untested.", "bullet")

    add("examples", "6. Saved text examples, scope and reproducibility", "heading2", True)
    add("example_intro", "Use the same historical anchor, prompt 0, ending '...Valkyria Chronicles III outside'. These are complete saved 64-token continuations with display whitespace normalized. This prompt is fixed for continuity, not chosen as representative; generated claims are unverified.")
    add("samples_caption", "Table 14. Shared S1 and both final models for the same prompt. Both final examples loop, despite reset's better aggregate scores. A separate saved output (prompt 52) contains exactly 64 newline tokens under continued training.", "caption")
    add("samples_table", "[TABLE_samples]", "table")
    add("scope_heading", "Interpretation and limitations", "heading3")
    add("scope_examples", "Example versus population: continued has more within-output four-gram repetition on 142 prompts, reset on 76, with 38 ties. Prompt 0 is a tie on that metric; its passages cannot explain or validate the entire entropy curve.", "bullet")
    add("scope_data", "Probe coverage: the first 256 consecutive prompts cover six adjacent training articles, not a random full-corpus sample. Real-text evaluation covers the fixed full held-out split. One seed, OPT-125M, WikiText-2, five-beam deterministic decoding and fixed real-training prompts reused every cycle constrain generalization.", "bullet")
    add("scope_causal", "Conclusion: retaining final-epoch weights exposes substantially greater real-text deterioration in the continued trajectory. Reset limits that deterioration here, but retains repetitive outputs and does not establish that collapse has been prevented.", "bullet")
    add("generated", "Generation accounting: D1 is imported; 17 new full datasets contain 41,285,248 tokens. Each final model adds only a 256-output probe; no full D11 is generated. Shared work is counted once.", "bullet")
    add("operations", "Execution: 11.97 hours; GPU shutdown verified. Account debit through the completion check was $7.07, including concurrent storage and billing timing, not an itemized compute invoice.", "small")
    add("recovery", "Evidence: 259 selectively recovered files / 13,094,380 bytes, including 20 unique 256x64 probes / 1,313,280 bytes, were checksum-verified. Diversity, decoded examples and saved NLL/PPL arithmetic were recalculated; no new model forward pass or bulk checkpoint/full-dataset download was performed. Recovery volumes remain preserved.", "small")
    add("links", "Complete per-cycle statistics, all saved examples, analysis scripts and evidence audit.", "link")

    tables = {
        "protocol": [["Setting", "Both arms"],
                     ["Model / corpus", "Pinned facebook/opt-125m / WikiText-2 raw; existing real-trained S0"],
                     ["Training / selection", "Five full epochs; always retain epoch 5; no validation-based early stopping"],
                     ["Generation", "Fixed 64-token real prompts -> exactly 64 new tokens; five deterministic beams; repetition penalty 1.0; EOS suppressed"],
                     ["Optimization", "Fresh AdamW/scheduler each cycle; LR 2e-5, epoch cosine towards 1e-6; weight decay 0.01"],
                     ["Batch / precision", "Microbatch 32 x accumulation 4 = 128; generation/evaluation batch 32; BF16; seed 42"],
                     ["Environment", "One RTX PRO 4000 Blackwell; PyTorch 2.8.0+cu128, Transformers 4.37.2, Datasets 3.4.1"],
                     ["Evaluation", "Same real validation/test arrays throughout; validation diagnostic only; test never selects checkpoints"]],
        "stats": [["Measure", "Shared S0", "Shared S1", "Reset A10", "Continue B10"],
                  ["Test corpus perplexity", *vals("test_perplexity")],
                  ["Validation perplexity", *vals("validation_perplexity")],
                  ["Mean test NLL (nats)", *vals("test_loss", ".3f")],
                  ["P95 test-block NLL", *vals("test_block_nll_p95", ".3f")],
                  ["Distinct-1 (%)", *vals("distinct_1", scale=100)],
                  ["Distinct-2 (%)", *vals("distinct_2", scale=100)],
                  ["Distinct-3 (%)", *vals("distinct_3", scale=100)],
                  ["Pooled repeated 4-grams (%)", *vals("repeated_4gram_rate", scale=100)],
                  ["Mean within-output repeated 4-grams (%)", *vals("mean_within_repeated_4gram_rate", scale=100)],
                  ["Duplicate outputs / 256", *[str(int(r["duplicate_outputs"])) for r in columns]],
                  ["Pooled token entropy (nats)", *vals("token_entropy_nats", ".3f")],
                  ["Mean within-output entropy", *vals("mean_within_entropy_nats", ".3f")],
                  ["Frozen-S0 mean NLL", *vals("reference_nll_mean", ".3f")],
                  ["Frozen-S0 P95 NLL", *vals("reference_nll_p95", ".3f")]],
        "samples": [["Model", "Complete saved continuation"]]
    }
    for label, arm, g in [("Shared S1", "continued", 1), ("Reset A10", "reset", 10), ("Continue B10", "continued", 10)]:
        text = read(ROOT / f"records/{arm}/seed_42/gen_{g:02d}/samples.json")[0]["continuation"]
        tables["samples"].append([label, re.sub(r"\s+", " ", text).strip()])

    figure_names = {"metrics": "recursive_metrics.png", "selection": "generalization_and_selection.png", "formatting": "formatting_diagnostics.png"}
    images = {key: str((RESULTS / "figures" / name).resolve()) for key, name in figure_names.items()}
    assert all(Path(path).exists() for path in images.values())
    source_url = f"https://github.com/BillLeoutsakosvl346/Model_Collapse_in_Synthetic_Data_Training/blob/main/reports/{PAIR}/report.md"
    draft = {"entries": entries, "tables": tables, "images": images, "source_url": source_url}
    PRIVATE.mkdir(parents=True, exist_ok=True)
    (PRIVATE / "draft.json").write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = []
    for e in entries:
        kind, identifier, text = e["kind"], e["id"], e["text"]
        if kind == "table":
            t = tables[identifier.removesuffix("_table")]
            lines += ["| " + " | ".join(t[0]) + " |", "| " + " | ".join(["---"] * len(t[0])) + " |"]
            lines += ["| " + " | ".join(row) + " |" for row in t[1:]]
        elif kind == "image":
            key = identifier.removesuffix("_image")
            lines += [f"![Week 3 {key}](analysis/results/figures/{figure_names[key]})"]
        elif kind == "link":
            lines += ["[Full metrics](analysis/results/metrics.csv) | [Saved examples](analysis/results/examples.json) | [Epoch histories](analysis/results/training_history.csv) | [Evidence audit](analysis/audit/audit.json) | [Analysis manifest](analysis/results/analysis_manifest.json) | [Reproducible analysis](../../comparisons/shared_start_final_epoch5/analyse.py) | [Evidence recovery](../../comparisons/shared_start_final_epoch5/analyse_evidence.py)"]
        elif kind == "code":
            lines += ["```text", text, "```"]
        elif kind == "bullet":
            lines += ["- " + text]
        else:
            prefix = {"heading1": "# ", "heading2": "## ", "heading3": "### "}.get(kind, "")
            lines += [prefix + text]
        lines += [""]
    (ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"entries": len(entries), "tables": len(tables), "figures": len(images),
                      "prose_words": sum(len(e["text"].split()) for e in entries if e["kind"] not in ("image", "table")),
                      "audit_passed": audit["checks_passed"], "summary_pair": summary["pair_id"]}))


if __name__ == "__main__":
    build()
