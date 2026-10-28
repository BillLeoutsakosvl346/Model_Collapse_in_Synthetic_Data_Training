# Graph, metric and experimental-protocol audit — 28 October 2026

**Conclusion:** independent recalculation found no numerical error in the saved metrics or their plotted data. The entropy graph correctly reports empirical token entropy on the fixed output probe. Repeated WikiText formatting tokens explain much of the final reset-versus-continued entropy difference. The audit found interpretation, presentation and reproducibility issues that should be corrected without replacing the original measurements.

Five reviewers started with fresh context and separate responsibilities: metric arithmetic; archived execution and lineage; plots and embedded figures; scientific interpretation; and prompt sampling/output diagnostics. The coordinator read both live Google Docs, checked historical-file preservation and reran the offline test suite. Reviewers were asked to challenge existing audit conclusions rather than treat them as verification.

## Scope and verification

Audited repository revision: `f44962f5f979b821fd4c378aebaab1fc4f7ccfc4`.

| Condition | Completed run | Initializer for a new synthetic stage |
| --- | --- | --- |
| Original pretrained reset | `20261020T211121Z-full` | Original pretrained OPT |
| Continued weights | `20261026T192740Z-full-continued` | Previous selected model |
| Shared-start reset to real-trained S0 | `20261027T211852Z-full-reset-s0` | Fixed S0; S0/S1/D2 imported from continued arm |

Checks included:

- **3,961 independent metric checks; zero numerical failures.** Recalculated 33 reported stage rows, including 31 distinct 256 × 64-token probes. These counts include shared-prefix references; they do not imply 33 independently executed stages.
- All 99 stage-level validation/test/frozen-reference NLL and perplexity pairs agree with saved sums and denominators. Independent per-block calculations reproduce passage statistics, including actual population SD of passage perplexity.
- Distinct-1/2/3, repeated four-grams and exact duplicates agree exactly. The largest entropy discrepancy is `1.78e-15` nats, from floating-point arithmetic.
- **30 actually executed training stages / 150 epoch records:** selected checkpoints match the minimum saved real-validation loss. This checks selection records, not fresh inference from model tensors.
- Archived runtime identities and initializer/generator/data handoffs agree with the stated conditions. The real train/validation/test arrays were independently reconstructed from the pinned public WikiText source and tokenizer, reproducing their file and token-content hashes exactly.
- Plot inputs, arm identities, generation indices, percentages and shared-prefix handling were checked. The 15 figures embedded in the two live Google Docs match their retained image assets byte for byte. Plot regeneration and visual-review details are in the private plot-audit receipt.
- **86 existing tests passed**, with 17 dependency deprecation warnings, in 58.46 seconds. No GPU was provisioned.
- All **312** files checked against the historical inventory and shared-start publication manifest remain byte-for-byte unchanged.

### Verification boundary

This audit independently checks saved score arrays, probe token IDs, source, checksums, selection histories and derived statistics. It did **not** download all model weights, run fresh model inference, or independently read every full synthetic training array. Exact shared S0/S1/D2 and subsequent full-dataset handoffs are supported by recorded content hashes, artifact indexes and historical worker checks, rather than a new bulk tensor verification. Consequently, this is not a claim that every full-run model output has been independently re-evaluated.

The current Week 1 six-panel figure's precise histogram recipe cannot be independently reconstructed without its missing authoring script. Its underlying saved scores were verified, but this source gap limits the figure-level conclusion. Historical forecast graphics were checked against the frozen forecast record; the original warmed-up benchmark-to-forecast derivation was not freshly revalidated. Live provider billing was outside this audit.

Missing small Week 1 evidence was recovered read-only from a pinned Hugging Face backup revision and checksum-verified. No paid experiment or cloud-resource mutation was performed. Both Google Docs and the historical results were left unchanged during the audit.

## Why the quoted example and entropy graph disagree

The quoted passage is fixed prompt index 0. Each such passage contributes only `64 / 16,384 = 0.390625%` of pooled token occurrences. It shows a real failure mode but was not selected as a statistically representative output.

| Final-output diagnostic | Reset A10 | Continued B10 |
| --- | ---: | ---: |
| Entropy of prompt 0 only, nats | 3.0604 | 2.0497 |
| Internal repeated-four-gram rate of prompt 0 | 36.07% | 68.85% |
| Pooled entropy of all 256 outputs, nats | 5.1245 | 5.7446 |
| Mean entropy calculated separately within each output, nats | 2.3821 | 2.7075 |
| Number of ` =` tokens, token ID 5457 | 3,739 | 1,803 |
| Share of all output tokens that are ` =` | 22.82% | 11.00% |
| Outputs with at least 32 of 64 tokens equal to that token | 64 / 256 | 25 / 256 |
| Pooled entropy after excluding only token 5457, nats | 5.9439 | 6.0655 |

For prompt 0, continued training is indeed more repetitive and has lower entropy. Across the full probe, reset has more concentrated token use. In particular, many reset outputs repeat WikiText heading markers. These are genuine generated tokens, not padding accidentally included by plotting. Saved decodes agree with their original token IDs.

Excluding that one token reduces the pooled entropy difference from **0.6201 to 0.1216 nats**. This is a post-hoc sensitivity diagnostic with renormalized frequencies and different retained token counts; it does not replace or redefine the primary historical metric, and it is not a semantic-quality score.

Formatting also affects phrase statistics. Counting only original contiguous four-token windows that contain no equality token gives diagnostic repeated-four-gram rates of **44.61% for reset and 45.22% for continued**, compared with the primary rates of **57.66% and 52.36%**. This exclusion changes the retained windows in each arm; no artificial n-grams were created by joining tokens across removed symbols. It demonstrates sensitivity of the ranking to formatting, rather than establishing a replacement primary comparison.

Entropy ignores token order. The same token counts in a different order have identical entropy. Repeated-four-gram counts add phrase information, but the published version pools counts across outputs and therefore includes repeated phrases shared by different outputs as well as loops within one output. These definitions are valid and were documented; they require clearer explanation beside the examples.

## Findings and recommended corrections

| Priority | Finding | Recommended correction |
| --- | --- | --- |
| Important interpretation | Prompt 0 reverses the aggregate ranking. The existing “illustrative, not representative” caveat is correct but does not resolve the apparent contradiction. | Explain the local-versus-pooled distinction beside the examples; show matched examples of both phrase loops and formatting loops, with a declared selection rule. |
| Important measurement scope | Output metrics use the first 256 consecutive real-training prompts, only 0.6746% of the prompt pool. They span six adjacent articles, not a random sample of WikiText. | Keep the original graphs, label their scope prominently, and describe conclusions as applying to this probe. A broader or stratified probe would be a separately declared supplementary analysis. |
| Important interpretation | The guide's “matched-block result supports broad deterioration” does not name the correct comparison. The cited 4,340/4,494 result compares final A10 against final B10. | Say “A10 has lower loss than B10 on 4,340/4,494 fixed test blocks.” Use separate S0 comparisons when discussing deterioration from baseline. |
| Scientific wording | Repeated “Likely causes” labels elevate plausible but unisolated mechanisms. Existing caveats reduce the risk, but do not establish those mechanisms. | Use “Possible explanations”; retain uncertainty about early dips, exact peaks and decoder/training interactions. |
| Scientific wording | “The initial real-to-synthetic transition produces the largest measured loss” is ambiguous about change versus level and causation. | Say “The largest increase in real-test loss occurs from G0 to G1.” |
| Presentation | The Week 1 main figure's distinct-n legend obscures the final distinct-2 marker. Some guide legends also overlap data. | Move legends outside data regions in a new figure revision; preserve the original artifact. |
| Reproducibility | The currently embedded Week 1 six-panel figure is retained only at ignored `.state/research_doc/metrics_panels.png`; a complete authoring script was not found in the audited checkout. Other retained plotting entry points regenerate their corresponding assets. | Retain a checked-in plotting script, input manifest and revised figure for that presentation asset. This is a reproducibility gap, not evidence of incorrect measured values. |
| Protocol explicitness | Enforcing exactly 64 generated tokens suppresses OPT's EOS/BOS token throughout the continuation. Real training includes 36,717 occurrences of ID 2; inspected generated probes contain none. | State the fixed-length/EOS policy explicitly. Treat its influence on the trajectories as untested; do not silently change it in an existing condition. |
| Reader clarity | Real scores at model Sk follow its training on Dk; output statistics describe Sk-generated D(k+1)-style text. | Add this one-cycle relationship near the guide's common-protocol diagram; the final point generates only a probe. No plotting-index defect was found. |

### Correct baseline comparisons

The separately recalculated matched-block counts are:

| Comparison | Blocks with higher final loss | Denominator |
| --- | ---: | ---: |
| Week 1 G10 versus its G0 | 4,025 | 4,494 |
| Continued B10 versus S0 | 4,377 | 4,494 |
| Reset A10 versus S0 | 4,072 | 4,494 |

These are descriptive comparisons of fixed test blocks, not independent training-run replicates or significance tests. The 4,340/4,494 A10-over-B10 improvement count is a different comparison.

## Disclosed limitations that are not newly discovered calculation bugs

- Empirical token entropy is correctly distinguished from entropy of model probabilities; its exponential is not real-test perplexity.
- All metrics count tokenizer units, including formatting when present, rather than semantic words or factual knowledge. High distinctness/entropy alone does not establish quality.
- Continued and reset arms share exact saved S0/S1/D2. From cycle 3 onward their synthetic datasets differ by design, so later differences are effects of the complete recursive policies, not a same-dataset-only weight experiment.
- The optimizer and scheduler restart in both arms. Five epochs execute per stage, but epoch 1 was selected for every new Week 2 synthetic stage. Executed epochs and inherited selected updates are already distinguished.
- One seeded trajectory per condition, original Week 1 GPU differences, real prompts in every cycle and real-validation checkpoint selection limit generalization and exact paper reproduction. No across-seed significance was established.
- P95 block loss is not a direct rare-event-loss measurement, and frozen-reference NLL is neither KL divergence nor a factuality/relevance score.

## Retained audit evidence

Machine-readable summary: [2026-10-28_graph_audit_summary.json](2026-10-28_graph_audit_summary.json).

Detailed local receipts, independent scripts and regenerated images are under ignored `.state/2026-10-28_graph-audit/`:

- `metrics/independent_results.json`, `independent_metrics.csv`, `paired_block_results.json`, `week1_recovery_receipt.json`.
- `execution/summary.json`, `audit_evidence.json`, `audit_execution.py` and pinned-source reconstruction receipts.
- `plots/` numerical, regeneration, native-image and visual-review receipts.
- `sampling/` per-cycle/per-output diagnostics, source coverage and diagnostic figure.
- `interpretation/findings.md`, both live native-document snapshots and `preservation_audit.json`.

These audit files are supplementary evidence. They do not supersede completed-run measurements or change the accepted training runtime.
