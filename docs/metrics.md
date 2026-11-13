# Metric definitions and interpretation

All comparisons must use matching tokenisation, context, masks, output counts and denominators. Raw measurements remain in the dated run reports and records; generated analysis must cite those inputs and hashes.

## Checkpoint-gradient diagnostics

Source gradients are means over scored continuation tokens. Their L2 norms describe magnitude, cosine alignment describes direction, and the actual 99% mixture is `0.01 * real_gradient + 0.99 * synthetic_gradient`. A norm ratio is not a percentage of causal influence. The same-prompt control replaces synthetic targets with the original real continuation at identical prompt rows.

Isolated diagnostic updates use complete-source mean gradients, FP32 and disabled dropout. They are not historical minibatch replays. Validation NLL changes measure immediate held-out effects; explicit zero-gradient AdamW steps control for carried momentum and weight decay. Paired document-bootstrap intervals describe validation-sample variability, not independent training seeds. Undefined cosine for a near-zero gradient remains null.

| Metric | Definition and unit | Interpretation limit |
| --- | --- | --- |
| Real validation/test NLL | Sum of next-token negative log-probabilities divided by the count of scored shifted targets; nats/token | Validation selects only under the best-validation policy; diagnostic under final-epoch selection. Test is reporting-only |
| Corpus perplexity | `exp(total NLL / total scored targets)` | Never average batch perplexities or substitute mean passage perplexity |
| Median / P95 block NLL | Median / 95th percentile of mean NLL on individual real 64-token blocks | High-loss blocks are not automatically annotated rare words/topics |
| Mean passage perplexity | Arithmetic mean of `exp(block mean NLL)` | Usually differs from corpus perplexity because exponentiation is nonlinear |
| Passage perplexity SD | Actual standard deviation across passage perplexities | Passage heterogeneity, not across-seed statistical uncertainty; not `exp(std(NLL))` |
| Distinct-1/2/3 | Unique n-grams divided by total n-gram occurrences, pooled across the probe | Token-combination diversity, not semantic quality or factuality |
| Repeated 4-gram rate | `(total 4-grams - unique 4-grams) / total 4-grams` | Includes phrases shared between passages, not only loops inside a passage |
| Duplicate continuation rate | `1 - unique complete continuations / continuation count` | Counts additional copies beyond first occurrences, not the number of distinct repeated passages |
| Empirical token entropy | `-sum(p(token) * log(p(token)))`; nats | Entropy of observed token frequencies, not predictive/logit entropy |
| Frozen-G0/S0 continuation NLL | Mean shifted-target NLL of a continuation alone under the fixed real-trained reference model | Predictability of different generated texts under S0; not a model-distance, KL-divergence or independent quality measure |
| Training loss | Token NLL on the current stage's synthetic/real training dataset | Different cycles train on different datasets, so raw values are not directly comparable difficulty scores |
| Selected epoch | Minimum-validation epoch under the default policy; terminal epoch under `final_epoch` | Record policy and executed epochs; final-epoch-5 conditions retain epoch 5 even if validation worsens |
| Throughput / peak memory | Tokens or continuations per second; peak allocated GPU bytes | Record warmed-up workload, batch sizes, precision and hardware |
| Runtime / cost | Measured stage durations; separate quoted/estimated USD and observed account debit | Account debit may include retained storage and is not an itemized compute invoice |

## Historical WikiText scopes

An independent 64-token real block has **63 scored targets**. Full real train, validation and test arrays contain 2,390,598, 247,086 and 283,122 such targets respectively.

Generated-text metrics use the same **first 256 real training prompts** at each generation, each with 64 continuation tokens: 16,384 tokens, 16,128 reference-scored targets and 15,616 four-gram positions. They span six adjacent source articles, with 79.39% of prompt tokens from the first three. This is a fixed probe, not a random representative sample or a full-corpus diversity measurement. N-grams form within each continuation; no n-gram crosses a continuation boundary. Counts are then pooled across continuations. Every original generated token ID counts, including formatting and any special tokens present; EOS suppression means the inspected probes contain no EOS or padding IDs.

Pooled token entropy ignores ordering and combines all outputs. It can remain stable while individual outputs become more repetitive, if different outputs repeat different tokens. Mean per-output entropy and within-output repetition are supplementary diagnostics with different definitions; they must not be substituted for historical pooled curves.

Frozen-S0 scoring feeds each saved continuation to S0 alone, teacher-forcing its preceding continuation tokens and scoring targets 2-64. The original generation prompt is not supplied to this metric. The mean aggregates all 16,128 scored probe targets; the distribution/P95 summarize per-continuation mean NLL. Comparing two arms scores two different sets of text. Similar means therefore do not establish similar quality, similar distance from S0 or similar performance on real WikiText. Predictable repetition/blank lines can score well even when useful output and real-text prediction deteriorate.

The [28 October audit](audits/2026-10-28_graph_and_protocol_audit.md) found that ` =` accounts for 22.82% of final reset-S0 probe tokens versus 11.00% for continued training. Excluding that token diagnostically shrinks the entropy gap from 0.620 to 0.122 nats. Exclusion renormalizes unequal retained sample sizes; it neither replaces the primary metric nor measures semantic quality. Raw-token distinctness and repetition also respond to formatting loops. A single quoted passage can reverse the aggregate model ranking.

The [Week 3 final-epoch-5 report](../reports/20261030T180755Z-full-paired-final5/report.md) identifies a different concentration: continued B10 has 38.62% newline tokens versus reset A10 at 2.80%, including ten entirely newline outputs. Excluding newlines reverses the entropy ordering; those filtered samples have unequal retained token/window counts and are sensitivity checks only. Entirely newline outputs have frozen-S0 mean NLL 0.465, illustrating why a low reference score is not a quality verdict. Preserve the primary raw-token curves and state exactly which filtering, denominator and sample scope a supplementary graph uses.

At generation Sk, probe metrics describe D(k+1)-style output. Real-text evaluation at Sk measures the model after its training on Dk. Do not attribute a model's own output dataset to the training stage that already produced it.

## FineWeb variable-length measurements

The fresh-real workflow uses its own fixed document collections and sequences up to 512 positions. For attention mask `m` and token sequence `x`, score target positions `t = 1,...,L-1` only where `m[t] = 1`. Corpus NLL is `sum(-m[t] log p(x[t] | x[:t])) / sum(m[t])`, pooled over all sequences. Perplexity is its exponential. Every sequence begins with BOS, so its first source token is scored. EOS contributes a target only at a genuine document ending.

- Source-text count measures tokenizer output before inserting specials or padding. This is the per-cycle training quota.
- Non-padding count includes inserted BOS/EOS. Scored-target count excludes each sequence's initial BOS. Padding contributes to neither attention nor loss.
- Optimization loss averages token losses collected while weights change during training. Post-epoch training NLL re-evaluates the completed model with dropout disabled on that cycle's data.
- The generalization gap is fixed validation NLL minus post-epoch training NLL, in nats per scored token. Training collections change across cycles and can differ in difficulty.
- The exploratory stability threshold is validation PPL at most 1.05 times its initial value throughout the full pilot. Test PPL remains a reporting measure. The small smoke trajectory establishes pipeline behaviour.

## Matched FineWeb completion measurements

The `completion_comparison` workflow supplies BOS, 64 prompt tokens and 64 continuation tokens. All 129 positions are visible through attention. Loss masks select positions 65–128, so the logit at position 64 predicts the first continuation token. Each example contributes exactly 64 targets. Corpus continuation NLL is the sum of their negative log-probabilities divided by the total target count. Perplexity is its exponential.

- Real and synthetic arms train on the same prompt chunks and ordering. The real targets are the following source tokens. Synthetic targets are frozen outputs of the preceding synthetic checkpoint.
- Primary validation/test scores always use the same real continuations. Their full scopes are 7,322 / 7,151 examples and 468,608 / 457,664 scored targets. Median/P95 example NLL summarize passage difficulty.
- Generated-output statistics use a fixed 256-document validation probe. They score only generated continuations. Pooled diversity/entropy, within-output repetition and newline concentration retain separate denominators.
- Secondary FineWeb, WikiText-103, C4, PG-19 and AG News scores reuse the frozen 512-token full-text arrays. Their context and scoring objective differ from the primary continuation metric.
- The GPU smoke uses 256 pairs per training cycle, 64 held-out pairs per split and a 64-document probe. Its trajectory validates execution. The full scientific comparison remains a separate run.

Repeated-chunk controls additionally score the fixed real first chunk at baseline and every checkpoint. `fixed_real_anchor_loss` uses the original real continuations in both arms; `real_anchor_generalization_gap` is test NLL minus this anchor NLL. The anchor is training data for the repeated-real arm, not a held-out collection. Interpret changes from its baseline because the two collections can differ in difficulty. Falling training loss together with rising held-out loss supports overfitting; a widening gap alone does not establish held-out deterioration. Synthetic own-target training NLL and real-anchor NLL score different targets and must remain separately labelled.

## What the existing results support

The Week 2 run has clear overall deterioration in fixed-real-text prediction and increased final repetition, with an early perplexity recovery and mixed diversity. Its final distinct scores exceed S0 and its final empirical token entropy is close to S0; it does not establish uniform vocabulary collapse. A concrete repeated output illustrates a failure mode but does not estimate its full-corpus prevalence.

Only one seed was run per full condition. Do not add confidence bands from passage variation, imply significance, claim every metric worsens or identify a causal explanation for the dip from these trajectories alone. Frozen-reference NLL increases can reflect many kinds of drift. Specific loss of rare events requires additional measurement beyond a P95 block-loss statistic.

See the [Week 1 report](../reports/20261020T211121Z-full/report.md) and [Week 2 report](../reports/20261026T192740Z-full-continued/report.md) for recorded values and analysis provenance.
