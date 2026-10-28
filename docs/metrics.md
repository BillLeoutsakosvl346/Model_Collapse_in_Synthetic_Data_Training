# Metric definitions and interpretation

All comparisons must use matching tokenisation, context, masks, output counts and denominators. Raw measurements remain in the dated run reports and records; generated analysis must cite those inputs and hashes.

| Metric | Definition and unit | Interpretation limit |
| --- | --- | --- |
| Real validation/test NLL | Sum of next-token negative log-probabilities divided by the count of scored shifted targets; nats/token | Validation selects checkpoints; test is reporting-only |
| Corpus perplexity | `exp(total NLL / total scored targets)` | Never average batch perplexities or substitute mean passage perplexity |
| Median / P95 block NLL | Median / 95th percentile of mean NLL on individual real 64-token blocks | High-loss blocks are not automatically annotated rare words/topics |
| Mean passage perplexity | Arithmetic mean of `exp(block mean NLL)` | Usually differs from corpus perplexity because exponentiation is nonlinear |
| Passage perplexity SD | Actual standard deviation across passage perplexities | Passage heterogeneity, not across-seed statistical uncertainty; not `exp(std(NLL))` |
| Distinct-1/2/3 | Unique n-grams divided by total n-gram occurrences, pooled across the probe | Token-combination diversity, not semantic quality or factuality |
| Repeated 4-gram rate | `(total 4-grams - unique 4-grams) / total 4-grams` | Includes phrases shared between passages, not only loops inside a passage |
| Duplicate continuation rate | `1 - unique complete continuations / continuation count` | Counts additional copies beyond first occurrences, not the number of distinct repeated passages |
| Empirical token entropy | `-sum(p(token) * log(p(token)))`; nats | Entropy of observed token frequencies, not predictive/logit entropy |
| Frozen-G0/S0 continuation NLL | Mean shifted-target NLL of a continuation alone under the fixed real-trained reference model | Distribution drift indicator, not KL divergence or an independent quality verdict |
| Training loss | Token NLL on the current stage's synthetic/real training dataset | Different cycles train on different datasets, so raw values are not directly comparable difficulty scores |
| Selected epoch | Epoch minimising fixed real validation NLL among newly trained epochs | Distinguish executed epochs from epochs inherited through the selected checkpoints |
| Throughput / peak memory | Tokens or continuations per second; peak allocated GPU bytes | Record warmed-up workload, batch sizes, precision and hardware |
| Runtime / cost | Measured stage durations; separate quoted/estimated USD and observed account debit | Account debit may include retained storage and is not an itemized compute invoice |

## Exact scopes

An independent 64-token real block has **63 scored targets**. Full real train, validation and test arrays contain 2,390,598, 247,086 and 283,122 such targets respectively.

Generated-text metrics use the same **first 256 real training prompts** at each generation, each with 64 continuation tokens: 16,384 tokens, 16,128 reference-scored targets and 15,616 four-gram positions. They span six adjacent source articles, with 79.39% of prompt tokens from the first three. This is a fixed probe, not a random representative sample or a full-corpus diversity measurement. N-grams form within each continuation; no n-gram crosses a continuation boundary. Counts are then pooled across continuations. Every original generated token ID counts, including formatting and any special tokens present; EOS suppression means the inspected probes contain no EOS or padding IDs.

Pooled token entropy ignores ordering and combines all outputs. It can remain stable while individual outputs become more repetitive, if different outputs repeat different tokens. Mean per-output entropy and within-output repetition are supplementary diagnostics with different definitions; they must not be substituted for historical pooled curves.

The [28 October audit](audits/2026-10-28_graph_and_protocol_audit.md) found that ` =` accounts for 22.82% of final reset-S0 probe tokens versus 11.00% for continued training. Excluding that token diagnostically shrinks the entropy gap from 0.620 to 0.122 nats. Exclusion renormalizes unequal retained sample sizes; it neither replaces the primary metric nor measures semantic quality. Raw-token distinctness and repetition also respond to formatting loops. A single quoted passage can reverse the aggregate model ranking.

At generation Sk, probe metrics describe D(k+1)-style output. Real-text evaluation at Sk measures the model after its training on Dk. Do not attribute a model's own output dataset to the training stage that already produced it.

## What the existing results support

The Week 2 run has clear overall deterioration in fixed-real-text prediction and increased final repetition, with an early perplexity recovery and mixed diversity. Its final distinct scores exceed S0 and its final empirical token entropy is close to S0; it does not establish uniform vocabulary collapse. A concrete repeated output illustrates a failure mode but does not estimate its full-corpus prevalence.

Only one seed was run per full condition. Do not add confidence bands from passage variation, imply significance, claim every metric worsens or identify a causal explanation for the dip from these trajectories alone. Frozen-reference NLL increases can reflect many kinds of drift. Specific loss of rare events requires additional measurement beyond a P95 block-loss statistic.

See the [Week 1 report](../reports/20261020T211121Z-full/report.md) and [Week 2 report](../reports/20261026T192740Z-full-continued/report.md) for recorded values and analysis provenance.
