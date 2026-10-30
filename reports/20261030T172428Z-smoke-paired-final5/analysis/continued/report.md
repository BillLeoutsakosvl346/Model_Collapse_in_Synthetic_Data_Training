# Generated smoke analysis: continued selected weights

Experiment: `20261030T172428Z-smoke-paired-final5-continued`. Seed: 42.

This is regenerated analysis of saved evidence, separate from the authored report. No model training, generation, re-evaluation or cloud reads were performed.

Pipeline validation only; a smoke experiment is not evidence of collapse.

Untouched pretrained real-test corpus perplexity: 122.8388.

| Cycle | Origin | Validation PPL | Test PPL | Distinct-2 | Repeated 4-grams |
|---|---|---:|---:|---:|---:|
| 0 | current_run | 30.7512 | 36.8864 | 0.2881 | 0.4855 |
| 1 | current_run | 37.3552 | 44.0323 | 0.3315 | 0.5348 |
| 2 | current_run | 36.4110 | 45.1084 | 0.3968 | 0.4488 |

Cycle 0 loads the saved real-data-trained G0/S0. Each next cycle starts from the preceding selected weights. Optimizer and scheduler restart.

Fixed real prompts generate continuations; only continuation tokens train subsequent cycles. Epoch 5 is retained regardless of validation loss; validation is diagnostic and test data only reports results.

## Metric definitions

- **corpus_perplexity:** exp(sum shifted-token NLL / scored targets), not arithmetic mean passage perplexity; 63 targets per independent 64-token block.
- **distinct_n:** Unique within-continuation n-grams / all within-continuation n-gram occurrences, pooled over the fixed probe; no n-grams cross row boundaries.
- **repeated_4gram_rate:** Occurrences beyond the first of each 4-gram / all within-continuation 4-grams, pooled across outputs; includes phrases shared between outputs.
- **duplicate_continuation_rate:** 1 - unique complete continuation rows / probe rows.
- **token_entropy_nats:** Empirical unigram token-frequency entropy of the generated probe, including special tokens; not predictive uncertainty.
- **frozen_s0_nll:** Mean shifted-token NLL of continuation-only text under frozen real-data G0/S0; not conditioned on the original prompt and not KL divergence.
- **test_block_nll_p95:** 95th percentile of mean NLL per real test block; not a specifically identified rare-token measure.
- **test_sequence_perplexity_sd:** Population standard deviation across real passages, not uncertainty across independent seeds.
- **probe_scope:** The same first 256 real training prompts (or the actual saved smaller probe); descriptive measurements, not a random corpus sample.
- **runtime:** Original recorded train/generation seconds retained; new-work seconds are zero for imported shared stages, with their source identity preserved separately.

## Optional benchmark forecasts

Conservative throughput estimates, not completed runs or itemized invoices. Setup and retained-storage reserves are additional. Imported shared work is excluded.

- Through cycle 5: 5.20 hours; $2.99.
- Through cycle 10: 10.22 hours; $5.88.
