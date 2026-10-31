# Generated smoke analysis: continued selected weights; 50% synthetic / 50% real targets

Experiment: `20261031T153536Z-smoke-continued`. Seed: 42.

This is regenerated analysis of saved evidence, separate from the authored report. No model training, generation, re-evaluation or cloud reads were performed.

Pipeline validation only; a smoke experiment is not evidence of collapse.

Untouched pretrained real-test corpus perplexity: 122.8388.

| Cycle | Origin | Validation PPL | Test PPL | Distinct-2 | Repeated 4-grams |
|---|---|---:|---:|---:|---:|
| 0 | current_run | 30.7512 | 36.8864 | 0.2881 | 0.4855 |
| 1 | current_run | 33.1903 | 39.6239 | 0.3022 | 0.5574 |
| 2 | current_run | 33.0874 | 40.8561 | 0.3215 | 0.5382 |

## Training mixtures

| Cycle | Real blocks | Synthetic blocks | Real tokens | Synthetic tokens |
|---|---:|---:|---:|---:|
| 1 | 128 | 128 | 8192 | 8192 |
| 2 | 128 | 128 | 8192 | 8192 |

Cycle 0 loads the saved real-data-trained G0/S0. Each next cycle starts from the preceding selected weights. Optimizer and scheduler restart.

Each cycle samples real training blocks and continuation-only synthetic targets without replacement, then shuffles and freezes the mixture for all epochs. Original generation prompts are excluded from synthetic targets. Epoch 5 is retained regardless of validation loss; validation is diagnostic and test data only reports results.

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
- **within_output_diagnostics:** Compute empirical token entropy and distinct/repeated n-grams within each 64-token output, then average outputs; pooled and within-output statistics answer different questions.
- **newline_diagnostics:** Count pinned OPT token 50118 (newline), outputs with at least 32 newlines, and entirely newline outputs. Formatting is included in the unfiltered diversity metrics.

## Optional benchmark forecasts

Conservative throughput estimates, not completed runs or itemized invoices. Setup and retained-storage reserves are additional. Imported shared work is excluded.

- Through cycle 5: 4.37 hours; $2.52.
- Through cycle 10: 9.33 hours; $5.37.
