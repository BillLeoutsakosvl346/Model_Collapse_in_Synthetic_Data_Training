# Week 1 smoke experiment

Experiment: `20261020T193206Z-smoke`. Seed: 42.

This small smoke experiment validates plumbing, not evidence of model collapse.

Untouched pretrained test corpus perplexity: 122.8392.

| Generation | Validation PPL | Test PPL | Distinct-2 | Repeated 4-grams |
|---|---:|---:|---:|---:|
| G0 | 94.6448 | 92.6759 | 0.2759 | 0.6095 |
| G1 | 113.1807 | 110.1995 | 0.2707 | 0.6424 |
| G2 | 113.0218 | 110.8285 | 0.2760 | 0.6388 |

Corpus PPL = exp(sum of NLL / count of scored shifted targets); each real block scores 63 targets. No average of batch perplexities.

Distinct-n counts unique n-grams divided by all n-grams within individual continuations. Repeated 4-gram rate counts occurrences beyond the first divided by all within-continuation 4-grams. Exact duplicate rate = 1 − unique continuations / continuation count. Special tokens remain in token metrics.

All training starts from the pinned original pretrained model. Prompts are fixed real training blocks; only continuation token IDs are used as subsequent training targets. Validation selects the best epoch; test data only reports results.

Reconstruction differences: split-wide concatenation (authors group map batches); explicit fresh-base initialization; BF16 autocast; corrected token-weighted corpus PPL; repetition penalty 1.0 for the paper's main condition (released code hardcodes 3.0). No real training targets retained after G0.

Paper: https://doi.org/10.1038/s41586-024-07566-y ; authors' code: https://github.com/iliaishacked/curse_recurse

## Full-data forecasts

Measured on NVIDIA RTX 4000 Ada Generation at compute plus running storage rate $0.2856/hour. These are conservative estimates, not completed full runs; setup validation spend and retained-storage reserve are additional.

- G5: 6.03 hours; $1.72.
- G10: 11.70 hours; $3.34.
