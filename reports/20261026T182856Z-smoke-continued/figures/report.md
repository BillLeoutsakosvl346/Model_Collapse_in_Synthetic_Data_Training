# Week 1 smoke experiment

Experiment: `20261026T182856Z-smoke-continued`. Seed: 42.

This small smoke experiment validates plumbing, not evidence of model collapse.

Untouched pretrained test corpus perplexity: 122.8727.

| Generation | Validation PPL | Test PPL | Distinct-2 | Repeated 4-grams |
|---|---:|---:|---:|---:|
| G0 | 30.7601 | 36.8755 | 0.2923 | 0.4887 |
| G1 | 32.3485 | 38.3117 | 0.3359 | 0.5236 |
| G2 | 35.0340 | 41.4817 | 0.3752 | 0.4830 |

Corpus PPL = exp(sum of NLL / count of scored shifted targets); each real block scores 63 targets. No average of batch perplexities.

Distinct-n counts unique n-grams divided by all n-grams within individual continuations. Repeated 4-gram rate counts occurrences beyond the first divided by all within-continuation 4-grams. Exact duplicate rate = 1 − unique continuations / continuation count. Special tokens remain in token metrics.

Continued-weight variant: cycle 0 loads the previous full-data real-trained G0, pinned by private Hub commit and file/tensor hashes. Each cycle's selected model generates the next dataset and initializes its own subsequent training. The model weights carry forward; AdamW and its cosine schedule restart each cycle. Validation selects the lowest-loss epoch among the newly trained epochs; there is no untrained-parent candidate. Test data only reports results. This is a separate variant, not a claim of exact paper reproduction. Prompts are fixed real training blocks; only continuation token IDs are used as subsequent training targets.

Reconstruction differences: split-wide concatenation (authors group map batches); explicit initialization policy; BF16 autocast; corrected token-weighted corpus PPL; repetition penalty 1.0 for the paper's main condition (released code hardcodes 3.0). No real training targets retained after G0.

Paper: https://doi.org/10.1038/s41586-024-07566-y ; authors' code: https://github.com/iliaishacked/curse_recurse

## Full-data forecasts

Measured on NVIDIA RTX 2000 Ada Generation at compute plus running storage rate $0.2456/hour. These are conservative estimates, not completed full runs; setup validation spend and retained-storage reserve are additional.

- G5: 7.57 hours; $1.86.
- G10: 14.92 hours; $3.67.
