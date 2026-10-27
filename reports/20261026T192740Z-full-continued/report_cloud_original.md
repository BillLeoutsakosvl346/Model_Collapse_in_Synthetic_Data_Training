# Week 1 full experiment

Experiment: `20261026T192740Z-full-continued`. Seed: 42.

One seed is preliminary evidence; non-monotonic changes are reported without adjustment.

Untouched pretrained test corpus perplexity: 115.4192.

| Generation | Validation PPL | Test PPL | Distinct-2 | Repeated 4-grams |
|---|---:|---:|---:|---:|
| G0 | 34.2114 | 33.8851 | 0.2881 | 0.4855 |
| G1 | 48.9853 | 48.3821 | 0.3335 | 0.5085 |
| G2 | 43.2837 | 42.9750 | 0.4319 | 0.3929 |
| G3 | 43.0424 | 42.7186 | 0.4089 | 0.4321 |
| G4 | 44.2018 | 43.8836 | 0.3859 | 0.4647 |
| G5 | 46.0809 | 45.8545 | 0.3744 | 0.4826 |
| G6 | 47.5523 | 47.2401 | 0.3472 | 0.5215 |
| G7 | 49.1720 | 48.9413 | 0.3552 | 0.5107 |
| G8 | 50.8444 | 50.6450 | 0.3372 | 0.5355 |
| G9 | 52.9896 | 53.0420 | 0.3320 | 0.5482 |
| G10 | 55.4860 | 55.5568 | 0.3471 | 0.5236 |

Corpus PPL = exp(sum of NLL / count of scored shifted targets); each real block scores 63 targets. No average of batch perplexities.

Distinct-n counts unique n-grams divided by all n-grams within individual continuations. Repeated 4-gram rate counts occurrences beyond the first divided by all within-continuation 4-grams. Exact duplicate rate = 1 − unique continuations / continuation count. Special tokens remain in token metrics.

Continued-weight variant: cycle 0 loads the previous full-data real-trained G0, pinned by private Hub commit and file/tensor hashes. Each cycle's selected model generates the next dataset and initializes its own subsequent training. The model weights carry forward; AdamW and its cosine schedule restart each cycle. Validation selects the lowest-loss epoch among the newly trained epochs; there is no untrained-parent candidate. Test data only reports results. This is a separate variant, not a claim of exact paper reproduction. Prompts are fixed real training blocks; only continuation token IDs are used as subsequent training targets.

Reconstruction differences: split-wide concatenation (authors group map batches); explicit initialization policy; BF16 autocast; corrected token-weighted corpus PPL; repetition penalty 1.0 for the paper's main condition (released code hardcodes 3.0). No real training targets retained after G0.

Paper: https://doi.org/10.1038/s41586-024-07566-y ; authors' code: https://github.com/iliaishacked/curse_recurse

## Full-data forecasts

Measured on NVIDIA RTX PRO 4000 Blackwell at compute plus running storage rate $0.5756/hour. These are conservative estimates, not completed full runs; setup validation spend and retained-storage reserve are additional.

- G5: 5.14 hours; $2.96.
- G10: 10.11 hours; $5.82.
