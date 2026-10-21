# Week 1 full experiment

Experiment: `20261020T211121Z-full`. Seed: 42.

One seed is preliminary evidence; non-monotonic changes are reported without adjustment.

Untouched pretrained test corpus perplexity: 115.4316.

| Generation | Validation PPL | Test PPL | Distinct-2 | Repeated 4-grams |
|---|---:|---:|---:|---:|
| G0 | 34.2136 | 33.8839 | 0.2923 | 0.4887 |
| G1 | 62.9982 | 62.4912 | 0.3361 | 0.5150 |
| G2 | 54.4245 | 53.8417 | 0.4058 | 0.4429 |
| G3 | 53.3268 | 52.8088 | 0.3821 | 0.4688 |
| G4 | 53.0158 | 52.4957 | 0.3814 | 0.4818 |
| G5 | 52.7082 | 52.1732 | 0.3535 | 0.5223 |
| G6 | 53.1723 | 52.8445 | 0.3801 | 0.4865 |
| G7 | 53.0834 | 52.4909 | 0.3390 | 0.5394 |
| G8 | 53.7194 | 53.2654 | 0.3382 | 0.5413 |
| G9 | 53.7934 | 53.2547 | 0.3274 | 0.5570 |
| G10 | 53.5398 | 52.8282 | 0.3058 | 0.5905 |

Corpus PPL = exp(sum of NLL / count of scored shifted targets); each real block scores 63 targets. No average of batch perplexities.

Distinct-n counts unique n-grams divided by all n-grams within individual continuations. Repeated 4-gram rate counts occurrences beyond the first divided by all within-continuation 4-grams. Exact duplicate rate = 1 − unique continuations / continuation count. Special tokens remain in token metrics.

All training starts from the pinned original pretrained model. Prompts are fixed real training blocks; only continuation token IDs are used as subsequent training targets. Validation selects the best epoch; test data only reports results.

Reconstruction differences: split-wide concatenation (authors group map batches); explicit fresh-base initialization; BF16 autocast; corrected token-weighted corpus PPL; repetition penalty 1.0 for the paper's main condition (released code hardcodes 3.0). No real training targets retained after G0.

Paper: https://doi.org/10.1038/s41586-024-07566-y ; authors' code: https://github.com/iliaishacked/curse_recurse

## Full-data forecasts

Measured on NVIDIA RTX 2000 Ada Generation at compute plus running storage rate $0.2456/hour. These are conservative estimates, not completed full runs; setup validation spend and retained-storage reserve are additional.

- G5: 7.13 hours; $1.75.
- G10: 13.77 hours; $3.38.
