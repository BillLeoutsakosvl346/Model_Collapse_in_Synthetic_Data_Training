# Matched FineWeb completion comparison

All conditions begin at original pretrained OPT. Each cycle supplies the same real 64-token prompts. Real targets are the actual next 64 source tokens. Each condition generates its synthetic targets using its own preceding checkpoint. Mixtures replace selected continuations at their original prompt rows and retain the remaining real targets. BOS and prompt positions remain visible, while the continuation alone contributes to training loss.

Primary perplexity scores fixed real continuations. Secondary full-text perplexities use the frozen 512-token source-bank arrays. Output statistics use a fixed validation probe. The shared cycle-zero point is reused evidence.

| Arm | Cycle | Validation PPL | Test PPL | Test targets |
|---|---:|---:|---:|---:|
| mixed80 | 0 | 39.27858 | 26.72833 | 4096 |
| mixed80 | 1 | 39.59177 | 26.89057 | 4096 |
| mixed80 | 2 | 40.27595 | 27.25182 | 4096 |
| mixed80 | 3 | 40.48079 | 27.35478 | 4096 |

## Figures

![Real continuation perplexity](test_perplexity.png)

![P95 real-example NLL](test_nll_p95.png)

![Pooled distinct-2](distinct_2.png)

![Pooled token entropy (nats)](token_entropy_nats.png)

![Mean within-output entropy (nats)](mean_within_entropy_nats.png)

![Mean within-output repeated 4-grams](mean_within_repeated_4gram_rate.png)

![Pooled repeated 4-grams](repeated_4gram_rate.png)

![Duplicate continuations](duplicate_continuation_rate.png)

![Learning rate](learning_rate_end.png)

![Training and persistence time (seconds)](training_seconds.png)

![fineweb edu full-text perplexity](secondary_fineweb_edu_perplexity.png)

![wikitext103 full-text perplexity](secondary_wikitext103_perplexity.png)

![c4 en full-text perplexity](secondary_c4_en_perplexity.png)

![pg19 full-text perplexity](secondary_pg19_perplexity.png)

![ag news full-text perplexity](secondary_ag_news_perplexity.png)

One seed gives descriptive trajectories. Pooled and within-output statistics have separate denominators. Exact training and evaluation counts are recorded in the saved manifests.
