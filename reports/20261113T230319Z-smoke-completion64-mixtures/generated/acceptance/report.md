# Matched FineWeb completion comparison

All conditions begin at original pretrained HuggingFaceTB/SmolLM2-135M. Each cycle supplies the same real 64-token prompts. Real targets are the actual next 64 source tokens. Each condition generates its synthetic targets using its own preceding checkpoint. Mixtures replace selected continuations at their original prompt rows and retain the remaining real targets. BOS and prompt positions remain visible, while the continuation alone contributes to training loss.

Primary perplexity scores fixed real continuations. Secondary full-text perplexities use the frozen 512-token source-bank arrays. Output statistics use a fixed validation probe. The shared cycle-zero point is reused evidence.

| Arm | Cycle | Validation PPL | Test PPL | Test targets |
|---|---:|---:|---:|---:|
| real | 0 | 23.98984 | 17.49658 | 8192 |
| real | 1 | 23.92631 | 17.44908 | 8192 |
| real | 2 | 23.92271 | 17.42260 | 8192 |
| mixed99 | 0 | 23.98984 | 17.49658 | 8192 |
| mixed99 | 1 | 27.09130 | 19.62328 | 8192 |
| mixed99 | 2 | 33.13176 | 24.01167 | 8192 |
| synthetic | 0 | 23.98984 | 17.49658 | 8192 |
| synthetic | 1 | 27.11154 | 19.64026 | 8192 |
| synthetic | 2 | 33.31431 | 24.14373 | 8192 |

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

## Actual-update SGD diagnostics

Training uses FP32 plain SGD at one calibrated constant rate, without momentum, weight decay, scheduler or gradient clipping. Generation and the fixed output-statistic probe use greedy decoding. Historical AdamW/five-beam results differ in several settings and are not optimizer-only controls.

Real and synthetic gradients are directly measured from the same training forward graphs and normalized by the complete effective batch's scored-target count. Only the ordinary combined SGD update advances training. A reusable scratch model measures weighted-real, weighted-synthetic and combined counterfactuals on the fixed real-validation probe. The latter never supplies an update. Empty-source states share exact measurements with their no-op or combined counterparts.

A negative mixed-minus-synthetic NLL means including that batch's real component improved real-probe predictions locally. Gradient norm alone does not establish quality. Separate source loss changes need not add: the interaction column records that nonlinearity. Small effects flagged numerically unresolved are retained.

Epoch means and intervals describe local interventions along the actual observed training path. They are not trajectories produced by deleting a source for an entire epoch, and their sums do not establish long-term causal influence. Probe documents are resampled together; repeated optimizer updates are not independent seed replicates.

| Arm | Cycle | Updates | Real-present updates | Mean real conditional NLL change | Document-bootstrap 95% interval |
|---|---:|---:|---:|---:|---|
| real | 1 | 4 | 4 | -0.00054099042 | [-0.00088663461, -0.00020159808] |
| real | 2 | 4 | 4 | -0.00036586357 | [-0.00067056844, -8.3948551e-05] |
| mixed99 | 1 | 4 | 2 | -8.0374334e-05 | [-0.0001004547, -5.9973484e-05] |
| mixed99 | 2 | 4 | 3 | -0.0061085497 | [-0.0067263838, -0.0055899839] |
| synthetic | 1 | 4 | 0 | 0 | [0, 0] |
| synthetic | 2 | 4 | 0 | 0 | [0, 0] |

Only sealed completed attempts contribute to these tables; replayed abandoned attempts remain excluded.

![sgd_weighted_source_gradients.png](sgd_weighted_source_gradients.png)

![sgd_source_alignment.png](sgd_source_alignment.png)

![sgd_conditional_source_effects.png](sgd_conditional_source_effects.png)

![sgd_real_local_update_effects.png](sgd_real_local_update_effects.png)

![sgd_mixed99_local_update_effects.png](sgd_mixed99_local_update_effects.png)

![sgd_synthetic_local_update_effects.png](sgd_synthetic_local_update_effects.png)

![sgd_layer_source_ratio.png](sgd_layer_source_ratio.png)

![sgd_epoch_real_effect.png](sgd_epoch_real_effect.png)

