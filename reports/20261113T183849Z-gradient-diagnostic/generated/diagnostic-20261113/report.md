# Why do a few real continuations change recursive training?

This is a local intervention study of saved SmolLM2 checkpoints, not a new ten-cycle training experiment.

## Method

- Restore the original M0 and the 99%/100% conditions’ M4/M9, together with exact cycle-1/5/10 targets and saved AdamW states.
- Calculate complete-source, token-averaged gradients with FP32 arithmetic and dropout disabled. Prompt labels remain masked; every continuation has 64 scored targets.
- Decompose the actual 99% objective into its 100 real and 9,900 synthetic rows. Independently compare original and generated continuations on identical prompt rows.
- Apply one isolated aggregate-gradient update per intervention from the same starting model/optimizer state. These do not replay the historical 128-example minibatches.
- Evaluate a frozen 1,024-pair validation probe; confirm historical-rate mixed/history effects on all 7,447 validation pairs. No validation gradients update the model.
- Compare recorded learning rates with a fixed 3e-6 diagnostic rate. Explicit zero gradients retain AdamW momentum and weight decay as a separate control.

## Source-gradient measurements

| Condition | Checkpoint | Real source NLL | Synthetic source NLL | 1% × real norm | 99% × synthetic norm | Same-prompt cosine |
|---|---|---:|---:|---:|---:|---:|
| mixed99 | M0 | 2.89206 | 0.639788 | 0.00984463 | 3.36464 | -0.151474 |
| mixed99 | M4 | 3.8523 | 0.043268 | 0.330086 | 0.569043 | 0.664494 |
| mixed99 | M9 | 4.66879 | 0.0290486 | 0.515483 | 1.11931 | 0.742097 |
| synthetic | M0 | 2.90614 | 0.639756 | — | — | -0.150101 |
| synthetic | M4 | 4.32343 | 0.0401964 | — | — | 0.788092 |
| synthetic | M9 | 6.50558 | 0.0275235 | — | — | 0.837078 |

The 100% condition has no actual real training component. Its real-only diagnostic uses the original real targets on the same 10,000 prompts. Its unweighted synthetic norm remains available in the raw records.

## Immediate effects at the historical learning rate

All entries below are changes in real-validation NLL. Negative values improve predictions; positive values worsen them.

| Condition | Checkpoint | History-only | Real-only | Synthetic-only | Mixed | Mixed minus synthetic |
|---|---|---:|---:|---:|---:|---:|
| mixed99 | M0 | 2.94675e-09 | -0.000406395 | 0.000176688 | 0.000171915 | -4.77313e-06 |
| mixed99 | M4 | 0.00373956 | -0.0702215 | 0.00181826 | -0.0010216 | -0.00283986 |
| mixed99 | M9 | 0.00208338 | -0.0526329 | -0.00104711 | -0.00331343 | -0.00226632 |
| synthetic | M0 | 2.94675e-09 | -0.000994371 | 0.000176679 | — | — |
| synthetic | M4 | 0.00684304 | -0.110144 | 0.000793039 | — | — |
| synthetic | M9 | -0.00127703 | -0.0948693 | -0.0119453 | — | — |

The difference from the history-only step matters: an update can improve on the optimizer’s inherited direction while still worsening loss relative to taking no step.

## Reading the evidence

- Compare the real and synthetic update directions and held-out effects together. A large gradient is not automatically useful.
- The actual source groups differ in size, so their mean-gradient norms also reflect cancellation and sampling variability. The matched-prompt comparison is a separate, equal-row check.
- Gradient components add linearly before AdamW. Their separate AdamW updates do not add linearly, because normalization and saved moments intervene.
- Small or sign-changing effects are retained. Document-bootstrap intervals describe this validation sample, not training-seed uncertainty.
- Six checkpoint cases from one historical seed do not establish a universal mixture threshold, permanent equilibrium or a complete explanation of model collapse.


## Supplementary mechanism check — exploratory

Per-parameter real, matched-real and synthetic norms are in [per_parameter_norms.csv](per_parameter_norms.csv). The synthetic breakdown was recovered in a separately accepted pass, with global norms/losses checked against the primary measurements.

After positive real/synthetic alignment was observed at the middle checkpoint, we added a diagnostic validation gradient. It is never supplied to an optimizer. We project the actual synthetic/mixed AdamW displacement onto that validation gradient: `first-order loss change = validation_gradient · weight_displacement`.

| Condition | Before cycle | Update | Synthetic/validation gradient cosine | First-order NLL change | Observed NLL change | Nonlinear residual |
|---|---:|---|---:|---:|---:|---:|
| mixed99 | 1 | synthetic_only | -0.0560466 | 0.000130686 | 0.00017669 | 4.60038e-05 |
| mixed99 | 1 | mixed | -0.0560466 | 0.000125926 | 0.000171934 | 4.60075e-05 |
| mixed99 | 5 | synthetic_only | 0.64003 | 0.00181239 | 0.00181821 | 5.82626e-06 |
| mixed99 | 5 | mixed | 0.64003 | -0.00102663 | -0.00102177 | 4.85679e-06 |
| mixed99 | 5 | synthetic_without_momentum | 0.64003 | -0.00191625 | -0.00190882 | 7.42906e-06 |
| mixed99 | 10 | synthetic_only | 0.737822 | -0.00105705 | -0.00104714 | 9.91037e-06 |
| mixed99 | 10 | mixed | 0.737822 | -0.00331228 | -0.00331346 | -1.18129e-06 |
| mixed99 | 10 | synthetic_without_momentum | 0.737822 | -0.00311989 | -0.00311293 | 6.96293e-06 |
| synthetic | 1 | synthetic_only | -0.0560211 | 0.000130698 | 0.000176637 | 4.59387e-05 |
| synthetic | 5 | synthetic_only | 0.77383 | 0.000799948 | 0.000793062 | -6.88545e-06 |
| synthetic | 5 | synthetic_without_momentum | 0.77383 | -0.00602432 | -0.00601823 | 6.09462e-06 |
| synthetic | 10 | synthetic_only | 0.836233 | -0.0119077 | -0.0119453 | -3.76733e-05 |
| synthetic | 10 | synthetic_without_momentum | 0.836233 | -0.0107045 | -0.0107416 | -3.71503e-05 |

Positive first-order values mean that the actual AdamW movement points toward worse held-out predictions locally. The residual includes higher-order loss changes and finite-precision effects. Raw gradient alignment and AdamW displacement alignment are different quantities. These additions are exploratory rather than preregistered confirmation.


## Figures

![01_weighted_gradient_strength.png](01_weighted_gradient_strength.png)

![02_gradient_alignment.png](02_gradient_alignment.png)

![03_mixed99_historical_update_effects.png](03_mixed99_historical_update_effects.png)

![03_mixed99_common_update_effects.png](03_mixed99_common_update_effects.png)

![03_synthetic_historical_update_effects.png](03_synthetic_historical_update_effects.png)

![03_synthetic_common_update_effects.png](03_synthetic_common_update_effects.png)

![06_effect_of_real_injection.png](06_effect_of_real_injection.png)

![05_weighted_gradient_strength_log.png](05_weighted_gradient_strength_log.png)

![04_mixed99_incremental_data_effect.png](04_mixed99_incremental_data_effect.png)

![04_synthetic_incremental_data_effect.png](04_synthetic_incremental_data_effect.png)

![07_layer_gradient_strength_ratio.png](07_layer_gradient_strength_ratio.png)

![08_optimizer_momentum_control.png](08_optimizer_momentum_control.png)
