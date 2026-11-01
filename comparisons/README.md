# Matched comparisons

| Comparison | Shared beginning | Deliberate difference | Report |
| --- | --- | --- | --- |
| [Shared S0/S1 reset versus continue](shared_s0_s1_reset_vs_continue/README.md) | Exact historical real S0, selected-epoch-1 S1 and D2 | Initialize later stages from fixed S0 versus preceding selected weights | [Week 2 matched extension](../reports/20261027T211852Z-full-reset-s0/report.md) |
| [Shared final epoch 5](shared_start_final_epoch5/README.md) | Historical S0/D1, new epoch-5 S1 and new D2 | Fixed S0 versus preceding epoch-5 weights | [Week 3](../reports/20261030T180755Z-full-paired-final5/report.md) |
| [50/50 versus synthetic-only](continued_mixed50_vs_synthetic_final5/README.md) | Exact S0 and initial D1 pool; matched recorded environment and training budget | Half real/half synthetic versus entirely synthetic targets; continued epoch-5 weights in both | [Week 3, Part 2](../reports/20261031T163428Z-full-continued/report.md) |
| [Mixed-fraction final-epoch campaign](mixed_fraction_final5/README.md) | Historical mixed50 prefix; a new shared mixed20 S0/S1/D2/M2 | 50/50 reset and matched 20/80 continued/reset; sequential execution | [Setup validated; full unlaunched](../docs/mixed-campaign-handoff.md), [GPU acceptance](../reports/20261101T022210Z-smoke-mixed-campaign/report.md) |

Each directory holds matching requirements, artifact provenance, configuration and independent analysis. The Week 2 and Week 3 shared S1/D2 are different artifacts; do not reuse one as the other. The earliest pretrained-reset and continued plots had different first updates and GPUs, so they are contextual trajectories rather than a shared-start causal comparison.

Keep independent evidence audits distinct from presentation scripts. Reproduce into a new output directory and retain original reports/hashes. See [metrics](../docs/metrics.md) for probe limitations and [reports](../reports/README.md) for verification scope. The 50/50 continued experiment is a separate condition; no completed matched reset mixture arm is implied.
