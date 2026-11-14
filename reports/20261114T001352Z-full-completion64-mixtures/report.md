# SmolLM2 SGD/greedy comparison

All three ten-cycle conditions completed: 0%, 99% and 100% synthetic targets, with constant SGD learning rate 0.01 and greedy decoding. Final fixed real-test perplexity is **17.726 / 164.413 / 1,408.607**, respectively, from shared baseline 17.958.

The [per-update analysis](generated/update-gradients-20261114/report.md) recovers and verifies all 2,370 saved diagnostic records and their per-example real-validation probe arrays. It includes six square figure families, cycle/update CSVs, document-bootstrap intervals and immutable source hashes.

In the 99% condition, including real data helps the probe in all 563 updates containing real examples; synthetic data hurts it in 778/790 updates. The weighted real correction strengthens over training. One real example in the final 16-example batch causes especially large corrections at cycles 6 and 10, explaining much of the jagged endpoint behavior on the probe. This remains one seed and a local-intervention analysis; it does not establish permanent stabilization or an alternate full trajectory.

- [Completion and provider status](../../docs/validation/2026-11-14_sgd-greedy-full-completion-status.json)
- [Per-update analysis receipt](../../docs/validation/2026-11-14_sgd-greedy-update-analysis.json)
- [Scientific protocol](../../comparisons/smollm2_sgd_greedy_effects/README.md)

Compute is stopped. Working/private recovery storage remains preserved. This analysis restores diagnostic evidence, not full model/optimizer checkpoints.
