# Continued-weight condition

**State:** full run completed. Presentation alias: Week 2. Legacy mode and initialisation policy: `continued`.

S0 is the saved real-data-trained G0 from Week 1, **not** Week 1's final G10. Each selected model generates the next synthetic dataset and provides the initial weights for training on that dataset. Optimizer and scheduler restart each cycle; weights carry.

```text
Saved real S0 -> generate D1
S0 + D1 -> S1 -> generate D2
S1 + D2 -> S2 -> ...
```

- Full result: [20261026T192740Z-full-continued](../../../reports/20261026T192740Z-full-continued/report.md).
- Metrics: [complete CSV](../../../reports/20261026T192740Z-full-continued/metrics.csv).
- Historical GPU acceptance: [20261026T182856Z-smoke-continued](../../../reports/20261026T182856Z-smoke-continued/report.md).
- Canonical definitions: [full.yaml](full.yaml), [smoke.yaml](smoke.yaml).
- Historical configurations remain at [configs/full-continued.yaml](../../../configs/full-continued.yaml) and [configs/smoke-continued.yaml](../../../configs/smoke-continued.yaml).
- Shared [methodology](../../../docs/methodology.md) and [metric definitions](../../../docs/metrics.md).

The ten-cycle run completed on 27 October 2026. Real validation selected epoch 1 in all ten cycles: fifty epochs ran, but only ten selected epochs were added along the inherited synthetic-training chain. One seed and the hardware difference from Week 1 limit causal comparisons.

The [new shared-start comparison](../../../comparisons/shared_s0_s1_reset_vs_continue/README.md) reuses this run's exact S0, S1 and D2 as its common prefix. It does not overwrite or restart this completed run.
