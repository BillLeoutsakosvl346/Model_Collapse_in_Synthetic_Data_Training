# Pretrained-reset condition

**State:** full run completed. Presentation alias: Week 1. Legacy mode and initialisation policy: `fresh_base`.

Untouched pretrained OPT is evaluated separately. G0 is fine-tuned on real WikiText-2. G0 generates D1; **original pretrained OPT** is fine-tuned on D1 to produce G1. At every later cycle, the preceding selected model generates the next dataset, while training again starts from the identical original pretrained checkpoint.

```text
P + real training data -> G0 -> generate D1
P + D1 -> G1 -> generate D2
P + D2 -> G2 -> ...
```

G0 is a reference baseline, not the training initializer of G1. Its appearance at cycle 0 on a connected curve must not imply carried weights.

- Full result: [20261020T211121Z-full](../../../reports/20261020T211121Z-full/report.md).
- Metrics: [original CSV](../../../reports/20261020T211121Z-full/metrics.csv).
- Canonical definitions: [full.yaml](full.yaml), [smoke.yaml](smoke.yaml).
- Historical configurations remain at [configs/full.yaml](../../../configs/full.yaml) and [configs/smoke.yaml](../../../configs/smoke.yaml).
- Shared [methodology](../../../docs/methodology.md) and [metric definitions](../../../docs/metrics.md).

This preserves the historical meaning of `fresh_base`. It is a different condition from [reset to real-trained S0](../reset_to_real_s0/README.md). Existing measurements, source records and report paths remain unchanged. New paid reruns require an explicit instruction; a completed run is not resumable into a new experiment.
