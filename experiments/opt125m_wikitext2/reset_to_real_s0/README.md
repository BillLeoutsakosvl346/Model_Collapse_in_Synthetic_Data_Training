# Reset-to-real-S0 condition

**State:** full run planned; no full result claimed. Condition ID: `reset_to_real_s0`. Initialisation policy: `reset_s0`.

This condition shares the first training cycle with the completed [continued run](../continued_from_real_s0/README.md), then resets training to the **fixed real-data-trained S0** on every new stage. It does not reset to untouched pretrained OPT.

```text
Shared import from 20261026T192740Z-full-continued:
S0 -> generate D1 -> train S0 on D1 -> S1 -> generate D2

New work:
S0 + D2 -> A2 -> generate DA3
S0 + DA3 -> A3 -> ...
S0 + DA10 -> A10
```

S0/S1 evaluations remain references to their original measurement event. They are not rerun or silently replaced with measurements from different hardware. Cycle 2 consumes the **identical saved D2** used by Week 2's S2, but starts from S0 rather than S1. At cycle 3 onward, each arm's own selected model generates its next dataset.

- [Comparison and shared artifact identities](../../../comparisons/shared_s0_s1_reset_vs_continue/README.md).
- [Full configuration](full.yaml): imported cycles 0/1, nine new cycles 2-10, five epochs per new cycle.
- [Smoke configuration](smoke.yaml): two new one-epoch stages, cycles 2/3, small data subsets.
- Shared [methodology](../../../docs/methodology.md), [metrics](../../../docs/metrics.md) and [operations](../../../docs/operations.md).

Every new training stage records its initializer, generator, training tokens and selected output separately. All new initial tensor hashes must equal the imported S0 hash; input datasets must match their generator lineage. The optimizer and scheduler restart, as in Week 2.

The intended full comparison matches Week 2's RTX PRO 4000 Blackwell/software, packing, prompts, frozen real evaluation, decoding, token budget, seed, precision and selection rule. If matching hardware is unavailable, leave the full launch pending rather than silently substituting an execution environment.

Run the explicit smoke acceptance before calling this condition ready. Setup ends after validation and does not start the full run. Do not treat a folder, configuration or previous continued-mode smoke certificate as evidence that the changed workflow passed.
