# Reset-to-real-S0 condition

**State:** full run `20261027T211852Z-full-reset-s0` completed on 28 October at 02:56 UTC/London; [shared-start comparison analysed](../../../reports/20261027T211852Z-full-reset-s0/report.md). [Completion record](../../../docs/validation/2026-10-28_reset-full-completion.json) · [Historical launch record](../../../docs/validation/2026-10-27_reset-full-launch.json). Condition ID: `reset_to_real_s0`. Initialisation policy: `reset_s0`.

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

The [final-source acceptance](../../../reports/20261027T203952Z-smoke-reset-s0/report.md) passed 14 GPU/provider checks after 86 offline tests. It verified imports, two actual training updates from S0, both injected exits, shard recovery, private checkpoint restoration/re-evaluation and independent shutdown. Temporary resources were deleted. The full-run estimate from warmed production-batch benchmarks is about 8.3 hours / USD 4.75, with a conservative margin.

Setup ended after validation. The user subsequently explicitly authorized the separate full launch on 27 October at 21:18 UTC/London; it completed all nine new stages after 5.62 hours, with five epochs per stage and automatic GPU shutdown. The report audits metadata, all new handoffs and small output probes; bulk restore remains separate. A later runtime change requires fresh acceptance. See the [setup handoff](../../../docs/implementation-handoff.md). Do not launch a duplicate or treat smoke results as full-run findings.
