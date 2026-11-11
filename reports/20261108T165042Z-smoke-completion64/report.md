# Repeated-chunk GPU validation

**Pipeline validation only.** This is not the ten-epoch scientific experiment.

The production coordinator executed three terminal epochs per arm on the same 256 bank-cycle-01 pairs, with 64 validation pairs, 64 test pairs and a fixed 64-prompt probe. The real arm reused the original targets; the synthetic arm regenerated targets from its own preceding checkpoint. Both retained independent model/optimizer/scheduler/RNG chains. Together they executed six epochs; each arm performed six optimizer updates.

## Verification

- all handoffs and optimizer states: passed.
- both private restores: passed.
- completed stages unchanged: passed.
- continuation only loss: passed.
- coordinator recovery: passed.
- detached progress verified: passed.
- direct masked nll: passed.
- fixed chunk inputs: passed.
- fixed real anchor: passed.
- generation shard recovery: passed.
- independent budget guard and provider stop verified: passed.
- matched initial weights and prompts: passed.
- recursive same prompt targets: passed.
- restored evaluation: passed.
- six terminal epochs: passed.
- training recovery: passed.

The pinned private restores verified **9,021,414,358 bytes** across both complete three-stage child snapshots and reproduced held-out loss. Root baseline/probe bytes were also recovered locally before temporary cleanup. The provider-confirmed watchdog reason was `validation_budget_guard`; temporary Pod `pa1qlnp9tk6w9n` and working volume `lb89qoa6e0` were removed. The new full-run backup store is retained separately.

## Evidence and spending

- [Acceptance receipt](../../docs/validation/2026-11-08_repeated-chunk-setup.json).
- [Frozen bank and first-chunk identity](../../docs/validation/2026-11-08_repeated-chunk-data.json).
- [Small validation metrics](metrics.csv); 4,096 scored test tokens per checkpoint.
- Runtime: `443b7a31a72910914bcd337919ffdc846294904b64b7a3ff7a7d6d70305d9580`.
- Protocol: `55d0e90184308cbfed3e68577b2623c7bd5d89ac9c6fbb5453e1d883133b6552`.
- Conservative validation allocation spend: **USD 0.27601**; provider billing was not yet finalized. Account balance deltas also include unrelated retained storage and are not task-only debits.

The full comparison has separate launch/completion records in [current state](../../docs/current-state.md). Smoke trajectories establish execution, lineage and recovery, not evidence of collapse or overfitting.
