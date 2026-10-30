# Matched five-epoch setup

Accepted task: implement and validate a new shared-start reset-versus-continued pair, then stop before full launch.

## Scientific change

- Preserve the exact existing real-trained S0 and historical continued-run D1.
- All new synthetic stages execute five epochs and **retain epoch 5**, instead of selecting the lowest real-validation loss.
- Train one new common S1 and generate one new D2. Continue starts cycle 2 from S1; reset starts from S0 on the same D2.
- Preserve fixed real prompts/evaluation, token counts, decoding, optimizer/scheduler restarts, batches, precision, seed and metric definitions.
- Complete continued cycles 1–10 and reset cycles 2–10: 19 unique stages / 95 epochs. Historical results remain unchanged.

## Execution and acceptance

- One detached coordinator, physical Pod, volume, ownership tuple, cost interval and independent watchdog deadline; two sibling logical run roots.
- Seal completed continued backup revisions, source/index/configuration hashes and S0/S1/D2 before materializing reset.
- Recover interrupted epochs, generation shards, the source handoff and completion-before-backup without retraining completed stages.
- Strict runtime/protocol acceptance for final-epoch selection; preserve legacy one-epoch smoke/five-epoch best-validation admission.
- Offline regressions, a fresh independent review, then one 256/64/64-block paired GPU smoke: continued stages 1/2 and reset stages 2/3, five epochs each.
- Verify both private restores and checkpoint re-evaluation, real SSH-disconnected progress, automatic provider shutdown and recovery-gated cleanup.

## Limits and handoff

Combined new GPU acceptance is capped at $0.75 within the historical $3 setup allowance, cumulative $25 ceiling and $2 storage/shutdown reserve. No credits or payment changes. Matching hardware/image/software are required; unavailable or unaffordable work remains pending.

Deliver tested commands, passed-check evidence, actual observed acceptance debit with its scope, artifact locations, measured per-arm/combined forecasts and remaining limitations. Full compute requires a later explicit launch instruction. See the [condition](../../comparisons/shared_start_final_epoch5/README.md).
