# Matched FineWeb completion pipeline validation

Acceptance `20261106T191336Z-smoke-completion64` passed all 13 GPU/provider checks and 429 offline tests. Three cycles per arm executed six terminal epochs on 256 pairs per cycle. Each epoch scored 16,384 continuation targets. These small trajectories validate the pipeline.

## Verified checks

- all handoffs and optimizer states.
- both private restores.
- completed stages unchanged.
- continuation only loss.
- coordinator recovery.
- detached progress verified.
- direct masked nll.
- generation shard recovery.
- independent budget guard and provider stop verified.
- matched initial weights and prompts.
- restored evaluation.
- six terminal epochs.
- training recovery.

## Records and figures

- [Square comparison plots, metrics CSV/JSON and same-prompt probe outputs](generated/20261106T193407Z-366d158c/report.md).
- [Real and synthetic training examples](dataset_examples.json).
- [Validation records](records/).

## Runtime and recovery

The accepted source is `35fbfb2f2a278f9cf2a58b1dbfeecaf90389c48241eed25db20b183cbcb59497`, protocol `b22e6d66d82ea7393ee265b757a0d11f9e1d16f5519059faf798e0e285ab45ac`. Both trained-arm private backups were fully restored on the GPU, including optimizer states, and held-out evaluation was reproduced. Detached progress, worker/generation/coordinator recovery and unchanged completed stages passed. The independent watchdog stopped compute for `validation_budget_guard`, and the provider verified shutdown. Temporary Pods/volumes were removed after recovery.

## Setup issues found and fixed

- Transient missing SSH metadata is retried, with the authenticated endpoint retained for the same session.
- The coordinator releases its CUDA allocator cache before spawning training workers. An observed 322 MB coordinator allocation verifies the corrected handoff.
- Startup success now requires a completed measured stage. Earlier failed attempts were recovered and their temporary resources removed.

All validation/recovery attempts together cost USD 0.29614 conservatively, within USD 0.50. This is ledger-estimated resource spending. The recorded account delta also includes historical retained storage.

## Full-run forecast

The warmed production-batch benchmark forecasts 3.80 hours and USD 2.20 running cost, plus USD 0.12 historical storage during the run and USD 0.97 for seven days of the new 60 GB volume. The forecast includes twenty training stages, ten full synthetic generation stages, twenty-one probe sets, evaluation, backup allowances and a 1.5 compute margin. Affordability passed at handoff. Full training awaits a separate launch instruction.
