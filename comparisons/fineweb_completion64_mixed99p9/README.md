# FineWeb 99.9% synthetic / 0.1% real targets

Use the same pretrained OPT-125M, pinned ten-chunk FineWeb completion bank and fixed held-out real continuations as the completed 99% experiment. The only full-experiment scientific change is the fraction of continuation targets replaced.

For each fresh 10,000-pair chunk, the condition's own preceding checkpoint generates exactly 64 tokens for 9,990 selected real prompts. Retain the original real continuations for the other 10 prompts, scatter generated targets back into their original rows, freeze the mixture, then train the preceding checkpoint for one terminal epoch. Carry model weights, AdamW, the global cosine schedule and training RNG through all ten cycles.

Every input contains BOS, 64 real prompt tokens and 64 continuation tokens. Loss scores the continuation only. Synthetic percentages refer to continuation targets; prompts remain real in every condition. Preserve seed 42, nested selection streams, five deterministic beams, repetition penalty 1, batches, learning rate and all existing metric definitions. Match RTX PRO 4500 Blackwell and the inspected pinned image.

| Profile | Cycles | Synthetic / real pairs per cycle | Scored targets per cycle |
| --- | ---: | ---: | ---: |
| Full | 10 | 9,990 / 10 | 640,000 |
| Smoke | 3 | 999 / 1 | 64,000 |

The usual 320-pair smoke would round this fraction to 320 synthetic / zero real. This condition explicitly uses 1,000 smoke pairs so GPU acceptance exercises both sources. Historical smoke scopes retain their original defaults. Full and smoke configurations bind this declared scope to the same new protocol. An older 99% certificate cannot certify this condition or its changed validation runtime.

The user requested the new condition on 7 November and explicitly [approved](../../docs/validation/2026-11-07_completion-mixed99p9-authorization.json) a USD 3.85 total setup allowance, then validation followed by full launch if checks pass. The condition has a USD 5 allocation including USD 2 reserved and at most USD 0.30 across validation attempts, inside the unchanged USD 64.03635 cumulative ceiling. No credits or payment changes are authorized. Preparation and offline checks are in progress; no paid validation or full launch has occurred.

The [dedicated 30 GB backup volume](../../docs/validation/2026-11-07_completion-mixed99p9-backup-volume.json) is `vx78hsoirz` in EU-RO-1. Both configurations use its verified descriptor. A private checksum-verified upload/download probe and 19.89 GB additional capacity admission plus a 3 GB margin passed. The prior backup volume is too full to admit this new condition safely; it remains unchanged. The launcher must verify actual capacity, certificates and budget before creating compute. Preserve all previous resources and backups.

After the remaining authorization, placement and GPU acceptance steps:

```powershell
python -m scripts.operations.validate_high_fraction_completions --percent 99.9
python -m scripts.operations.launch_high_fraction_completions --percent 99.9
```

These commands are prepared interfaces, not evidence that either run has launched. Use the returned comparison ID for existing status, resume, fetch, report and recovery-gated cleanup. Expected full runtime is similar to 99% (approximately two hours); refresh its conservative forecast from the new acceptance benchmark. One seed provides a descriptive trajectory.
