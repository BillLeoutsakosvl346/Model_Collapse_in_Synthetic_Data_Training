# SmolLM2: 50% and 80% synthetic targets with SGD

This comparison extends the completed 0%/99%/100% SGD experiment with two independently initialized conditions. It uses the exact model-native FineWeb bank and original pretrained SmolLM2-135M, not checkpoints or synthetic targets from another condition.

## Protocol

| Synthetic fraction | Real / synthetic pairs each cycle | Cycles | Optimizer updates |
|---|---:|---:|---:|
| 50% | 5,000 / 5,000 | 10 | 790 |
| 80% | 2,000 / 8,000 | 10 | 790 |

Each cycle uses the next frozen 10,000-pair chunk. Inputs contain BOS, 64 real prompt tokens and 64 continuation tokens; only continuations contribute to loss. Each condition generates selected targets using its own preceding checkpoint, freezes them, then trains for one terminal epoch. Weights and training RNG continue across cycles.

The seeded selection uses `SeedSequence([42, cycle, 6401])`; the first 5,000 and 8,000 permuted row indices are synthetic. Remaining targets are unchanged real continuations, and each prompt appears exactly once. Generated answers remain independent between arms.

Training uses plain SGD at constant **0.01**, without momentum, decay, clipping or scheduler; FP32 with TF32 disabled; microbatch 32 and effective batch 128. Every cycle includes 78 full updates and one 16-pair tail averaged over those 16 pairs, matching the historical condition. Generation and the fixed output probe are greedy, exactly 64 new tokens, repetition penalty 1.0. The verified historical real-only learning-rate calibration is reused; its candidate weights are never initializers.

## Diagnostics and interpretation

- Direct source gradients share the actual training forward graphs and whole-batch scored-token denominator. The ordinary combined gradient alone advances training.
- Every update records source losses, weighted contributions, counts, per-layer/global norms, alignment, reconstruction, and local held-out effects of real-only, synthetic-only and actual mixed updates.
- The fixed 128-pair validation probe never supplies training gradients. Separate fixed validation/test collections and the 256-prompt generated probe remain unchanged.
- `epoch_source_summary.csv` separates mean batch-loss shares from corpus-token-weighted shares, and all updates from full/tail batches. Norms are means of per-update norms. Cancellation compares the actual norm with perpendicular vectors of the measured component strengths; it does not measure quality.
- Effects are local interventions along the actual path. Loss shares are not gradient shares, and no loss-share threshold is asserted to predict stabilization. One seed and a small repeated diagnostic probe limit generalization.

Reports retain square perplexity, diversity and repetition figures, plus source-loss, gradient geometry and intervention figures. Historical 0%/99%/100% comparisons require matching scientific identities; calibration snapshot location is operational, while the immutable receipt hash stays scientific.

## Acceptance and execution

The smoke uses two fresh chunks of 400 pairs per arm: 200/200 and 80/320 real/synthetic, 128 validation/test pairs, the unchanged 128-pair update probe and 32 generated probe prompts. It checks 16 successful updates plus one recorded replay, both terminal cold restores, detached progress and ordinary independent shutdown. Calibration is not repeated. Combined validation is capped at $0.20 / 900 paid seconds.

The authorized full sequence runs 50% then 80% on one matching RTX PRO 4000 Blackwell with one 60 GB working volume and a separate 31 GB backup volume. One coordinator, resource owner and watchdog govern both arms. The task cap is $12 including a $2 reserve: $0.20 validation plus a conservative $11.80 full suballocation. No historical resources may be removed.

From the repository root:

```powershell
python -m model_collapse smoke --config comparisons/smollm2_sgd_greedy_mixtures/smoke.yaml
python -m model_collapse launch --config comparisons/smollm2_sgd_greedy_mixtures/full.yaml --generations 10
python -m model_collapse status COMPARISON_ID
python -m model_collapse fetch COMPARISON_ID --metadata-only
python -m model_collapse report COMPARISON_ID --reference-comparison 20261114T001352Z-full-completion64-mixtures
python -m model_collapse cleanup COMPARISON_ID
```

Full launch requires this suite's newly accepted runtime and pinned benchmark. Launch is authorized automatically after successful review/acceptance and affordability checks. Shutdown does not delete recovery storage; cleanup remains gated on verified recovery. See [current state](../../docs/current-state.md) for recorded execution, not live status.

On this Windows workstation, deeply nested diagnostic paths require the extended-length Hugging Face cache path during fetching: set `$env:HF_HUB_CACHE = '\\?\C:\hf'` and `$env:HUGGINGFACE_HUB_CACHE = '\\?\C:\hf'` before the fetch command. That recovery path was exercised by acceptance; credentials and model tensors remain excluded from Git.

The first full mixed checkpoint can take longer than the launcher's existing 600-second observation window, including generation, training and evaluation. The recorded launch releases only its local observer after verifying the detached processes, then uses the existing `verify_detached(run_id, max_wait_seconds=1800)` function to observe and publish complete-stage evidence. This changes neither remote training nor its watchdog deadline. It is an operational limitation of the default startup window, not a completed-stage claim from heartbeats alone; see the full launch receipt/current state for the actual outcome.
