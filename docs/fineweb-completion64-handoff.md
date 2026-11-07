# Matched FineWeb completion experiment handoff

**Latest status — resumed at 23:03 UTC on 6 November.** The [resume receipt](validation/2026-11-06_completion64-resumed-after-storage-repair.json) verifies fresh R4 training and its pinned private backup on replacement Pod `udbovqu03vp3h0`, original volume `s9pmtnqkir`. The laptop can close. R1–R3 were reused after a full 4.56 GB restoration and unchanged-stage checks. The free project organization now hosts private checkpoints and optimizer states. Conservative completion is around 02:50 UTC on 7 November. The original [quota failure](validation/2026-11-06_completion64-failure-status.json) remains recorded. Earlier launch estimates below describe the initial attempt.

Setup passed on 6 November 2026. The full comparison was subsequently authorized in [the launch instruction](validation/2026-11-06_completion64-full-authorization.json). The [verified full launch](validation/2026-11-06_completion64-full-launch.json) is `20261106T193918Z-full-completion64`. Use [current state](current-state.md) for execution status.

## What runs

- Both arms start from identical original pretrained OPT-125M weights.
- Ten fresh, document-disjoint chunks each contain 10,000 pairs. Each pair has a 64-token real prompt and a 64-token continuation.
- The real arm trains on the actual source continuations. The synthetic arm generates all targets for a chunk using its preceding checkpoint, freezes them, then trains that checkpoint.
- Inputs contain BOS plus both segments. Attention covers all 129 positions. Only the 64 continuation targets contribute to loss.
- Each arm executes ten one-epoch cycles and carries independent model, AdamW, global cosine schedule and training RNG state. There are 790 updates and 6.4 million scored training-token exposures per arm.
- One detached coordinator runs real first, then synthetic on one RTX PRO 4000 Blackwell and a 60 GB Standard volume.

## Data and measurements

The [frozen data receipt](validation/2026-11-06_completion64-data.json) records all 100,000 reconstructed pairs, document/passage separation, source locations and pinned private backup. The local bank is approximately 164 MB. Validation contains 7,322 pairs / 468,608 targets. Test contains 7,151 pairs / 457,664 targets.

Primary curves score the same fixed real continuations. Secondary 512-token full-text evaluations cover FineWeb, WikiText-103, C4, PG-19 and AG News. A fixed document-stratified validation probe supplies diversity, entropy, repetition, duplicates and newline statistics. Loss, example-NLL median/P95, training NLL, optimization loss, learning rates, timing and training memory are saved separately.

## Validation and recovery

- **429 offline tests and all 13 GPU/provider checks passed.**
- Three cycles per arm executed six actual terminal epochs with 256 pairs per cycle and 64 held-out pairs per split.
- Both optimizer chains, original starting weights, visible prompts, continuation-only targets, terminal checkpoint reloads and direct NLL calculations passed.
- Deliberate generation, training and coordinator interruptions recovered. Completed stages stayed unchanged. SSH disconnection preserved progress.
- Both private trained-arm backups were fully restored on the GPU, including optimization states, and held-out evaluation was reproduced.
- Independent watchdog shutdown and provider EXITED state were verified. All temporary validation resources were removed.

The [validation report](../reports/20261106T191336Z-smoke-completion64/report.md) includes square plots, metrics and [actual real/synthetic training examples](../reports/20261106T191336Z-smoke-completion64/dataset_examples.json). The [setup receipt](validation/2026-11-06_completion64-setup.json) records exact hashes and restoration coverage. Accepted runtime is `35fbfb2f2a278f9cf2a58b1dbfeecaf90389c48241eed25db20b183cbcb59497` and protocol is `b22e6d66d82ea7393ee265b757a0d11f9e1d16f5519059faf798e0e285ab45ac`.

Setup exposed transient SSH metadata and coordinator CUDA-cache retention. Both were fixed and retested. The corrected startup check requires a completed measured stage. The shared 0.5 GB baseline was also recovered locally for cleanup verification. Trained-arm checkpoint and optimizer bulk remain in private cloud backups.

## Timing and budget

| Item | Conservative estimate |
| --- | ---: |
| All setup validation/recovery attempts | USD 0.29614 |
| Full twenty-epoch comparison | 3.80 hours |
| GPU and running storage | USD 2.20 |
| Other retained storage during that run | USD 0.12 |
| Seven days of the new 60 GB volume | USD 0.97 |

The USD 10 comparison allocation includes USD 2 reserved for storage/shutdown. Combined validation cost stayed below USD 0.50. Technical readiness and affordability passed. Launch rechecks stock, credit, ledger and padded watchdog coverage.

## Commands and laptop closure

```powershell
$comparison = "comparisons/fineweb_real_vs_synthetic_completion64"
python -m model_collapse launch --config "$comparison/full.yaml" --generations 10
python -m model_collapse status COMPARISON_ID
python -m model_collapse resume COMPARISON_ID
python -m model_collapse fetch COMPARISON_ID --metadata-only
python -m model_collapse report COMPARISON_ID
python -m model_collapse cleanup COMPARISON_ID
```

Use the existing comparison ID after launch. The launcher returns success after a complete measured training stage, detached progress and advancing supervisor/watchdog heartbeats. The laptop can close after that verified return. Training, arm handoff, private backups and GPU shutdown continue remotely. The 60 GB volume remains after full completion for recovery.

Metadata-only fetching keeps trained-arm bulk in cloud storage. Cleanup requires verified recovery, including the coordinator's shared baseline artifacts. Interrupted full work resumes from verified checkpoints and optimization state. The repaired run uses the host-side `python -m scripts.operations.resume_completion64 RUN_ID` entry point after stopped-state confirmation and storage admission. The active comparison needs no additional command. Completed full comparisons cannot be resumed.

The smoke establishes pipeline validity. Scientific conclusions require the full trajectories. This comparison has one seed. Within-experiment document/passage separation is verified; original pretraining exposure and human authorship remain unaudited.
