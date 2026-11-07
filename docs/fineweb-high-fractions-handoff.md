# FineWeb high-synthetic-fraction campaign

Three independent continued-training conditions use the same frozen completion64 bank and original pretrained OPT-125M. Each processes ten fresh 10,000-pair chunks, one terminal epoch per chunk. Inputs are BOS + 64 prompt tokens + 64 continuation tokens; only continuation tokens contribute to loss. Each condition carries its own model, AdamW, global cosine schedule and training RNG.

| Synthetic fraction | Synthetic / real pairs per cycle | Full comparison ID |
| --- | --- | --- |
| 85% | 8,500 / 1,500 | `20261107T150834Z-full-completion64-mixed85` |
| 90% | 9,000 / 1,000 | `20261107T151427Z-full-completion64-mixed90` |
| 95% | 9,500 / 500 | `20261107T151610Z-full-completion64-mixed95` |

Each cycle uses the nested `SeedSequence([42, cycle, 6401])` selection. Selected prompts are answered by that condition's preceding checkpoint, with five deterministic beams and exactly 64 new tokens. Generated continuations replace their original targets at the same row positions; other real targets remain unchanged. All 10,000 rows and 640,000 scored targets are retained. Evaluation collections and the 256-prompt probe stay fixed.

## Execution and recovery

The three jobs run on separate matching RTX PRO 4500 Blackwell GPUs, with separate 60 GB working volumes, supervisors, watchdogs, logical arm directories and optimizer chains. Admission/deployment is serialized; first-stage observation and training overlap. The scoped admission guard allows only this campaign's verified resource associations, with at most three active Pods and no duplicate condition. A single ledger tracks separate compute/storage intervals and reserves sibling allowances before launch.

RTX PRO 4500 was the user-approved capacity fallback. The earlier fractions used RTX PRO 4000 Blackwell. The frozen bank, model/data revisions, precision and scientific settings are unchanged; exact numerical equivalence across hardware is not assumed.

Small metadata and immutable indexes remain in the existing private Hugging Face repositories. Bulk checkpoints, optimizer states and arrays use SHA-addressed objects in the task-owned 80 GB Runpod S3 volume `mng23iliig`. Credentials remain in ignored local state and ephemeral remote secrets. Closing the laptop does not affect remote execution. Shutdown follows completion, failure or the funded deadline; full-run working volumes and backup storage remain for recovery.

## Validation and limits

- The 506-test suite and nine additional host startup/recovery regressions passed: 515 distinct offline tests.
- Each condition passed all sixteen GPU/provider checks, including actual updates, mixture reconstruction, interrupted generation/training recovery, unchanged completed stages, persistent optimizer state and detached progress.
- Each private three-stage backup was cold-restored, all stage/optimizer checksums verified and held-out evaluation reproduced. Temporary validation Pods and volumes were removed after recovery, including the relocated 90% source disk.
- Conservative combined validation cost, including failed-allocation and shared-backup storage before full launch, was approximately USD 0.7843. The approved cap is USD 0.90, total setup allowance USD 3.50, campaign ceiling USD 11.75 and cumulative ceiling USD 64.03635, with the existing reserve preserved. No credits were purchased or payment settings changed.
- Measured admission forecasts are approximately 2.15 / 2.23 / 2.52 hours and USD 1.56 / 1.62 / 1.83 running cost. Actual full-run benchmarks and progress may refine these. Seven days of the new 260 GB network-volume storage is approximately USD 4.19; historical retained storage bills separately.

**Verified at 15:32 UTC/London on 7 November: all three full jobs are healthy, their first checkpoints/backups are verified, and the laptop can close.** Current full-job benchmarks forecast completion around 17:19 / 17:26 / 17:27 London, approximately USD 4.77 combined running cost. These are estimates. See the [full launch receipt](validation/2026-11-07_completion-high-fractions-full-launch.json).

The [setup receipt](validation/2026-11-07_completion-high-fractions-setup.json), [provisioning receipt](validation/2026-11-07_completion-high-fractions-full-provisioning.json) and [current state](current-state.md) distinguish validation, allocation and verified full-job progress. Full-run checkpoint cold restoration remains separate from first-stage startup verification.

## Commands

Replace `RUN_ID` with one of the comparison IDs above:

```powershell
python -m model_collapse status RUN_ID
python -m model_collapse fetch RUN_ID --metadata-only
python -m model_collapse report RUN_ID
```

Use `resume RUN_ID` only for interrupted unfinished work; completed runs reject resume. Use `cleanup RUN_ID` only after verified full recovery. Preserve the shared backup volume until all three conditions are adequately recovered. Avoid launching duplicate full jobs.

Initial entry points are `scripts/operations/launch_high_fraction_completions.py` and the scoped concurrent-startup helper `scripts/operations/launch_concurrent_high_fraction.py`. Their host source identities are recorded separately from accepted worker runtime `eb6b3104dbe395bbe192ccd14afbc74de6d24d9d22578486804f0c8c1a490e7f`.
