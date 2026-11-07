# FineWeb 99% synthetic / 1% real

Continue the matched completion experiment with 9,900 synthetic and 100 real targets in each fresh 10,000-pair chunk. All ten chunks, source revisions, prompts, held-out evaluations and optimizer settings match the 85/90/95% conditions. Start independently from original pretrained OPT-125M.

For each cycle, use the preceding checkpoint to generate exactly 64 continuation tokens for the selected 9,900 real prompts. Keep the source continuations for the remaining 100 prompts. Preserve row positions, then apply the existing training shuffle. Train that preceding checkpoint for one terminal epoch with continuation-only loss. Carry weights, AdamW, the global scheduler and training RNG into the next fresh chunk. Ten cycles give 790 optimizer steps and 6.4 million scored training-target exposures.

Selection uses the same `SeedSequence([42, cycle, 6401])` permutation, so each cycle's synthetic selection includes the corresponding 95% selection. The mixture and generated outputs are frozen before training and verified across resume.

| Profile | Cycles | Synthetic / real pairs per cycle | Evaluation / probe |
| --- | ---: | ---: | --- |
| Full | 10 | 9,900 / 100 | Complete fixed held-out pairs / 256 prompts |
| Smoke | 3 | 317 / 3 | 64 validation and test pairs / 64 prompts |

Smoke uses the established 320-pair scope. Half-up rounding gives an effective 99.0625% synthetic fraction; the full experiment is exactly 99%.

Pin RTX PRO 4500 Blackwell and the same official image to match the 85/90/95% runs. Reuse the pinned bank and dedicated 30 GB private content-addressed backup store, preserving all historical objects. Confirm capacity, recorded allocation and remaining allowances before paid work.

This is a standalone condition. The old three-job parallel authorization and its GPU certificates do not certify it. The dedicated 99% authorization is recorded separately. The user subsequently authorized paid validation and full launch. Actual acceptance/launch status belongs in current state and dated receipts. The reorganised runtime also needs its own GPU acceptance. Per-session caps in the configuration are limits, not spending authorization.

The [GPU acceptance](../../docs/validation/2026-11-07_completion-mixed99-setup.json) passed all sixteen checks, including cold restoration and shutdown. Validation cost USD 0.21584; its temporary Pod/volume were removed. The separately authorized [full run](../../docs/validation/2026-11-07_completion-mixed99-full-launch.json) `20261107T183326Z-full-completion64-mixed99` completed all ten cycles at 20:17 UTC on 7 November. Its [completion and comparison receipt](../../docs/validation/2026-11-07_completion-mixed99-completion-analysis.json) verifies ten model/optimizer/generator chains, exact recorded counts, pinned small metrics and provider shutdown. Final test PPL is 82.67730; full bulk restoration remains separate. See [current state](../../docs/current-state.md); do not resume or duplicate completed compute.

Recorded validation/launch entry points:

```powershell
python -m scripts.operations.validate_high_fraction_completions --percent 99
python -m scripts.operations.launch_high_fraction_completions --percent 99
```

Use the returned comparison ID for status, resume, fetch, report and recovery-gated cleanup. Retain the existing primary/secondary perplexity and fixed-probe metric families. One seed produces a descriptive trajectory.
