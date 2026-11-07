# Validated FineWeb continuation mixtures

**Both full mixture conditions are complete**, verified 7 November. The original real/synthetic comparison also completed. The [four-condition analysis receipt](validation/2026-11-07_fineweb-four-fraction-completion-analysis.json) verifies completed stages, matching scientific settings, saved metrics and provider shutdown for both GPUs. Preserve full-run recovery storage. The launch instructions below are historical.

**Full launch verified on 7 November at 00:52 UTC.** `20261107T004033Z-full-completion64-mixtures` runs 50/50 followed by 20/80 on its second GPU, alongside the original real/synthetic comparison. The first full mixed epoch, 79 optimizer updates, checkpoint, pinned private backup and detached supervisor/watchdog are verified in the [launch receipt](validation/2026-11-07_completion-mixtures-full-launch.json). **The laptop can close.** Conservative finish is around 03:41 UTC/London, approximately three hours from provisioning, with USD 1.74 running cost plus USD 0.97 seven-day retention. The independent budget deadline is 09:14 UTC. These forecasts remain estimates. Automatic GPU shutdown follows completion, failure or budget exhaustion.

**Setup passed on 7 November 2026.** The subsequent full launch is recorded above. [Acceptance `20261106T235904Z-smoke-completion64-mixtures`](validation/2026-11-07_completion-mixtures-setup.json) passed the 466-test full suite, an additional matched-reference reporting regression, fresh independent review and all sixteen GPU/provider checks.

## What the full comparison does

- Start 50/50 and 20/80 independently from original pretrained OPT-125M.
- Use the same ten fresh 10,000-pair FineWeb chunks and fixed real evaluation collections.
- In each chunk, generate 5,000 or 2,000 continuations using that condition's own preceding checkpoint. Keep the remaining 5,000 or 8,000 original real targets.
- Train that checkpoint for one terminal epoch on the completed 10,000-example mixture, scoring only its 64 continuation tokens.
- Carry each condition's model, optimizer, global schedule and RNG state for ten cycles. Run 50/50 then 20/80 automatically under one detached coordinator.

The [methodology](../comparisons/fineweb_completion64_mixtures/README.md) defines exact nested selections, checkpoint/data lineage, metrics and recovery.

## What passed

- Six actual GPU epochs, with 320 pairs per cycle and exact synthetic counts of 160 and 64.
- Both optimizer chains reached nine updates. Every terminal checkpoint reloaded correctly and weights changed.
- Original-row target reconstruction, own-checkpoint generation, visible prompts and exactly 64 scored targets per example.
- Injected generation/training/coordinator exits, verified shard reuse and unchanged completed stages.
- Both private backups fully restored, covering **9,022,500,703 bytes**, with every sealed stage verified and held-out evaluation reproduced.
- Detached progress, independent watchdog shutdown and provider EXITED state, followed by removal of the temporary Pod and 60 GB volume.

The [validation report](../reports/20261106T235904Z-smoke-completion64-mixtures/report.md) contains square plots, CSV/JSON metrics and [actual prompt/real/generated examples](../reports/20261106T235904Z-smoke-completion64-mixtures/dataset_examples.json). Training and data-bank implementations retain their prior bytes. Accepted runtime is `98f59dfdc0edc3205fe24fcb72b7202537b0efbe77eeaa0f20c3be0f8f866e37` and protocol is `c411a4e72d8215365c4f0acc84809bcacd3f20af8bdbe627d33d714af6660c13`. The [source bundle](validation/2026-11-06_completion-mixtures-source-bundle.json) is privately pinned and checksum-verified.

## Time, cost and admission

| Item | Result or conservative estimate |
| --- | ---: |
| Validation cost, including shutdown/storage reconciliation | USD 0.21386 |
| Combined validation cap | USD 0.50 |
| 50/50 full arm | 1.81 hours |
| 20/80 full arm | 1.19 hours |
| Combined full comparison | 3.01 hours / USD 1.74 running cost |
| Seven-day 60 GB volume retention | USD 0.97 |

The running-cost forecast includes a 1.5 throughput margin, checkpoint/backup overhead and setup. The USD 10 allocation includes USD 2 reserved. Affordability passed using current credit, ledger and private capacity. Provider itemized billing has not settled, so validation cost is labelled an estimate. The account-wide debit also includes the protected production GPU and historical storage.

The original full real/synthetic comparison was observed RUNNING after validation cleanup, on Pod `udbovqu03vp3h0` and volume `s9pmtnqkir`. Full launch rechecked active ownership, matching stock, current credit and private storage capacity.

## Windows commands

Recovery exposed Windows' default cache path-length limit. The user-level Hugging Face cache is configured at `C:\hf`. Set it explicitly in an already-open terminal before Python starts.

```powershell
$env:HF_HUB_CACHE = "C:\hf"
$comparison = "comparisons/fineweb_completion64_mixtures"

$run = "20261107T004033Z-full-completion64-mixtures"
python -m model_collapse status $run
python -m model_collapse fetch $run --metadata-only
python -m model_collapse report $run

# Recovery after an unfinished run has stopped
python -m scripts.operations.launch_parallel_completion_mixtures --resume $run

# Cleanup after verified recovery
python -m model_collapse cleanup $run

# After matched completed reference evidence is retrieved
python -m model_collapse report $run --reference-comparison 20261106T193918Z-full-completion64
```

The full launcher proves detached supervision, watchdog health and actual new training before returning. Full training is now active. The smoke establishes pipeline validity. One seed gives descriptive trajectories.
