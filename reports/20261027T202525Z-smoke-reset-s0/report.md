# Shared-start reset branch: GPU acceptance

Run: `20261027T202525Z-smoke-reset-s0`.

**Pipeline validation only. No full-data reset experiment was launched.**

This successful acceptance covers its recorded source version. A subsequent source version is validated separately; this receipt must not authorize that changed runtime.

## What ran

Imported the exact Week 2 S0 and S1 checkpoints and D2 source dataset. The two new cycles, 2 and 3, each trained from fixed S0 with a fresh optimizer for one epoch on 256 continuation-only 64-token blocks. Active real validation/test sets each have 64 blocks. Smoke D2 is explicitly the first 256 rows of the pinned full D2; full execution uses all 37,946 rows.

Imported S0/S1 metrics retain their original full-data measurement scope. They are not directly comparable to the tiny smoke evaluation. New cycles have 4,032 test targets each.

All **14 GPU/provider acceptance checks passed**: exact shared-prefix import, distinct generator/initializer lineage, actual weight changes, both injected worker exits, shard recovery, unchanged completed stage, private backup restore, checkpoint re-evaluation, direct NLL calculation and actual budget-guard shutdown.

Restored final-checkpoint test PPL: **38.07099949**. Direct shifted-NLL maximum discrepancy: **0**.

| New cycle | Real-test PPL, 64 blocks | Initializer | Data generator |
| --- | ---: | --- | --- |
| 2 | 38.15908414 | fixed S0 | preceding selected model |
| 3 | 38.07099949 | fixed S0 | preceding selected model |

## Execution and recovery

GPU: NVIDIA RTX PRO 4000 Blackwell, EU-RO-1, quoted $0.57/hour. Pipeline completion took **6.72 minutes** from provisioning. Observed account debit through cleanup: **$0.053039**, including other retained storage and subject to provider billing timing.

Temporary Pod `znaxmymhfagg24` and volume `i6ct6780vi` were deleted after recovery verification. Historical full-run resources were preserved. Private backups remain in the existing model/dataset repositories under this run ID.

Local metadata recovery: 105 files / 14,873,593 bytes. Bulk objects were restored and checked on the GPU, not downloaded to the laptop. Runtime SHA-256: `686e0ff403f6cd9d35f153b3a8fdbe303958e41e10b377de84f3556fb93afb57`.

## Measured remaining-work estimate

Warmed training: 34,402 target tokens/s; deterministic five-beam generation: 1,303 new tokens/s. Production batch sizes: effective 128, microbatch 32, generation batch 32.

Nine new full training stages and eight full generation stages plus the final probe: **8.27 hours / $4.76** conservatively, including the forecast's 1.5x compute margin and upload/setup allowances. This is an estimate, not a completed experiment; retained-storage reserves and prior setup spend are separate.

[Execution receipt](execution_receipt.json) · [Raw records](records/) · [Generated analysis](figures/report.md)

Regenerate analysis with the maintained report command. Export this acceptance from verified local metadata using `python scripts/analysis/export_acceptance.py RUN_ID`; the exporter refuses to overwrite an existing published acceptance.
