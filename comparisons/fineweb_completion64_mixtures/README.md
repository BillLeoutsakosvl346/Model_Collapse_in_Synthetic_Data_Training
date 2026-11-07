# Matched FineWeb continuation mixtures

Two continued-training conditions measure 50% and 20% synthetic targets using the same frozen FineWeb-Edu prompt bank as the real-versus-synthetic comparison. Both begin independently at original pretrained OPT-125M. Each condition generates its selected continuations using its own preceding checkpoint, then trains that checkpoint on the completed mixture.

## Protocol

| Setting | 50/50 | 20/80 |
| --- | ---: | ---: |
| Fresh prompt chunks | 10 | 10 |
| Examples per chunk | 10,000 | 10,000 |
| Original real targets | 5,000 | 8,000 |
| Generated targets | 5,000 | 2,000 |
| Scored target tokens per epoch | 640,000 | 640,000 |
| Epochs per chunk | 1 | 1 |

Each input contains BOS, 64 real prompt tokens and a 64-token continuation. Attention covers all 129 positions. Only the continuation contributes to loss. All 10,000 original prompts appear once per cycle. Generation receives BOS plus the selected prompt and uses five deterministic beams, exactly 64 new tokens and repetition penalty 1.0.

For each cycle, NumPy `SeedSequence([42, cycle, 6401])` produces a permutation of original row indices. Its first 2,000 rows receive generated targets in both conditions. The next 3,000 also receive generated targets in 50/50. Generated continuations are scattered back to their original prompt rows. Remaining targets retain their original source continuation. The existing seeded training shuffle determines batch order.

Each completed mixture is frozen before its training epoch. Model, AdamW, global cosine schedule and training RNG state continue within each condition. Learning rate decays from 5e-6 to 1e-6 across 790 updates. Microbatch is 32, effective batch 128, with BF16 autocast and FP32 parameters. Terminal epoch weights supply evaluation and the next cycle's generation/training.

The pinned bank has ten document-disjoint chunks, fixed validation/test collections of 7,322/7,151 pairs and a fixed 256-document probe. Document IDs, normalized URLs/text and exact sliding 64-token source passages are separated within the experiment. Bank identity and preprocessing follow the [source comparison](../fineweb_real_vs_synthetic_completion64/README.md).

## Evidence and recovery

Selection permutations, source labels, generated-row mappings, real/generated/mixed target hashes, generation manifests, input masks and checkpoint/optimizer parent identities are saved and verified before reuse. The training engine scores exactly 64 targets per example. A completed stage includes an immutable file seal. Private backups include sealed nested probe journals alongside model and optimizer state.

One detached coordinator runs `mixed50`, then `mixed20`, with separate logical arm directories and optimization chains. The parent owns one GPU, 60 GB volume, supervisor, watchdog and budget association. Each handoff requires complete verified stages and a pinned private backup. Interrupted generation reuses verified shards. Unfinished training recovers from its verified starting state.

## Commands

```powershell
$env:HF_HUB_CACHE = "C:\hf"
$comparison = "comparisons/fineweb_completion64_mixtures"
python -m model_collapse smoke --config "$comparison/smoke.yaml"

# Full comparison follows a separate launch instruction
python -m model_collapse launch --config "$comparison/full.yaml" --generations 10

python -m model_collapse status COMPARISON_ID
python -m model_collapse resume COMPARISON_ID
python -m model_collapse fetch COMPARISON_ID --metadata-only
python -m model_collapse report COMPARISON_ID
python -m model_collapse validate COMPARISON_ID
python -m model_collapse cleanup COMPARISON_ID

# After matching reference evidence has been retrieved
python -m model_collapse report COMPARISON_ID --reference-comparison 20261106T193918Z-full-completion64
```

The optional reference report checks source-bank identity, pretrained weights, held-out target counts, optimization, decoding, hardware/software settings and cycle scope before combining the 0/20/50/100% curves. Generated analysis records input/output hashes and uses square figures. Primary continuation perplexity, secondary full-text perplexity and fixed-probe output statistics retain their respective denominators.

## Acceptance and allocation

The smoke uses three cycles per condition, 320 training pairs per cycle, 64 validation/test pairs and a 64-prompt probe. Synthetic counts are exactly 160 and 64. Generation shards contain 32 examples. Acceptance exercises six terminal epochs, own-checkpoint generation, nested selections, interrupted workers/coordinator, complete private cold restores, reproduced evaluation and independent watchdog/provider shutdown.

The comparison has a separate USD 10 allocation with USD 2 reserved and a combined USD 0.50 validation cap. Launch verifies private capacity across historical large objects and reserves 40 GB additional space plus a 10 GB storage margin. The user explicitly authorized one separate validation GPU alongside the protected real/synthetic comparison. Ownership checks bind that exception to its registered Pod/volume association. The [authorization receipt](../../docs/validation/2026-11-06_completion-mixtures-parallel-validation-authorization.json) records its scope.

Readiness and measured forecasts are recorded in the [handoff](../../docs/fineweb-completion-mixtures-handoff.md). One seed gives descriptive trajectories.

## Validated setup

[GPU acceptance `20261106T235904Z-smoke-completion64-mixtures`](../../docs/validation/2026-11-07_completion-mixtures-setup.json) passed all sixteen checks, six terminal epochs, both full private restorations and provider-confirmed watchdog shutdown. Temporary resources were removed. The 466-test full suite and an additional matched-reference reporting regression passed. Validation cost is estimated conservatively at USD 0.21386. The combined full forecast is 3.01 hours / USD 1.74 running cost, with storage retained separately. Full training was subsequently authorized and is running as recorded below.

## Full execution

Comparison `20261107T004033Z-full-completion64-mixtures` launched on 7 November at 00:40 UTC, with detached progress and the first complete full-data checkpoint/private backup verified at 00:52 UTC. A separate RTX PRO 4000 Blackwell runs 50/50 then 20/80 sequentially, alongside the protected original real/synthetic comparison. The [launch receipt](../../docs/validation/2026-11-07_completion-mixtures-full-launch.json) records resources, source identity, exact first-stage counts, forecast and independent deadline. The [full parallel authorization](../../docs/validation/2026-11-07_completion-mixtures-parallel-full-authorization.json) permits this second full GPU. Use the existing comparison ID for status and recovery. Avoid a duplicate launch.
