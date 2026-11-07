# Matched FineWeb real and synthetic completions

The two continued-training arms share original pretrained OPT-125M, ten different FineWeb prompt chunks and a conditional next-token objective. Each cycle contains 10,000 examples and one epoch. Each example has 64 real prompt tokens and 64 target tokens. Inputs contain BOS, prompt and continuation, totalling 129 positions. The prompt remains visible to attention and only the continuation contributes to loss.

The real arm uses the actual next 64 source tokens. In the synthetic arm, the preceding selected checkpoint first generates 64 tokens for every prompt in the next chunk, then trains on that frozen synthetic continuation dataset. Both arms start from identical original pretrained weights and carry their own model, AdamW, scheduler and training RNG state for ten cycles.

## Frozen source bank

- FineWeb-Edu `sample-10BT`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, with the original pinned OPT tokenizer.
- Ten chunks, each with 10,000 within-document source windows of 128 tokens. Source selections are document-disjoint between cycles. Each chunk contains 1,280,000 source tokens and 640,000 continuation targets.
- Seeded shard order and a 10,000-document shuffle buffer. Document IDs, normalized URLs/text and exact sliding 64-token passages are separated from held-out collections and other cycles.
- Existing FineWeb held-out documents supply every complete within-document pair. Validation has 7,322 pairs / 468,608 targets. Test has 7,151 pairs / 457,664 targets. Incomplete final windows are recorded through the saved document/window selections.
- A fixed validation probe uses one selected window from each of 256 randomly selected eligible documents. Both arms use the same probe.
- Original human authorship and OPT pretraining exposure remain unaudited. Freshness applies within this experimental training sequence.

The [data receipt](../../docs/validation/2026-11-06_completion64-data.json) records the pinned private bank, independent reconstruction and exact content hashes. Bulk text and arrays live in ignored local data and private backups. The local `data/fineweb_completion64/examples.json` file shows actual prompts and real continuations.

## Matched settings

| Setting | Value |
|---|---|
| Seed | 42 |
| Checkpoint selection | Terminal epoch 1 |
| AdamW | LR 5e-6, betas 0.9/0.999, epsilon 1e-8, weight decay 0.01 |
| Schedule | Global per-step cosine decay to 1e-6 |
| Batch | Microbatch 32, effective batch 128 |
| Precision | BF16 autocast, FP32 parameters |
| Generation | Five deterministic beams, exactly 64 new tokens, repetition penalty 1.0 |
| Full update count | 79 optimizer updates per cycle, 790 per arm |

The same selected prompt chunk and training permutation are used in both arms at each cycle. Generating and measuring probe outputs leave the persistent training RNG independent of those operations. A controlled offline test using identical targets requires both arms to produce identical weights.

## Measurements and recovery

Primary validation/test perplexity exponentiates the pooled mean negative log-probability of the 64 real continuation tokens per held-out pair. Median/P95 NLL summarize per-example losses. Training loss, evaluated training NLL and learning rates are retained separately.

Secondary scores reuse the existing 512-token full-text test arrays for FineWeb, WikiText-103, English C4, PG-19 and AG News. Their objective and context lengths are recorded separately from primary continuation scores.

Probe statistics include distinct-1/2/3, pooled and within-output token entropy, pooled and within-output repeated 4-grams, duplicate continuations and newline concentration. Their token counts and denominators are retained. These are descriptive one-seed measurements.

One detached coordinator runs real then synthetic on the same GPU and 60 GB volume. Each arm has an independent optimizer chain, checkpoint history and pinned backup. The parent owns the provider resource association and budget deadline. Interrupted synthetic generation reuses verified shards. Interrupted training replays from its verified starting checkpoint and optimization state. Completed stages are verified before reuse.

## Commands

```powershell
$comparison = "comparisons/fineweb_real_vs_synthetic_completion64"
python -m model_collapse prepare --config "$comparison/full.yaml" --root data/fineweb_completion64
python -m model_collapse smoke --config "$comparison/smoke.yaml"

# Full comparison requires recorded authorization and matching GPU acceptance
python -m model_collapse launch --config "$comparison/full.yaml" --generations 10

python -m model_collapse status COMPARISON_ID
python -m model_collapse verify-detached COMPARISON_ID
python -m model_collapse resume COMPARISON_ID
python -m model_collapse fetch COMPARISON_ID --metadata-only
python -m model_collapse report COMPARISON_ID
python -m model_collapse validate COMPARISON_ID
python -m model_collapse cleanup COMPARISON_ID
```

The prepare command streams/tokenizes data on CPU and publishes a pinned private bank. Launch admission requires that complete bank, current-runtime/protocol acceptance and sufficient allowance. The allocation covers USD 10 including validation, failed attempts and storage, with USD 2 reserved and USD 0.50 maximum combined GPU validation spending. Historical resources and cost receipts are preserved.

## Validated execution

The [setup handoff](../../docs/fineweb-completion64-handoff.md) records 429 offline tests and all thirteen GPU/provider checks passed on RTX PRO 4000 Blackwell. Six actual smoke epochs, both private restores, direct loss calculations, interruption recovery and watchdog shutdown passed. Setup attempts together cost USD 0.29614 conservatively. Temporary resources were removed. The [full launch instruction](../../docs/validation/2026-11-06_completion64-full-authorization.json) was subsequently received. Use [current state](../../docs/current-state.md) for launch status.

The tested runtime is preserved in a [checksum-verified private source bundle](../../docs/validation/2026-11-06_completion64-source-bundle.json). Full-run forecasting gives 3.80 hours / USD 2.20 running cost, with historical/retained storage recorded separately. The [smoke report](../../reports/20261106T191336Z-smoke-completion64/report.md) describes validation scope.
