# SmolLM2-135M: matched synthetic fractions

**Execution status:** GPU acceptance passed; full run `20261111T210854Z-full-completion64-mixtures` is running with verified detached training on the accepted RTX PRO 4000 Blackwell. Do not launch a duplicate. See the [handoff](../../docs/smollm2-fractions-handoff.md).

Seven independently initialized conditions use synthetic fractions **0%, 50%, 80%, 95%, 99%, 99.9% and 100%**. This is a model-native replication of the OPT completion protocol, using base `HuggingFaceTB/SmolLM2-135M` at commit `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.

## Scientific protocol

- Each condition starts from the same untouched pretrained checkpoint. Carry its own model, AdamW, scheduler and training RNG across ten cycles; do not reset between cycles or share optimization state between conditions.
- Each cycle uses the next frozen 10,000-pair chunk. Inputs have 129 positions: BOS, 64 real prompt tokens, then 64 target tokens. Prompts remain visible; only the 64 continuation tokens contribute to loss.
- For mixed conditions, deterministic nested selections replace targets at their original prompt rows. The condition's preceding checkpoint generates exactly 64 new tokens using five deterministic beams, repetition penalty 1.0. All targets are frozen before training that cycle. Real targets retain the original source continuation.
- One terminal epoch per chunk; microbatch 32 / effective batch 128, BF16 autocast with FP32 parameters, seed 42, AdamW and the historical global cosine LR 5e-6 to 1e-6. Each condition performs 790 optimizer updates and scores 6.4 million training targets.
- Primary evaluation scores fixed real continuations with token-weighted conditional NLL and corpus perplexity. Output diversity/repetition uses a fixed document-stratified validation probe. Secondary full-text evaluation is recorded separately.

| Synthetic fraction | Generated targets/cycle | Original targets/cycle |
| --- | ---: | ---: |
| 0% | 0 | 10,000 |
| 50% | 5,000 | 5,000 |
| 80% | 8,000 | 2,000 |
| 95% | 9,500 | 500 |
| 99% | 9,900 | 100 |
| 99.9% | 9,990 | 10 |
| 100% | 10,000 | 0 |

## Native data bank

`science/completion_bank_v2.py` retokenizes the pinned FineWeb source with SmolLM2's tokenizer. Historical OPT bank bytes and builders stay unchanged. Original held-out document assignments and training-cycle allocations are reused; deterministic extensions, when needed, cannot borrow documents assigned to other cycles. Indexed original documents are checksum-verified before reading.

Non-overlapping 128-token windows stay within documents. Native document identities and exact sliding 64-token passages are checked for separation. The bank records actual held-out counts, token arrays, source/window identities, source coverage, examples and hashes. Native BOS/EOS are both token 0. Newline diagnostics identify native tokens containing LF rather than using OPT's token ID.

Secondary evaluations retokenize the exact historically selected source spans. The historical OPT tokenizer is pinned and downloadable; preparation does not require a private report-tokenizer cache. See the validation handoff for the final bank identity and measured counts.

## Commands and acceptance scope

```powershell
$comparison = "comparisons/smollm2_135m_fineweb_completion64_fractions"
python -m model_collapse prepare --config "$comparison/full.yaml" --root data/smollm2_completion64_v2
python -m model_collapse smoke --config "$comparison/smoke.yaml"

# User-authorized after actual acceptance and affordable admission:
python -m model_collapse launch --config "$comparison/full.yaml" --generations 10
python -m model_collapse status COMPARISON_ID
python -m model_collapse resume COMPARISON_ID
python -m model_collapse fetch COMPARISON_ID --metadata-only
python -m model_collapse report COMPARISON_ID
python -m model_collapse cleanup COMPARISON_ID
```

The short smoke executes **only mixed50**, two successive fresh chunks of 128 pairs, one optimizer update each; 32 validation/test pairs and 32 probe prompts. Production batches and decoding stay unchanged. One terminal model/optimizer backup is cold-restored and re-evaluated. Seven fractions are checked offline, not all trained on GPU. This scoped certificate does not claim the legacy interruption matrix or deliberate budget exhaustion.

The original 900-second combined validation limit was extended to 1,200 seconds for recovery-only finalization, with no additional optimizer updates; the USD 0.50 cap stayed unchanged. Both accepted GPU tests passed ordinary independent shutdown. See the dated acceptance/startup receipt. The independent watchdog handles completion, failure and budget shutdown.

## Full-launch admission and interpretation

The planned sweep runs seven arms sequentially on the accepted GPU. It requires 200 GB working storage and a separately verified 200 GB private S3 backup volume. Both full-size volumes are now allocated for the authorized launch. The launcher rejects a missing/undersized backup descriptor (`.state/smollm2_backup_store.json`) before provisioning compute, and checks actual free capacity, budget, credit and the measured forecast. The later full-task authorization is USD 30, with USD 2 reserved, under the USD 110 cumulative ceiling.

Compare fractions directly within this native tokenizer. Cross-model figures should emphasize baseline-relative trajectories and explicitly label tokenizer, pretraining and hardware differences; absolute OPT/SmolLM2 perplexities are not directly interchangeable. SmolLM2 pretraining includes FineWeb-Edu, so experimental held-out separation does not establish pretraining novelty. These are one-seed descriptive trajectories with decreasing learning rate.

See [current state](../../docs/current-state.md), [authorization](../../docs/validation/2026-11-11_smollm2-authorization.json), [metric definitions](../../docs/metrics.md) and [operations](../../docs/operations.md).
