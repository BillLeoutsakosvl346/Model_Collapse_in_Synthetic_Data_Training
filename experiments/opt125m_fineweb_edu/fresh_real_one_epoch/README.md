# FineWeb-Edu real-only continued pilot

This condition starts from original pretrained OPT-125M and continues model weights, AdamW moments and one global cosine scheduler across ten cycles. Each cycle uses one epoch on 640,000 previously unused source-text tokens. The named smoke exercises three cycles of 16,384 source-text tokens, with real OPT-125M and the production workflow.

## Data

The pinned `HuggingFaceFW/fineweb-edu` revision uses the published `sample-10BT` collection and fourteen explicit Parquet shard paths. Seed 42 controls a shard permutation and a 10,000-document shuffle buffer. This is a reproducible shuffled-stream sample. The saved coverage records which files and rows were read, including buffer reads.

Reserve 1,000 validation and 1,000 test documents before allocating ten training collections. Deduplicate document IDs, normalized URLs and normalized text hashes. Check every sliding 64-source-token passage against earlier collections using 128-bit fingerprints. Document identities and matching passages are disjoint between collections. This check addresses exact passages; shorter matches and semantic near-duplicates remain outside its scope. Freshness describes the experimental fine-tuning sequence.

Retain original source text. Tokenize without automatic special tokens, divide each selected document into payloads of at most 510 source tokens, prefix BOS, and append EOS only at a genuine document ending. Pad to 512 positions and mask padding from attention and targets. The final training document may be trimmed to meet the exact source-text quota, with its whole identity reserved against later reuse. Evaluation processes all usable text in its fixed documents. Record source, special, padding and scored-target counts separately.

Real-only denotes the provided web-text source. Human authorship and overlap with OPT pretraining remain unaudited.

## Optimization and evaluation

- Full parameter fine-tuning with BF16 autocast and FP32 parameters, seed 42.
- AdamW, LR 5e-6, betas (0.9, 0.999), epsilon 1e-8, weight decay 0.01.
- Microbatch 4 and effective batch 16 sequences. Accumulation weights each microbatch by its actual prediction-target count.
- One global cosine schedule reaches 1e-6 after the last optimizer update. The number of steps is resolved from the frozen prepared collections.
- Evaluate untouched pretrained weights, then terminal weights after each one-epoch cycle. Persist optimizer, scheduler, global-step and RNG state with each checkpoint.
- Record optimization loss and evaluation-mode training NLL separately, plus fixed validation/test corpus NLL and perplexity.
- The exploratory full-pilot stability criterion allows at most 5% validation PPL growth relative to the initial model at every cycle. One seed supplies a descriptive trajectory.

## Commands and readiness

The initial [GPU validation report](../../../reports/20261106T143315Z-smoke-fresh-real/report.md) records passed setup and 402 offline tests. A [Blackwell capacity fallback](../../../docs/validation/2026-11-06_fineweb-blackwell-acceptance.json) subsequently passed private bulk restoration, actual watchdog shutdown and resource cleanup. The user authorized the [full launch](../../../docs/validation/2026-11-06_fineweb-full-launch.json) of `20261106T151257Z-full-fresh-real`, which [completed ten cycles with provider shutdown verified](../../../docs/validation/2026-11-06_fineweb-full-completion-status.json). See the [handoff](../../../docs/fineweb-real-only-handoff.md) for status and recovery commands.

```powershell
$experiment = "experiments/opt125m_fineweb_edu/fresh_real_one_epoch"
python -m model_collapse smoke --config "$experiment/smoke.yaml"
python -m model_collapse launch --config "$experiment/full.yaml" --generations 10
python -m model_collapse status RUN_ID
python -m model_collapse verify-detached RUN_ID
python -m model_collapse resume RUN_ID
python -m model_collapse fetch RUN_ID --metadata-only
python -m model_collapse report RUN_ID
python -m model_collapse cleanup RUN_ID
```

The full command requires matching runtime/protocol GPU acceptance and a fresh affordability check. Setup and later full-launch authorizations are recorded separately. A failed or interrupted cycle replays from its verified starting model and optimization state. Completed cycles retain their verified artifacts; backup-only interruptions retry the backup before advancing.

Use cloud execution for guarded training. The local `prepare` and `evaluate` stages support this profile. Metadata-only fetch keeps model tensors, optimization state, token arrays and original document text in private cloud storage. Full recovery includes those objects. Cleanup checks ownership and verified recovery.

See the [setup authorization](../../../docs/validation/2026-11-06_fineweb-setup-authorization.json) for the combined USD 0.50 validation cap, cumulative USD 50 ceiling and USD 2 reserve.
