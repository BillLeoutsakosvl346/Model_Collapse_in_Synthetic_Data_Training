# Real-only continued-training control, final epoch 5

Status: offline checks passed; no GPU acceptance or full run is recorded for this condition. See the [preparation receipt](../../../docs/validation/2026-11-02_real-only-control-preparation.json).

This control tests whether repeated continued fine-tuning on the finite real WikiText-2 training pool can itself worsen held-out prediction. It does not assume that repetition explains the existing 20%/50% reversal. See the [methodology](../../../docs/methodology.md) and [audit](../../../docs/audits/2026-11-01_20-vs50-perplexity.md).

## Matched protocol

- Import the exact historical real-trained S0 used by the mixed conditions. Do not import an already mixed/synthetic-trained S1.
- Run ten new training cycles, five epochs each, always retaining epoch 5. Carry that checkpoint's weights into the next cycle. Restart the optimizer and scheduler every cycle as before.
- Preserve the model/dataset revisions, packing, seed 42, batch sizes, precision, learning-rate schedule, decoding and frozen evaluation arrays.
- Each full training dataset has 37,946 real blocks / 2,428,544 input tokens, with zero synthetic training blocks. Ten cycles add fifty passes over the real training pool after S0.

## Selection and split separation

Use the existing mixer with `synthetic_fraction: 0.0`, rather than a new sampler. Its separate real/synthetic/shuffle RNG streams use `SeedSequence([42, cycle, stream])`, with streams 0/1/2. Sampling remains without replacement. At 100% real, every real block appears exactly once in each cycle's stored dataset; the cycle-dependent selection order and final permutation change the order, not the set of contents. Epoch shuffling remains the historical training-engine shuffle.

Real rows come only from `data/train.npy`. Validation and test rows are prepared independently and never supply training targets or generation prompts. This establishes split separation, not an exhaustive audit of naturally duplicated text across the original corpus splits.

Persist source indices, source labels, shuffle, hashes and the atomic mixture manifest through the existing path. The stored mixture stays fixed for all five epochs and through recovery. A completed stage cannot substitute a changed mixture.

## Existing execution path and overhead

The configuration changes only `experiment_id` and `synthetic_fraction` relative to the 50/50 definitions. No training/runtime code changes are needed. For this first control, retain the existing generation, import and verification path: initial D1 is pinned as before and later full synthetic pools are still generated, backed up and verified, but **zero rows from any synthetic pool enter training**. This preserves the exact tested code path and output metrics at the cost of unnecessary full-pool generation. Do not claim that the run skips generation or forecast it as training-only. A future probe-only optimization would be a separate runtime change requiring validation.

## Checks and execution

Run offline checks without cloud resources:

```powershell
python -m pytest tests/test_real_only_control.py tests/test_mixing.py -q
python scripts/maintenance/check_repository.py
```

The named smoke configuration uses two cycles, 256 real training blocks, 64 validation blocks and 64 test blocks, executing ten training epochs. It has not yet been GPU-accepted. Historical 20%/50% or campaign certificates must not certify 0% synthetic, and unchanged runtime identity does not replace protocol acceptance.

These are future paid commands, subject to appropriate authorization, current acceptance and the existing budget guard; they were not executed while preparing these definitions:

```powershell
python -m model_collapse smoke --config experiments/opt125m_wikitext2/continued_real_only_final5/smoke.yaml
python -m model_collapse launch --config experiments/opt125m_wikitext2/continued_real_only_final5/full.yaml --generations 10
```

## Interpretation

A rising real-only held-out PPL would show that deterioration is possible without synthetic training targets under this continued-training schedule. It would strengthen, but not prove, the cumulative real-data overfitting explanation for the ratio reversal. Low real-only PPL would weaken repetition alone as an explanation and motivate examining the interaction with synthetic targets. The control deliberately changes the number of real exposures while holding total training workload fixed; it does not independently isolate every mechanism. One seed remains descriptive. Compare real-training and held-out NLL as well as the existing output metrics before assigning a cause.
