# Real-only continued-training control, final epoch 5

Status: actual two-cycle GPU acceptance passed on RTX PRO 4500 Blackwell Server Edition. See the [acceptance receipt](../../../docs/validation/2026-11-02_real-only-gpu-acceptance.json) and [current state](../../../docs/current-state.md) before any launch; avoid duplicate compute.

This control tests whether repeated continued fine-tuning on the finite real WikiText-2 training pool can itself worsen held-out prediction. It does not assume that repetition explains the existing 20%/50% reversal. See the [methodology](../../../docs/methodology.md) and [audit](../../../docs/audits/2026-11-01_20-vs50-perplexity.md).

## Matched protocol

- Import the exact historical real-trained S0 used by the mixed conditions. Do not import an already mixed/synthetic-trained S1.
- Run ten new training cycles, five epochs each, always retaining epoch 5. Carry that checkpoint's weights into the next cycle. Restart the optimizer and scheduler every cycle as before.
- Preserve the model/dataset revisions, packing, seed 42, batch sizes, precision, learning-rate schedule, decoding and frozen evaluation arrays.
- Each full training dataset has 37,946 real blocks / 2,428,544 input tokens, with zero synthetic training blocks. Ten cycles add fifty passes over the real training pool after S0.

## Selection and split separation

Use the existing mixer with `synthetic_fraction: 0.0`, rather than a new sampler. Its separate real/synthetic/shuffle RNG streams use `SeedSequence([42, cycle, stream])`, with streams 0/1/2. Sampling remains without replacement. At 100% real, every real block appears exactly once in each cycle's stored dataset; the cycle-dependent selection order and final permutation change the order, not the set of contents. Epoch shuffling remains the historical training-engine shuffle.

Real rows come only from `data/train.npy`. Validation and test rows are prepared independently and never supply training targets or generation prompts. A separate [packed-split audit](../../../docs/validation/2026-11-02_real-only-split-audit.json) reproduced the full frozen arrays and found zero exact 64-token train/validation/test overlaps. This does not audit shorter shared passages or pretrained-model contamination.

Persist source indices, source labels, shuffle, hashes and the atomic mixture manifest through the existing path. The stored mixture stays fixed for all five epochs and through recovery. A completed stage cannot substitute a changed mixture.

## Evaluation probes and execution profiles

The scientific training configuration changes only `experiment_id` and `synthetic_fraction` relative to the 50/50 definitions; the smoke's operational session cap is USD 0.25. The unchanged training engine consumes all real blocks each cycle.

For zero synthetic fraction, newly generated text is restricted to the same fixed 256-prompt measurement probe. Full synthetic pools would supply zero training tokens and are unnecessary. The historical initial D1 remains imported and verified unchanged; subsequent `synthetic_for_gen_*` directories hold evaluation probes rather than full training pools. The mixer still records zero selected synthetic rows and the complete real training selection. This changes neither real training token selection nor probe prompts/decoding. Epoch-local seeding preserves training RNG independence. Tiny-OPT tests verify identical probe outputs to the full-generation prefix, two real-only five-epoch updates from a smaller probe pool, unchanged held-out arrays and exact interruption/recovery.

`full.yaml`/`smoke.yaml` retain the historical RTX PRO 4000 Blackwell hardware requirement. An explicit `--gpu "GPU NAME"` override on new standalone runs changes only that resolved requirement when matching stock is unavailable. Each GPU needs its own actual acceptance. A fallback run must explicitly record its hardware difference and must not be presented as an exact hardware-matched comparison with the earlier mixed runs. Fresh warmed benchmarks determine whether the full run fits the USD 50 ceiling and reserve. Resume keeps recorded hardware and rejects an override.

The launcher honors the [authorized ledger ceiling](../../../docs/validation/2026-11-02_real-only-budget-authorization.json); provisioning is bounded by both its technical timeout and its budget deadline. These runtime changes require new acceptance. Earlier preparation/preflight estimates included unused full generation; the optimized forecast counts ten training stages and ten new 256-output probes, with historical D1 generation excluded.

## Checks and execution

Run offline checks without cloud resources:

```powershell
python -m pytest tests/test_real_only_control.py tests/test_mixing.py -q
python scripts/maintenance/check_repository.py
```

The named smoke configuration uses two cycles, 256 real training blocks, 64 validation blocks and 64 test blocks, executing ten training epochs. The pod-storage variant passed actual GPU acceptance, including terminal checkpoints, two worker interruptions, detached progress, private full restoration/re-evaluation and the actual watchdog. Historical 20%/50% or campaign certificates must not certify 0% synthetic, and unchanged runtime identity does not replace protocol acceptance.

The accepted fallback uses the pod-storage profiles and an explicit GPU name. These are paid commands; consult current state before executing them and never duplicate an active or completed run:

```powershell
python -m model_collapse smoke --config experiments/opt125m_wikitext2/continued_real_only_final5/smoke-pod-storage.yaml --gpu "NVIDIA RTX PRO 4500 Blackwell Server Edition"
python -m model_collapse launch --config experiments/opt125m_wikitext2/continued_real_only_final5/full-pod-storage.yaml --gpu "NVIDIA RTX PRO 4500 Blackwell Server Edition" --generations 10
```

`--gpu auto` is available only for a small standalone smoke; it selects an available modern physical NVIDIA GPU at no more than USD 1.19/hour and excludes MIG slices. Full launch must name the exact GPU that passed acceptance. Resume rejects hardware overrides.

`full-pod-storage.yaml`/`smoke-pod-storage.yaml` explicitly select a 30 GB persistent Pod disk at `/workspace`, replacing the independent network volume when compatible network-mounted placement is unavailable. The disk survives stop/start but is deleted with its Pod; do not terminate the production Pod before verified recovery. Private Hub backups remain mandatory. Conservative accounting reserves the stopped-disk rate (USD 0.20/GB/month) throughout. The network-backed original profiles retain their defaults.

The accepted warmed benchmark forecasts about 1.71 hours / USD 1.25 for ten full real-only cycles, including margins and running storage. This excludes older retained storage and future retention; it is an estimate rather than measured completion. Actual commands and selected hardware belong in the run receipt.

## Interpretation

A rising real-only held-out PPL would show that deterioration is possible without synthetic training targets under this continued-training schedule. It would strengthen, but not prove, the cumulative real-data overfitting explanation for the ratio reversal. Low real-only PPL would weaken repetition alone as an explanation and motivate examining the interaction with synthetic targets. The control deliberately changes the number of real exposures while holding total training workload fixed; it does not independently isolate every mechanism. One seed remains descriptive. Compare real-training and held-out NLL as well as the existing output metrics before assigning a cause.
