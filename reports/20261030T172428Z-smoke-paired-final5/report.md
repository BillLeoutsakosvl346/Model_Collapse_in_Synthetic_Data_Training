# Paired final-epoch-5 validation

**Pipeline validation only. The full-data comparison has not been launched.**

- Run: `20261030T172428Z-smoke-paired-final5`; matching RTX PRO 4000 Blackwell, PyTorch 2.8.0+cu128, Transformers 4.37.2, BF16.
- Runtime: `962243987067fd6e4aaaa7ac77acee05ae0775779d3904d7cdc6f6712e6583ca`; implementation commit `50f0ffec92702fdb9d98b1d5d1c0efa4f86e124d`.
- 131 offline tests pass. Fresh independent review found and fixed backup-recovery, cleanup-provenance and legacy-admission gaps before GPU acceptance.
- The same production pipeline ran 256 training / 64 validation / 64 test blocks, with continued stages 1/2 and reset stages 2/3: **four unique stages, twenty epochs**.

## Scientific behavior verified

- Reused the exact real-trained S0 and pinned historical D1; prompt and continuation arrays contain 64 tokens each. Training targets contain continuations only.
- Trained one new S1 for five epochs, retained epoch 5 and generated one exact D2. Both arms share byte-identical S0/S1 metrics and the same D2 hash.
- Continued cycle 2 starts from S1. Reset cycles 2/3 start from fixed S0, while their datasets come from the immediately preceding branch model.
- Every new stage changed weights; selected checkpoint tensors match terminal epoch-5 weights. Validation remains diagnostic and test never selects.

| Arm / cycle | Executed / selected epoch | Lowest-validation epoch | Validation PPL | Test PPL |
|---|---:|---:|---:|---:|
| continued / 1 | 5 / 5 | 1 | 37.3552 | 44.0323 |
| continued / 2 | 5 / 5 | 2 | 36.4110 | 45.1084 |
| reset / 2 | 5 / 5 | 1 | 38.0553 | 46.2149 |
| reset / 3 | 5 / 5 | 1 | 38.1844 | 46.9877 |

These small split scores are not comparable to the historical full-test scores. The smoke endpoints deliberately differ (2/3) to validate two new reset stages; **production endpoints are 10/10**.

![Smoke validation only: shared first two points](analysis/comparison.png)

The seven plotted rows include two duplicated shared measurements. They do not represent seven separate training stages. Diversity uses the unchanged fixed 256-prompt probe and existing definitions; no collapse conclusion is drawn from this smoke run.

## Recovery and shutdown

- Both arms passed two actual worker exits: interrupted generation-shard reuse and resume after a completed training stage.
- The parent exited/restarted after sealing the source, then continued into reset without changing completed continued artifacts.
- Closed SSH connections between observations; stage/event progress and both independent heartbeats advanced.
- Restored both private backups, checked every indexed file, reloaded the final checkpoint of each arm and repeated real-test evaluation plus direct shifted-token NLL checks.
- Local recovery verified 226 metadata files / 26,532,283 bytes. Bulk checkpoint/token recovery was verified on the GPU, not downloaded to the laptop.
- The independent budget guard stopped Pod `1f5peckcr6lgwg`; provider state was EXITED. Recovery-gated cleanup removed that Pod and temporary volume `8kcglgf2t4`. No GPU is running; the three historical recovery volumes remain.
- Observed account debit through cleanup: **$0.154364**, within the combined $0.75 acceptance cap. This includes concurrent storage/billing timing and is not an itemized compute bill.

## Measured forecast and current budget gate

- Warmed training: **33,089.6 shifted target tokens/s**; five-beam generation: **1,296.0 new tokens/s**, using production batch sizes.
- Conservative continued estimate: **9.44 h / $5.43**.
- Conservative reset estimate: **8.36 h / $4.81**.
- Combined, including handoff allowance: **18.05 h / $10.39**. This includes a 1.5× compute margin and checkpoint/upload allowances, not a measured full run; storage/shutdown reserves remain separate.
- **The current budget guard would refuse the full pair:** its remaining conservative allowance is $5.64, below the $10.39 forecast. Account credit ($18.88) is a separate quantity. Existing conservative historical receipts require reconciliation before a launch can be described as affordable within the unchanged $25 ceiling; no limit or payment setting was changed.

## Handoff

The full configuration matches the actual GPU acceptance certificate. After a later explicit launch instruction **and** a passing live budget/stock check:

```powershell
python -m model_collapse launch --config comparisons/shared_start_final_epoch5/full.yaml --generations 10
```

The detached coordinator runs continued then reset sequentially on one GPU. Wait for `detached_supervisor_and_watchdog_verified: true`; afterward laptop/browser/Codex closure does not control execution. Use the returned pair ID with `status`, `resume`, `fetch --metadata-only`, `report` and `cleanup`. Metadata-only fetch avoids bulk local downloads.

See [complete metrics and graph families](analysis/report.md), [stage validation](stage_validation.csv), [machine-readable validation receipt](../../docs/validation/2026-10-30_final-epoch5-setup.json) and [condition commands](../../comparisons/shared_start_final_epoch5/README.md).
