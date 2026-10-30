# Final-epoch-5 paired workflow handoff

**Implementation and GPU validation are complete. Full compute has not started.**

- Acceptance: [`20261030T172428Z-smoke-paired-final5`](../reports/20261030T172428Z-smoke-paired-final5/report.md).
- Runtime SHA-256: `962243987067fd6e4aaaa7ac77acee05ae0775779d3904d7cdc6f6712e6583ca`; implementation commit `50f0ffe`.
- 131 offline tests and fresh independent review passed. A real matching RTX PRO 4000 Blackwell ran four training stages / twenty epochs, retaining epoch 5 throughout.
- Verified exact S0/S1/D2 sharing, fixed-S0 versus carried initializers, five actual process exits/restarts across the arms/coordinator, generation-shard recovery, both private restores/re-evaluations, SSH-disconnected progress and independent budget shutdown.
- Pod `1f5peckcr6lgwg` and temporary volume `8kcglgf2t4` were removed after verified recovery. No GPU remains running. Historical resources remain preserved.
- Observed account debit through cleanup: **$0.154364**, including concurrent retained storage and billing timing, not an itemized invoice. About 26.5 MB of metadata was recovered locally; bulk model/token restoration occurred on the GPU.

## Later full launch

After the user's separate launch instruction and passing live admission checks:

```powershell
python -m model_collapse launch --config comparisons/shared_start_final_epoch5/full.yaml --generations 10
```

This imports the original full S0/D1, creates a **new full-data** epoch-5 S1/D2, completes continued training through cycle 10, then automatically runs reset cycles 2–10 from the sealed prefix on the same GPU. It does not reuse the smoke S1/D2. There are 19 unique new training stages / 95 executed epochs.

Wait for the launcher to return `detached_supervisor_and_watchdog_verified: true`. Then closing VS Code, the browser, Codex or the laptop does not stop remote execution. One watchdog owns the shared allowance and stops the GPU after both arms finish, a failure or budget exhaustion.

## Current affordability limitation

The real warmed benchmark measured 33,089.6 target tokens/s in training and 1,296.0 new tokens/s in five-beam generation at production batch sizes. The conservative forecast, including compute margins, checkpoint/upload time and handoff, is:

| Arm | Hours | Compute and running-storage estimate |
|---|---:|---:|
| Continued | 9.44 | $5.43 |
| Reset | 8.36 | $4.81 |
| Pair including handoff | **18.05** | **$10.39** |

These are forecasts, not measured full-run durations or invoices. The runs previously already executed five epochs; this change retains the fifth checkpoint rather than multiplying historical training work by five.

**The current launch guard would refuse the pair.** Its remaining conservative task allowance is $5.64 after the $2 reserve, versus the $10.39 estimate. Account credit was $18.88 at the final read; it does not override the cumulative $25 ceiling. Reconcile the existing conservative historical receipts against verified provider billing before a later launch. No budget limit, credit purchase or payment setting was changed.

## Operations

```powershell
python -m model_collapse status PAIR_ID
python -m model_collapse resume PAIR_ID
python -m model_collapse fetch PAIR_ID --metadata-only
python -m model_collapse report PAIR_ID
python -m model_collapse cleanup PAIR_ID
```

Use the **pair ID** for cloud operations. Resume is for unfinished runs and can provision paid compute; completed pairs are rejected. Metadata-only fetch keeps large checkpoints and token arrays in private cloud storage. Cleanup requires verified recovery of both arms. Retained historical storage still costs $0.009/hour; it was not deleted by this setup.

See the [machine-readable receipt](validation/2026-10-30_final-epoch5-setup.json), [condition](../comparisons/shared_start_final_epoch5/README.md) and [accepted plan](plans/matched-five-epoch.md).
