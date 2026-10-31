# Final-epoch-5 paired workflow handoff

**Full pair `20261030T180755Z-full-paired-final5` completed on 31 October at 06:05:57 UTC/London**, 11.97 hours after provisioning. Both arms reached cycle 10; 19 unique stages / 95 epochs were checked, retaining epoch 5 throughout. Exact shared S0/S1/D2 and recorded initializer/generator/data handoffs passed. Runpod confirms automatic shutdown at 06:06:47, reason `complete`; no compute remains active. [Completion receipt](validation/2026-10-31_final-epoch5-full-completion.json). Do not launch or resume completed compute.

Pinned backup indexes contain 11 checkpoints per arm; 121 small records / 1,506,757 bytes were checksum-verified. Bulk weights and token arrays remain in private backups and retained volume `axxpvi80iq`; they were not downloaded or independently re-evaluated during this completion check. Observed account debit since launch was **$7.0722**, including concurrent retained storage and billing timing, not an itemized compute invoice. Current account spend is storage only, **$0.012/hour**. Detailed analysis remains separate from this completion verification.

- Acceptance: [`20261030T172428Z-smoke-paired-final5`](../reports/20261030T172428Z-smoke-paired-final5/report.md).
- Runtime SHA-256: `962243987067fd6e4aaaa7ac77acee05ae0775779d3904d7cdc6f6712e6583ca`; implementation commit `50f0ffe`.
- 131 offline tests and fresh independent review passed. A real matching RTX PRO 4000 Blackwell ran four training stages / twenty epochs, retaining epoch 5 throughout.
- Verified exact S0/S1/D2 sharing, fixed-S0 versus carried initializers, five actual process exits/restarts across the arms/coordinator, generation-shard recovery, both private restores/re-evaluations, SSH-disconnected progress and independent budget shutdown.
- Validation Pod `1f5peckcr6lgwg` and temporary volume `8kcglgf2t4` were removed after verified recovery. Full Pod `0quyi2z2jv8vd6` is now exited and volume `axxpvi80iq` remains for recovery; the three historical Pods remain exited and their recovery volumes remain preserved.
- Observed account debit through cleanup: **$0.154364**, including concurrent retained storage and billing timing, not an itemized invoice. About 26.5 MB of metadata was recovered locally; bulk model/token restoration occurred on the GPU.

## Authorised full launch

The following command was executed after the user's separate launch instruction and passing live admission checks. It is recorded for reproducibility; the pair has completed, so do not execute it again:

```powershell
python -m model_collapse launch --config comparisons/shared_start_final_epoch5/full.yaml --generations 10
```

This imports the original full S0/D1, creates a **new full-data** epoch-5 S1/D2, completes continued training through cycle 10, then automatically runs reset cycles 2–10 from the sealed prefix on the same GPU. It does not reuse the smoke S1/D2. There are 19 unique new training stages / 95 executed epochs.

Wait for the launcher to return `detached_supervisor_and_watchdog_verified: true`. Then closing VS Code, the browser, Codex or the laptop does not stop remote execution. One watchdog owns the shared allowance and stops the GPU after both arms finish, a failure or budget exhaustion.

## Reconciled budget and runtime limits

The following were the launch estimates and limits. The pair subsequently completed in 11.97 hours, before the budget deadline.

The real warmed benchmark measured 33,089.6 target tokens/s in training and 1,296.0 new tokens/s in five-beam generation at production batch sizes. The conservative forecast, including compute margins, checkpoint/upload time and handoff, is:

| Arm | Hours | Compute and running-storage estimate |
|---|---:|---:|
| Continued | 9.44 | $5.43 |
| Reset | 8.36 | $4.81 |
| Pair including handoff | **18.05** | **$10.39** |

These are forecasts, not measured full-run durations or invoices. The runs previously already executed five epochs; this change retains the fifth checkpoint rather than multiplying historical training work by five.

The historical reset run's ledger had charged compute until the morning status check. Runpod's `lastStatusChange` independently records shutdown at **28 October 02:58:40 UTC**; itemized hourly billing supports that record. Only that compute interval was corrected. Original ledger bytes, billing records and the correction were preserved in a pinned private backup; historical receipts, all other launches, storage intervals and the accepted runtime were preserved.

At admission, the corrected conservative task total was **$12.5474**, leaving **$10.4526** for this pair after the **$2 reserve**, within the unchanged **$25 cumulative ceiling**. Account credit was $18.8745; automatic payments and backup payments remained disabled. No credits were purchased.

The independent watchdog uses a padded $0.661959/hour rate, including provisioning time. Its maximum session is **15.79 hours**, ending **31 October 09:55 UTC/London**. The 18.05-hour forecast is more conservative than this budget deadline: a slow pair can save recoverable state and stop before both arms finish. Previous comparable arms totalled about 12.4 hours, which is context rather than a measured duration for this pair. Completion is not guaranteed. Actual spend and results will be checked after shutdown.

## Operations

```powershell
python -m model_collapse status PAIR_ID
python -m model_collapse resume PAIR_ID
python -m model_collapse fetch PAIR_ID --metadata-only
python -m model_collapse report PAIR_ID
python -m model_collapse cleanup PAIR_ID
```

Use **`20261030T180755Z-full-paired-final5`** as the pair ID for cloud operations. Resume is for unfinished runs and can provision paid compute; completed pairs are rejected. Do not resume while this Pod is still running. Metadata-only fetch keeps large checkpoints and token arrays in private cloud storage. Cleanup requires verified recovery of both arms. Retained historical storage still costs $0.009/hour, plus the new pair's volume; historical volumes were not deleted.

See the [machine-readable receipt](validation/2026-10-30_final-epoch5-setup.json), [condition](../comparisons/shared_start_final_epoch5/README.md) and [accepted plan](plans/matched-five-epoch.md).
