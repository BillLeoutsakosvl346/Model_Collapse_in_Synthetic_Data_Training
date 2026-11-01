# Current recorded state

Last execution evidence: **1 November 2026**, [50/50 completion receipt](validation/2026-11-01_mixed-50-full-completion.json). All ten cycles completed and actual GPU shutdown was verified. This is a saved observation, not a live status feed; retained storage remains chargeable.

## Latest completed experiment

- Run: `20261031T163428Z-full-continued`; [50/50 continued-training protocol](../experiments/opt125m_wikitext2/continued_mixed_50_final5/README.md).
- Ten cycles, five epochs each, retain epoch 5; 18,973 real + 18,973 synthetic 64-token blocks per training mixture. Historical S0/D1 are imported; subsequent weights continue from the preceding selected model.
- Finished at **31 October 22:51:45 UTC/London**, after **6.29 hours**. All fifty epochs were finite and each new stage retained terminal epoch 5. Final real-test PPL is **62.71**, versus S0 **33.89**; full interpretation/plots remain to do.
- Pod `aa1rxwnq8uwyss` is **EXITED**; provider records shutdown at **22:55:32**, with watchdog reason `complete`. Preserve 30 GB volume `xo027a9d51` for recovery. All listed historical Pods were also exited at this check.
- Pinned private backups index 2,940 files / 5,857,080,018 bytes and eleven checkpoints. Checked 44 selected records / 246,706 bytes, all stage handoffs, mixture counts and epoch histories, plus pinned checkpoint-object LFS hashes/sizes. This is not a full metadata fetch, bulk restore or independent model re-evaluation.
- Observed account debit since launch: **USD 3.7539**, including concurrent retained storage/billing timing, not an itemized invoice. Account spend is **USD 0.015/hour** for retained storage. The compute ledger ends at provider-verified shutdown; other launch intervals are unchanged.
- Do not provision a duplicate or resume completed compute.

```powershell
python -m model_collapse status 20261031T163428Z-full-continued
```

## Authorization and acceptance

The user explicitly authorized the full mixed run and lifted the earlier cost constraint. The applied conservative limit is **USD 30 cumulative**, with a **USD 2 storage/shutdown reserve**, existing account credit only, no credit purchases and no automatic/backup payments. See the [authorization receipt](validation/2026-10-31_mixed-50-budget-authorization.json). New paid experiments require separate authorization.

The accepted mixed-data runtime is `266afa984a2a51b63f850390232a12b3ec98a93975ad27f9401983d1afc060c5`; [acceptance evidence](validation/2026-10-31_mixed-50-setup.json) covers the 50% mixture/five-epoch/final-checkpoint protocol. The [launch record](validation/2026-10-31_mixed-50-full-launch.json) documents the explicit local USD 30 operational override; the accepted runtime's historical default remains USD 25. Do not treat the default as a new authorization or alter accepted source just to change a budget parameter.

## Completed work and recovery

Week 1 pretrained reset, Week 2 continued weights, the matched shared-start reset branch and both Week 3 final-epoch-5 arms are completed and analysed. Use the [reports index](../reports/README.md) for their exact identities, conclusions and verification scope. All are one-seed trajectories; their conditions are not interchangeable.

Recovery volumes remain: Week 1 `wortjkxvr7`, continued `z6ziujxxus`, reset `08ypgkuw40`, final-epoch pair `axxpvi80iq` and mixed `xo027a9d51`. Do not delete recovery storage as repository housekeeping. Published metadata/probe audits do not establish full bulk checkpoint/dataset recovery. Storage continues to accrue until provider-verified deletion.

## Where to update next

Next: fetch the remaining metadata/small probes, audit diversity and matched-condition statistics, then publish the mixed-data analysis without retraining. Bulk recovery remains a separate prerequisite to deleting recovery storage. Preserve earlier receipts, forecasts and authored reports. The original long agent chronology is archived in [history](history/2026-10-31_agent-context.md).
