# Current recorded state

Last execution evidence: **31 October 2026, 16:39 UTC/London**, [50/50 launch receipt](validation/2026-10-31_mixed-50-full-launch.json). This is a saved observation, not a live status feed. Check provider state and completion/backup manifests before claiming subsequent progress, success or shutdown.

## Active recorded experiment

- Run: `20261031T163428Z-full-continued`; [50/50 continued-training protocol](../experiments/opt125m_wikitext2/continued_mixed_50_final5/README.md).
- Ten cycles, five epochs each, retain epoch 5; 18,973 real + 18,973 synthetic 64-token blocks per training mixture. Historical S0/D1 are imported; subsequent weights continue from the preceding selected model.
- Pod `aa1rxwnq8uwyss`, 30 GB volume `xo027a9d51`, RTX PRO 4000 Blackwell in EU-RO-1. First-epoch finite losses and epoch progress across closed SSH sessions were verified; this does not establish full completion.
- Detached worker, supervisor, private backup route and independent watchdog were checked. Forecast: 9.37 hours / USD 5.39 conservatively; watchdog deadline 1 November 03:27 UTC/London. Forecasts are not measurements of a completed run.
- Do not provision a duplicate or resume an active/completed run.

```powershell
python -m model_collapse status 20261031T163428Z-full-continued
```

## Authorization and acceptance

The user explicitly authorized the full mixed run and lifted the earlier cost constraint. The applied conservative limit is **USD 30 cumulative**, with a **USD 2 storage/shutdown reserve**, existing account credit only, no credit purchases and no automatic/backup payments. See the [authorization receipt](validation/2026-10-31_mixed-50-budget-authorization.json). New paid experiments require separate authorization.

The accepted mixed-data runtime is `266afa984a2a51b63f850390232a12b3ec98a93975ad27f9401983d1afc060c5`; [acceptance evidence](validation/2026-10-31_mixed-50-setup.json) covers the 50% mixture/five-epoch/final-checkpoint protocol. The [launch record](validation/2026-10-31_mixed-50-full-launch.json) documents the explicit local USD 30 operational override; the accepted runtime's historical default remains USD 25. Do not treat the default as a new authorization or alter accepted source just to change a budget parameter.

## Completed work and recovery

Week 1 pretrained reset, Week 2 continued weights, the matched shared-start reset branch and both Week 3 final-epoch-5 arms are completed and analysed. Use the [reports index](../reports/README.md) for their exact identities, conclusions and verification scope. All are one-seed trajectories; their conditions are not interchangeable.

Historical recovery volumes remain: Week 1 `wortjkxvr7`, continued `z6ziujxxus`, reset `08ypgkuw40`, final-epoch pair `axxpvi80iq`. The new mixed run has its own volume above. Do not delete recovery storage as repository housekeeping. Published metadata/probe audits do not establish full bulk checkpoint/dataset recovery. Storage continues to accrue until provider-verified deletion.

## Where to update next

On a later status check, record the observation time and evidence scope here and in the [catalogue](../experiments/catalog.yaml), with a dated receipt. Add results only after verifying expected stages, checkpoints/data lineage, backups and actual provider shutdown. Preserve earlier receipts, forecasts and authored reports. The original long agent chronology is archived in [history](history/2026-10-31_agent-context.md).
