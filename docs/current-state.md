# Current recorded state

**20/80 reset analysis completed, 2 November:** the [report](../reports/20261101T030306Z-full-mixed-campaign-reset-mixed20/report.md) and main Google Doc now include matched continued/reset methods, six square-panel figure families, three native tables, examples and a descriptive comparison across reset fractions. Final reset PPL is **35.89 versus continued 72.06** (50.19% lower), with lower NLL on 4,450/4,494 real-test blocks. Reset still has lower probe diversity/entropy and higher within-output repetition (54.57% versus 46.73%). Across reset fractions 100/50/20%, endpoint PPL is 44.50 / 37.15 / 35.89; the continued-only ratio reversal is absent, without establishing its exact cause.

Recovery verified **1,421 metadata files / 53,710,269 bytes** plus nine new fixed probes / 590,976 bytes. The main pair audit checks 22 metric rows, twenty unique probes, 100 saved epoch entries (95 unique), exact S0/S1/D2/M2 and all recorded handoffs. Analysis and figures reproduce byte for byte; earlier document pages and published records remain unchanged. See the [delivery receipt](validation/2026-11-02_reset20-analysis-and-doc.json). No Runpod/SSH calls, resource changes, bulk model downloads or model re-evaluation were part of this analysis; preserve recovery storage.

**Campaign completed — verified 2 November:** all three jobs of `20261101T030306Z-full-mixed-campaign` finished through cycle 10. Parent completion is **1 November 23:20:44 UTC/London**; Runpod independently verifies Pod `faocwszff0eaq6` **EXITED at 23:21:09**, with watchdog reason `complete`. All 28 new stages / 140 epochs retained terminal epoch 5; all 33 metric rows, finite epoch histories, recorded initializer/generator/completion hashes, sealed backup indexes and shared 20/80 M2 identities were checked. Final test PPL: **37.15 reset50, 72.06 continued20, 35.89 reset20**. See the [completion receipt](validation/2026-11-02_mixed-campaign-full-completion.json).

Do not duplicate or resume completed compute. The ledger compute interval is closed at provider-verified shutdown. Preserve the 60 GB volume `11hvbmunz5` and all historical recovery storage: completion and analysis recovered small evidence, not full weights/training arrays or independent model evaluations. Earlier running-state observations below are historical and superseded by this verified completion.

**Latest scientific update, 1 November:** the 20/80 continued child completed at **16:00:29 UTC/London**, verified from its sealed backup. Its [three-fraction report](../reports/20261101T030306Z-full-mixed-campaign-continued-mixed20/report.md) and main Google Doc compare final PPL **72.06 (20%) / 62.71 (50%) / 107.37 (100% synthetic)**. The 20% run has better endpoint phrase-diversity/repetition measures, so the ranking differs by metric. Recovery verified 1,560 metadata files / 52,452,006 bytes plus ten new probes / 656,640 bytes. The same document update uses a separate [square-panel presentation revision](../reports/presentation_revisions/2026-11-01_square-reset50/report.md) for the 50/50 reset section, preserving its original evidence. See the [delivery receipt](validation/2026-11-01_continued20-analysis-and-square-doc.json).

This update used Hugging Face reads and offline analysis only. It does **not** refresh live Runpod state, claim completion of reset20/the parent, or authorize cleanup. The earlier launch/status observations below remain dated evidence.

Last execution evidence: **1 November 2026**, [full mixed-campaign launch](validation/2026-11-01_mixed-campaign-full-launch.json). The campaign is running with verified detached startup. This is a saved observation, not a live status feed; check status before acting.

**Later scientific evidence, 1 November:** the 50/50 reset child completed at **08:46:17 UTC/London**, verified from its immutable sealed Hugging Face backup. Its [matched report](../reports/20261101T030306Z-full-mixed-campaign-reset-mixed50/report.md) and the main Google Doc now contain the full analysis. Final test PPL is **37.15 reset versus 62.71 continued**, but reset has lower distinct-2/3 and higher average within-output repetition. All nine new five-epoch stages and twenty unique paired probes were audited. Recovery verified 1,421 metadata files / 53,699,537 bytes plus nine new probe arrays / 590,976 bytes; bulk restore remains separate. See the [analysis receipt](validation/2026-11-01_mixed50-reset-analysis-and-doc.json).

This analysis used Hugging Face and offline records only: **no Runpod API, SSH, resource, ledger or worker changes**. It does not claim parent-campaign completion or refresh the live state of either 20/80 job. The launch-time observations below are preserved as dated evidence; the first job's scientific completion supersedes its earlier running state.

## Historical campaign launch observations (completed above)

- Parent `20261101T030306Z-full-mixed-campaign`; Pod `faocwszff0eaq6`, 60 GB Standard volume `11hvbmunz5`, EU-RO-1. Matching RTX PRO 4000 Blackwell costs USD 0.57/hour; pinned image/software/CUDA/BF16 checks and private backup probes passed.
- Started at **03:03 UTC/London**. First job is 50/50 reset, cycle 2; its first full training stage subsequently completed all five epochs, retained epoch 5 and moved to next-pool generation. The initial startup check had independently observed two finite new epochs. Exact full M2 has 18,973 real + 18,973 synthetic blocks and its original token/selection identity. The supervisor and independent watchdog advanced across closed SSH sessions. The laptop is no longer required.
- One coordinator runs reset50 cycles 2–10, continued20 cycles 1–10, then reset20 cycles 2–10: 28 new stages / 140 epochs, retaining epoch 5. Later jobs were queued at this check. Do not provision a duplicate or resume active compute.
- The admission forecast was 24.09h / USD 13.93. The rented GPU's live benchmark is slower: **26.40h / USD 15.27** conservatively, versus **25.64h** funded watchdog coverage, ending **2 November 04:41 UTC/London**. Historical runs suggest 17–21h may be feasible, but a slow campaign can pause before all jobs finish. The USD 45 cumulative ceiling and USD 2 reserve were not raised or relaxed.
- Outputs go to the new volume and private per-job Hugging Face backups. Historical volumes remain untouched. Startup verification is not completion or a final backup audit.

```powershell
python -m model_collapse status 20261101T030306Z-full-mixed-campaign
```

## Latest completed experiment

- Run: `20261031T163428Z-full-continued`; [50/50 continued-training protocol](../experiments/opt125m_wikitext2/continued_mixed_50_final5/README.md).
- Ten cycles, five epochs each, retain epoch 5; 18,973 real + 18,973 synthetic 64-token blocks per training mixture. Historical S0/D1 are imported; subsequent weights continue from the preceding selected model.
- Finished at **31 October 22:51:45 UTC/London**, after **6.29 hours**. All fifty epochs were finite and each new stage retained terminal epoch 5. The [audited comparison](../reports/20261031T163428Z-full-continued/report.md) and main Google Doc now include the methods, trajectories, endpoint statistics, training diagnostics and examples.
- Pod `aa1rxwnq8uwyss` is **EXITED**; provider records shutdown at **22:55:32**, with watchdog reason `complete`. Preserve 30 GB volume `xo027a9d51` for recovery. All listed historical Pods were also exited at this check.
- Pinned private backups index 2,940 files / 5,857,080,018 bytes and eleven checkpoints. Subsequent analysis recovered all 1,563 metadata files / 52,418,722 bytes and ten new small probes / 656,640 bytes; the shared S0 probe and historical comparator evidence were reused. Twenty-one unique probes, all 100 epoch histories, handoffs, selection hashes and NLL/PPL arithmetic were checked. Bulk checkpoint/full training array recovery and independent model re-evaluation remain separate.
- Observed account debit since launch: **USD 3.7539**, including concurrent retained storage/billing timing, not an itemized invoice. Account spend is **USD 0.015/hour** for retained storage. The compute ledger ends at provider-verified shutdown; other launch intervals are unchanged.
- Do not provision a duplicate or resume completed compute.

```powershell
python -m model_collapse status 20261031T163428Z-full-continued
```

## Authorization and acceptance

The user authorized setup and small GPU validation of a new sequential campaign: 50/50 reset, 20/80 continued, then 20/80 reset. The current limit is **USD 45 cumulative**, with a **USD 2 storage/shutdown reserve** and **USD 0.75 combined new acceptance cap** within the existing USD 3 setup allowance. Use existing credit only; no credit purchases or automatic/backup payment changes. See the [current authorization](validation/2026-11-01_mixed-campaign-budget-authorization.json). The user subsequently explicitly authorized the full campaign; its verified launch is recorded above. Earlier USD 25/30 limits remain historical receipts.

The latest accepted campaign runtime is `35c89c3b9fb78c7967c4f62f07972739eabbe57b353a0360749eba71d1af2171`, protocol `770a627c02821b1301f3194268e65fb9635eccdc6a43e639477d886f26dac377`. The campaign uses its authorized private-ledger ceiling. Historical [50/50 acceptance](validation/2026-10-31_mixed-50-setup.json) and its [USD 30 launch override](validation/2026-10-31_mixed-50-full-launch.json) retain their original runtime and scope; they do not certify the new campaign.

## Prepared three-job campaign

- Acceptance: `20261101T022210Z-smoke-mixed-campaign`; [report](../reports/20261101T022210Z-smoke-mixed-campaign/report.md), [preview](../reports/20261101T022210Z-smoke-mixed-campaign/dataset_preview.json) and [handoff](mixed-campaign-handoff.md).
- 345 offline tests, independent review and all ten parent GPU/provider checks passed. Six actual training stages / thirty epochs retained epoch 5, with six worker exits and two coordinator exits recovered. All three backups were restored and terminal checkpoints re-evaluated on the GPU: 423 files / 5,606,740,325 bytes. Local recovery verified 380 metadata files / 45,099,674 bytes plus 21 small arrays / 1,084,032 bytes; all six mixtures reconstructed exactly and evaluation arrays matched.
- Full historical 50/50 M1/M2 reconstructed separately on the GPU. The reset50 smoke's reduced mixture is explicitly different; the two 20/80 smoke arms shared exact S0/S1/D2/M2, including selections/shuffle, with the required different cycle-2 initializers.
- Pod `6osztj3lxeen6h` stopped at **02:46:00 UTC/London**, watchdog reason `validation_budget_guard`; it and volume `gc7nd5v1on` were removed. An [interrupted preliminary attempt](validation/2026-11-01_mixed-campaign-preliminary-recovery.json) exposed backup/status recovery faults, which were fixed and reviewed before final acceptance. Its five temporary Pods and volume were also removed after recovery. No account compute was running at the final check; all five historical volumes remain.
- All new validation/recovery attempts conservatively total **USD 0.46846**, within USD 0.75; observed whole-account debit was **USD 0.42162**, including concurrent historical storage/billing timing. The cumulative ledger estimated USD 25.59434 at reconciliation.
- The full 28-stage / 140-epoch batch is **technically validated and affordable at the recorded prices/credit**: forecast **24.09 hours / USD 13.93** running cost, plus approximately USD 0.35 retained historical storage. Padded watchdog requirement **USD 16.0231** fits the **USD 17.0592** session allowance, covering **25.64 hours**, with USD 2 reserved. Matching GPU stock was available at USD 0.57/hour. These are estimates and saved availability, not a full-run completion claim.

The command in the handoff was subsequently executed once after explicit authorization. The campaign is now active; do not run it again. Final-runtime provider-session resume checks were exercised offline; actual GPU acceptance exercised worker/coordinator restarts within one provider session. Historical full-run bulk recovery remains separate.

## Completed work and recovery

Week 1 pretrained reset, Week 2 continued weights, the matched shared-start reset branch and both Week 3 final-epoch-5 arms are completed and analysed. Use the [reports index](../reports/README.md) for their exact identities, conclusions and verification scope. All are one-seed trajectories; their conditions are not interchangeable.

Recovery volumes remain: Week 1 `wortjkxvr7`, continued `z6ziujxxus`, reset `08ypgkuw40`, final-epoch pair `axxpvi80iq` and mixed `xo027a9d51`. Do not delete recovery storage as repository housekeeping. Published metadata/probe audits do not establish full bulk checkpoint/dataset recovery. Storage continues to accrue until provider-verified deletion.

## Where to update next

The [50/50 analysis and document delivery](validation/2026-11-01_mixed-50-analysis-and-doc.json) is complete. Final real-test PPL is **62.71 versus 107.37** for five-epoch synthetic-only continued training (41.60% lower), but still above S0 **33.89**. Mixing reduces blank-line concentration and average within-output repetition; pooled distinct-2/3 remain slightly lower, and space-heavy output persists. Neither condition preserves S0 performance. All conclusions describe one seed and the fixed probe.

Bulk recovery remains a separate prerequisite to deleting recovery storage. Preserve earlier receipts, forecasts and authored reports. The original long agent chronology is archived in [history](history/2026-10-31_agent-context.md).
