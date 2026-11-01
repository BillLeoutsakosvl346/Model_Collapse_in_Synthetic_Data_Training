# Three-experiment mixed-data campaign: setup handoff

**Setup completed on 1 November 2026: technically validated and affordable at the recorded prices/balance. The full campaign is unlaunched.** Acceptance `20261101T022210Z-smoke-mixed-campaign` passed all three jobs and the parent GPU/provider checks. [Setup receipt](validation/2026-11-01_mixed-campaign-setup.json) · [Validation report and figures](../reports/20261101T022210Z-smoke-mixed-campaign/report.md).

The [campaign definition](../comparisons/mixed_fraction_final5/README.md) runs 50/50 reset, 20/80 continued and 20/80 reset sequentially on one matching RTX PRO 4000 Blackwell GPU and a new 60 GB Standard network volume. It has one authoritative resource owner, ledger interval, supervisor/watchdog and deadline. Historical recovery resources remain preserved.

## Scientific workload

- Full: reset50 cycles 2–10, continued20 cycles 1–10, reset20 cycles 2–10; 28 new stages / 140 epochs.
- Smoke: reset50 cycles 2/3, continued20 cycles 1/2, reset20 cycles 2/3; six new stages / thirty actual epochs.
- Every stage trains five epochs, retains epoch 5 and restarts AdamW/cosine scheduling. Each full mixture has 37,946 blocks / 2,428,544 tokens; 50/50 counts are 18,973/18,973 real/synthetic and 20/80 counts are 30,357/7,589.
- Reset50 imports the completed `20261031T163428Z-full-continued` S0/S1/D2. Reset20 imports the new continued20 S0/S1/D2 only after completed private backup sealing. Both full arms within a ratio must use identical cycle-2 M2 tokens/selections.
- The reset50 smoke mixture uses a reduced sampling population and differs from full historical M2; full-source reconstruction is checked separately on the GPU. Imported S0/S1 metrics retain their original full-split scope.

## Verified implementation

- **345 offline tests passed**, with seventeen dependency warnings, in the final integrated suite. The original 170-test baseline was retained.
- A fresh independent final review passed **92 focused tests**, with no material findings remaining. [Recovery/review receipt](validation/2026-11-01_mixed-campaign-recovery-review.json); preceding [transport review](validation/2026-11-01_mixed-campaign-transport-review.json) and [implementation review](validation/2026-11-01_mixed-campaign-independent-review.json) preserve earlier checks.
- Implementation commit: `92481b7671f73881f3f6ddadbe81378815b57bc8`.
- Accepted runtime: `35c89c3b9fb78c7967c4f62f07972739eabbe57b353a0360749eba71d1af2171`.
- Accepted campaign protocol: `770a627c02821b1301f3194268e65fb9635eccdc6a43e639477d886f26dac377`.

An earlier validation attempt exposed a checkpoint-backup transport failure and stale-status handling during recovery. The frozen implementation includes bounded retries for the observed missing-LFS diagnostic, immutable source/ownership checks and corrected status handling. Earlier partial success does not certify this final runtime; no scientific protocol or training-engine change was introduced by these recovery fixes.

The [preliminary recovery receipt](validation/2026-11-01_mixed-campaign-preliminary-recovery.json) records twenty executed epochs, private recovery and deletion of all five preliminary/recovery Pods and their temporary volume after verification. Its conservative closed cost is **$0.204170**, included in the shared $0.75 validation cap; it is not an itemized provider invoice or a complete campaign acceptance.

## Completed GPU acceptance and recovery

- **All ten parent checks passed:** six actual new stages / thirty epochs, terminal epoch-5 checkpoints, both initializer policies, immutable mixtures, exact shared S0/S1/D2/M2, automatic handoffs and unchanged completed evidence.
- **All child checks passed:** reset50 16/16, continued20 14/14 and reset20 16/16. Six actual worker exits and two coordinator exits recovered; progress across closed SSH connections was verified.
- Each child restored its private backup and re-evaluated its terminal checkpoint: **423 files / 5,606,740,325 bytes** collectively. Local recovery checked **380 metadata files / 45,099,674 bytes** plus **21 small smoke arrays / 1,084,032 bytes**, with all six mixtures reconstructed and equal fixed evaluation arrays. Model weights were not downloaded to the laptop.
- Historical full M1/M2 reconstruction passed separately from the reduced reset50 smoke mixture. The 20/80 cycle-2 arrays were directly byte-compared and matched despite different S0/S1 initializers.
- The independent budget watchdog stopped Pod `6osztj3lxeen6h`; Runpod confirms shutdown at **02:46:00 UTC/London**. That Pod and temporary 60 GB volume `gc7nd5v1on` were deleted after recovery. All preliminary/recovery resources were also removed. No account compute was active at the final check; all five historical recovery volumes remain preserved.

[Dataset examples](../reports/20261101T022210Z-smoke-mixed-campaign/dataset_preview.json) and [local reconstruction proof](../reports/20261101T022210Z-smoke-mixed-campaign/local_mixture_reconstruction.json) accompany the report. The certificate covers the **sequential campaign workflow**; standalone condition launches retain separate acceptance requirements. Smoke results validate execution, not collapse or mitigation. Final-runtime worker/coordinator recovery was exercised on the GPU; a separate provider stop/reprovision/resume cycle was not. Prior-session ownership and provider-resume admission are covered by offline regressions.

## Budget and launch readiness

The explicitly authorized cumulative ceiling is **$45**, preserving a **$2 reserve**, existing credit only and no credit purchases or payment-setting changes. [Authorization receipt](validation/2026-11-01_mixed-campaign-budget-authorization.json). **All new validation attempts combined are capped at $0.75**, within the remaining setup allowance; the configured smoke session is capped at $0.65. Historical receipts and costs are preserved.

All new validation attempts closed at a conservative **$0.468458**, below $0.75. The observed whole-account debit was **$0.421623**, including concurrent historical storage and billing timing, not an itemized validation-compute invoice. The recorded cumulative task estimate was **$25.59434**. The later launcher rechecks matching hardware, the source-bound certificate, current credit/ledger and both padded cost/runtime coverage.

| Readiness gate | Recorded state |
| --- | --- |
| Code, offline checks and independent review | Passed |
| Complete final-runtime GPU/provider acceptance | Passed |
| Temporary-resource cleanup and recovery verification | Passed |
| Full-batch affordability | Passed at the setup receipt's recorded credit and prices |
| Full training | Unlaunched; requires a separate instruction |

## Measured throughput and conditional full forecast

The final continued20 worker's warmed production-batch benchmark measured **38,095 training target tokens/s** and **1,438 generated tokens/s**, using effective batch 128, microbatch 32 and generation batch 32 with five beams. These are benchmark measurements, not completed full-data timings. The controller's forecast includes 1.5x compute margins, setup/checkpoint/upload allowances and two source handoffs.

| Planned full work | New stages / full pools | Conservative hours | Compute and running storage |
| --- | --- | ---: | ---: |
| 50/50 reset | 9 / 8 | 7.54 | $4.36 |
| 20/80 continued | 10 / 9 | 8.51 | $4.92 |
| 20/80 reset | 9 / 8 | 7.54 | $4.36 |
| Two source-handoff allowances | — | 0.50 | $0.29 |
| **Whole campaign** | **28 / 25** | **24.09** | **$13.93** |

This conditional estimate uses the observed $0.57/hour GPU rate plus 60 GB Standard network storage and 20 GB container storage, giving $0.578493/hour. Admission requires **$16.0231 of watchdog allowance**; the recorded allowance after historical-storage reservation was **$17.0592**, providing **25.64 hours** at the 15%-padded guard rate against the **24.09-hour** forecast. Existing retained volumes add approximately **$0.35** during that forecast, separately from the new running volume. Seven days of the new 60 GB volume costs approximately **$0.97**, covered within the $2 reserve. Matching hardware was available with LOW stock in EU-RO-1 at the check. Historical execution suggests 17–21 hours may be feasible, but it is not the admission forecast or a guarantee. Live prices, credit and stock are rechecked at launch.

## Commands

The validation commands below describe the **already completed** acceptance. Repeating GPU smoke is paid and is not required to launch the accepted full campaign:

```powershell
python -m model_collapse smoke --config comparisons/mixed_fraction_final5/smoke.yaml
```

Do not resume or relaunch the completed smoke. Its temporary resources are already removed. Offline reports can be regenerated into a new destination without training; published evidence remains unchanged.

Future full launch, left unexecuted during setup:

```powershell
python -m model_collapse launch --config comparisons/mixed_fraction_final5/full.yaml --generations 10
```

For a later full run, use its returned campaign ID for `status`, `fetch --metadata-only`, `report`, `resume` and `cleanup`, not a child logical ID. `verify-detached` applies while that Pod is active. Resume only unfinished stopped work. Failure or budget pause blocks subsequent jobs; completed stages remain unchanged across recovery. Cleanup requires verified recovery of all three jobs. Metadata-only fetching leaves bulk checkpoints/full arrays in the cloud.

The eventual launcher returns after detached supervision, the independent watchdog and actual progress are verified. At that point closing the laptop does not stop the campaign. Completion/failure/budget exhaustion stops compute; the volume remains until recovery and explicit cleanup. Published historical evidence, private credentials and recovery resources remain preserved.
