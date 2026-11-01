# Three-experiment mixed-data campaign: setup handoff

**Implementation and acceptance are pending. The full campaign is unlaunched.** This document will be completed with measured checks, runtime/protocol identities, spending, recovery evidence and technical/financial readiness after validation. No pending item below is a completion claim.

The [campaign definition](../comparisons/mixed_fraction_final5/README.md) runs 50/50 reset, 20/80 continued and 20/80 reset sequentially on one matching RTX PRO 4000 Blackwell GPU and a new 60 GB Standard network volume. It has one authoritative resource owner, ledger interval, supervisor/watchdog and deadline. Historical recovery resources remain preserved.

## Scientific workload

- Full: reset50 cycles 2–10, continued20 cycles 1–10, reset20 cycles 2–10; 28 new stages / 140 epochs.
- Smoke: reset50 cycles 2/3, continued20 cycles 1/2, reset20 cycles 2/3; six new stages / thirty actual epochs.
- Every stage trains five epochs, retains epoch 5 and restarts AdamW/cosine scheduling. Each full mixture has 37,946 blocks / 2,428,544 tokens; 50/50 counts are 18,973/18,973 real/synthetic and 20/80 counts are 30,357/7,589.
- Reset50 imports the completed `20261031T163428Z-full-continued` S0/S1/D2. Reset20 imports the new continued20 S0/S1/D2 only after completed private backup sealing. Both full arms within a ratio must use identical cycle-2 M2 tokens/selections.
- The reset50 smoke mixture uses a reduced sampling population and differs from full historical M2; full-source reconstruction is checked separately on the GPU. Imported S0/S1 metrics retain their original full-split scope.

## Acceptance record to complete

- Existing baseline and new offline regressions: pending.
- Fresh independent integrated review and runtime freeze: pending.
- Actual GPU updates, epoch-5 reloads, all initializers/data handoffs and immutable mixture reconstruction: pending.
- Worker/coordinator/shard recovery, detached progress and both automatic handoffs: pending.
- Private backup restoration, checksum verification and restored-checkpoint evaluation for all three jobs: pending.
- Independent watchdog and provider-confirmed shutdown, local evidence retrieval and temporary-resource deletion: pending.

The acceptance certificate must bind the exact runtime and all three job protocols. Historical smoke certificates cannot certify changed code, fractions or initialization policies. Smoke results are pipeline validation, not evidence of collapse prevention.

## Budget and launch readiness

The selected cumulative ceiling is **$45**, preserving a **$2 reserve**, existing credit only and no credit purchases or payment-setting changes. **All new validation attempts combined are capped at $0.75**, within the remaining setup allowance; the configured smoke session is capped at $0.65. Record updated authorization and reconcile live account/resource state before paid work.

Actual acceptance spending, per-job measured throughput, conservative full forecast, padded watchdog runtime coverage and remaining credit are pending. The later launcher must refuse when matching hardware, a source-bound certificate, sufficient credit or affordable cost/deadline coverage is unavailable. Technical validation and affordability are separate readiness assessments.

## Commands

Setup validation:

```powershell
python -m model_collapse smoke --config comparisons/mixed_fraction_final5/smoke.yaml
python -m model_collapse verify-detached CAMPAIGN_ID
python -m model_collapse status CAMPAIGN_ID
python -m model_collapse fetch CAMPAIGN_ID --metadata-only
python -m model_collapse validate CAMPAIGN_ID
python -m model_collapse report CAMPAIGN_ID
python -m model_collapse cleanup CAMPAIGN_ID
```

Future full launch, left unexecuted during setup:

```powershell
python -m model_collapse launch --config comparisons/mixed_fraction_final5/full.yaml --generations 10
```

Use the campaign ID for cloud operations, not a child logical ID. Resume only unfinished stopped work. Failure or budget pause blocks subsequent jobs; completed stages remain unchanged across recovery. Cleanup requires verified recovery of all three jobs. Metadata-only fetching leaves bulk checkpoints/full arrays in the cloud.

The eventual launcher returns after detached supervision, the independent watchdog and actual progress are verified. At that point closing the laptop does not stop the campaign. Completion/failure/budget exhaustion stops compute; the volume remains until recovery and explicit cleanup. Published historical evidence, private credentials and recovery resources remain preserved.
