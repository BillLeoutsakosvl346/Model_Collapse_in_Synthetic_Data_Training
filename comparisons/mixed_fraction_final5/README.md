# Mixed-data reset/continued campaign

**Implementation and acceptance are pending; full training is unlaunched.** This campaign prepares three logical experiments under one detached coordinator, one GPU Pod, one 60 GB Standard network volume, one cost interval and one independent watchdog deadline.

| Ordered job | Source and initializer | New full stages | Blocks per mixture: real / synthetic |
| --- | --- | --- | --- |
| [50/50 reset](../../experiments/opt125m_wikitext2/reset_mixed_50_final5/README.md) | Exact completed `20261031T163428Z-full-continued` S0/S1/D2; initialize every new stage from S0 | 2–10 | 18,973 / 18,973 |
| [20/80 continued](../../experiments/opt125m_wikitext2/continued_mixed_20_final5/README.md) | Historical S0/D1; initialize from previous epoch-5 weights | 1–10 | 30,357 / 7,589 |
| [20/80 reset](../../experiments/opt125m_wikitext2/reset_mixed_20_final5/README.md) | Newly completed 20/80 continued S0/S1/D2; initialize every new stage from S0 | 2–10 | 30,357 / 7,589 |

There are 28 unique new training stages / 140 epochs. Every stage executes all five epochs and retains epoch 5, even if validation loss worsens. Every full dataset has 37,946 independent 64-token blocks / 2,428,544 tokens. Source fractions round half upwards by block; the real source is the existing WikiText training split. Model/data revisions, seed 42, fixed real prompts/evaluation, decoding, optimizer restarts, scheduler, batches and precision match the completed experiments.

## Shared datasets and provenance

Within each ratio, the reset and continued arms share exact S0/S1/D2 and the cycle-2 mixed array M2, including source indices, labels and shuffle. Later pools are arm-specific. Generation prompts do not enter synthetic training targets. Sampling is without replacement within each source/cycle, and a saved mixture is immutable throughout its five epochs and recovery.

The importer preserves historical source manifests. A branch-local manifest may gain import metadata; exact token/selection identity is checked separately from that expected operational metadata difference. Imported measurements retain original identities, scope and costs rather than being counted as new execution.

The coordinator advances only after verified completion and pinned backup. It seals the completed 20/80 continued source before starting its reset arm. Failure or budget pause blocks later jobs; resume verifies completed stages and skips them. Historical recovery volumes and published results remain untouched.

## Validation and operations

The smoke campaign runs reset50 stages 2/3, continued20 stages 1/2, then reset20 stages 2/3: six actual new stages / thirty epochs. Each uses 256 training / 64 validation / 64 test blocks, with 128/128 or 205/51 real/synthetic counts. The reduced reset50 mixture is not the full historical M2; acceptance separately reconstructs and hashes full M2 on the GPU. Imported full-source S0/S1 measurements are labelled with their original scope. Smoke results validate the pipeline and cannot establish collapse or mitigation.

```powershell
python -m model_collapse smoke --config comparisons/mixed_fraction_final5/smoke.yaml
python -m model_collapse verify-detached CAMPAIGN_ID
python -m model_collapse status CAMPAIGN_ID
python -m model_collapse fetch CAMPAIGN_ID --metadata-only
python -m model_collapse validate CAMPAIGN_ID
python -m model_collapse report CAMPAIGN_ID
python -m model_collapse cleanup CAMPAIGN_ID
```

Use the parent campaign ID for cloud operations. Child directories have distinct logical IDs but do not own independent provider resources or shutdown deadlines. `resume CAMPAIGN_ID` applies only to unfinished stopped campaigns; completed/active work cannot be restarted. Cleanup requires verified recovery of all three jobs. Metadata-only fetching leaves weights/full arrays in private cloud storage.

Future full launch, **not executed during setup**:

```powershell
python -m model_collapse launch --config comparisons/mixed_fraction_final5/full.yaml --generations 10
```

The full launcher requires acceptance for the exact runtime and all three scientific profiles, matching GPU/software, sufficient account credit and affordable cost/deadline forecasts. After it verifies detached supervision, watchdog and progress, the laptop can close. Compute stops after completion, failure or budget exhaustion; storage remains for recovery.

The authorized cumulative ceiling is $45, with a $2 storage/shutdown reserve and no credit purchases/payment changes. All new acceptance attempts share a $0.75 ceiling within the setup allowance; the smoke session has a $0.65 cap. Live reconciliation, measured forecasts and final readiness belong in the [handoff](../../docs/mixed-campaign-handoff.md). Existing historical receipts preserve their earlier limits.

Metrics and probe limitations follow [shared definitions](../../docs/metrics.md). This is one seed per full condition; no independent-run significance or universal safe synthetic fraction is implied.
