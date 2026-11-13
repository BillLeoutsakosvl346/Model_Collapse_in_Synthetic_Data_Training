# Operations

Use [current state](current-state.md) for recorded run/authorization, [AGENTS.md](../AGENTS.md) for stable rules, and the [catalogue](../experiments/catalog.yaml) for exact configurations. This is a shared runbook; historical commands and receipts are not new execution authority.

## Environments and credentials

Python 3.12 is tested. Follow the [README installation](../README.md#install-and-check-offline) for local CPU PyTorch and pinned dependencies. Local inspection/reporting does not need full checkpoint downloads. Remote environments reuse the official image's GPU PyTorch; do not replace it with a CPU wheel.

Keep `RUNPOD_API_KEY` in ignored `.env`; Hugging Face uses `HF_TOKEN` or its saved token. Optional `HF_MODEL_REPO`/`HF_DATASET_REPO` select private repositories. Never expose credentials in commands, logs, source bundles or reports.

```powershell
python -m model_collapse preflight --probe
```

Preflight is external: it reads live account/stock and exercises a private upload/download/checksum probe. It is not an offline test. Resolve pinned source commits, artifact-index hashes and per-file checksums before renting compute.

A small upload probe establishes access. Private-storage admission must also count existing large objects across repository history and reserve room for all remaining checkpoints and optimization states. The dedicated free project namespace currently contains only its two private experiment repositories. [The read-only host guard](../scripts/operations/hf_storage_admission.py) counts those repositories separately and reserves native Git/metadata growth. Its quota is the recorded plan allowance. Supplying every repository in the namespace is required.

For the repaired completion comparison, [the guarded resume entry point](../scripts/operations/resume_completion64.py) validates the repair receipt and destinations, requires 40 GB additional capacity plus a 10 GB storage margin, then calls the accepted launcher. This host-side addition preserves the accepted worker bundle. The ordinary legacy preflight retains its upload probe. Use current state first and resume only stopped unfinished work.


## Authorization, budget and ownership

- The single [current authorization table](current-state.md#authorized-work-and-limits) links cumulative, allocation, reserve and validation limits. Dated receipts retain older authorizations without making them current again.
- The private ledger records enforceable amounts. Costs are conservative estimates, distinct from account debits. Stop compute accrual only after verified shutdown; close storage at deletion and avoid duplicate intervals across resumes.
- `.state/registry/<run_id>/resources.json` and `.state/ledger.json` bind run/launch/Pod/volume ownership. Historical fetched controls cannot replace them. Act only on task-owned resources. Concurrency requires its exact recorded exception; ordinary launches remain serialized.
- Never buy credits, enable automatic/backup payments or purchase Hugging Face compute. Keep the independent watchdog and private recovery route operational.
- Runtime-changing cleanup creates a candidate needing new GPU acceptance before a separately authorized paid launch. Existing deployed bundles and historical certificates remain unchanged. Do not resume an old run using an incompatible new bundle.

## Validation and full launch

The [SGD/greedy comparison](../comparisons/smollm2_sgd_greedy_effects/README.md) first runs its bounded real-only learning-rate calibration and three-arm acceptance in one GPU session. Its pending rate cannot authorize a full launch. Calibration uses at most 192 updates; acceptance adds 24 logical updates and one deliberately replayed update. Combined paid time is limited to 30 minutes and USD 0.40. The setup must recover all three terminal checkpoints and their diagnostics before deleting temporary resources. A later full launch requires the immutable calibration receipt, accepted benchmark, matching runtime/protocol and a separately funded allocation; setup creates no full-run storage.

The [FineWeb continuation mixtures](../comparisons/fineweb_completion64_mixtures/README.md) extend the completion coordinator with ordered 50/50 and 20/80 conditions. Selection replaces targets at their original prompt rows. Every condition generates with its own preceding checkpoint and carries its own optimizer chain. Its separate USD 10 allocation includes USD 2 reserved and a USD 0.50 combined validation cap. Its later full execution and completion are recorded in current state.

The historical user-authorized parallel smoke exception binds to the registry entry for the existing real/synthetic comparison. It permits one additional validation Pod while checking the protected Pod/volume association. Full-launch active-Pod admission remains enforced. Its private-capacity check counts historical large objects before provisioning, and mixture throughput caches have a separate name. Metadata-only recovery can support validation cleanup after both trained arms have been fully restored and re-evaluated privately, and the shared baseline files have been verified locally.

The [matched FineWeb completion comparison](../comparisons/fineweb_real_vs_synthetic_completion64/README.md) prepares and privately pins a CPU-built 100,000-pair data bank before GPU provisioning. Its `completion_comparison` coordinator runs independent real and synthetic arms sequentially on one 60 GB volume. Both use visible 64-token prompts and score only the next 64 tokens. Its three-cycle-per-arm smoke exercises the production coordinator, persistent optimization state, interrupted generation/training and both private restores. Use the comparison ID for status, resume, fetch, report, validate and recovery-gated cleanup. Full launch requires its own exact runtime/protocol certificate, matching accepted hardware and a funded padded forecast.

The [FineWeb-Edu real-only pilot](../experiments/opt125m_fineweb_edu/fresh_real_one_epoch/README.md) has its own `fresh_real` workflow and runtime/protocol acceptance. It starts from original pretrained OPT, carries AdamW/scheduler state across cycles and uses padded passages up to 512 tokens. Its smoke has three one-epoch cycles, with a combined USD 0.50 allowance across attempts. Full data use ten disjoint 640,000-source-token collections. Use its named configurations with the existing smoke/launch commands. Its forecast uses a separate masked-training/evaluation benchmark and includes data preparation, evaluation and checkpoint/backup allowances.

Offline checks do not rent resources:

```powershell
python -m pytest -q
python scripts/maintenance/check_repository.py
```

For a separately authorized new condition, choose its named profiles. These placeholders deliberately do not choose a completed experiment:

```powershell
python -m model_collapse smoke --config PATH_TO_SMOKE_CONFIG
python -m model_collapse verify-detached RUN_ID
python -m model_collapse fetch RUN_ID --metadata-only
python -m model_collapse validate RUN_ID
```

Smoke is paid. Inspect its actual acceptance scope, recovery, provider shutdown and temporary-resource cleanup. Full launch is a separate action requiring authorization, current runtime/protocol acceptance, live budget admission and matching environment:

```powershell
python -m model_collapse launch --config PATH_TO_FULL_CONFIG --generations 10
```

The catalogued 50/50 continued full run has completed; do not repeat it. For paired or mixed-campaign workflows, use the parent ID for cloud operations. One coordinator owns sequential jobs, budget and shutdown; it does not provision separate GPUs between jobs. Historical shared stages retain their original execution identity and costs.

The [mixed campaign](../comparisons/mixed_fraction_final5/README.md) has three flat logical jobs and one 60 GB network volume (legacy single/paired defaults remain 30 GB). It requires 40 GB free initially and 3 GB before training stages. Full admission covers the measured conservative forecast, padded watchdog runtime, other retained storage and reserve. The [handoff](mixed-campaign-handoff.md) records its acceptance and current affordability separately.

Acceptance belongs to a particular runtime/protocol, not a calendar week or generic "current" certificate. The [reports index](../reports/README.md#gpu-validation) lists separate reset, paired-final-epoch and mixed acceptances. Adding documentation does not change runtime identity; changing runtime, fraction, selection policy or matched settings requires appropriate new acceptance.

Standalone conditions may explicitly use `storage_backend: pod_persistent` when network-volume placement is unavailable. This uses a 30 GB durable `/workspace` disk belonging to the exact Pod; its registered volume ID equals its Pod ID. Stop preserves it; deleting the Pod deletes it. Cleanup still requires verified recovery, and resume reuses the same stopped Pod. The ledger conservatively reserves the stopped-disk rate throughout; legacy network storage remains the default. Paired/campaign workflows reject this option. See the [accepted real-only profiles](../experiments/opt125m_wikitext2/continued_real_only_final5/README.md).

An explicit standalone `--gpu "GPU NAME"` resolves hardware before acceptance/admission. Smoke-only `--gpu auto` restricts discovery to modern physical NVIDIA GPUs up to USD 1.19/hour, excluding MIG; full launch must pin its accepted GPU. Resume keeps recorded hardware. GPU or storage changes require matching acceptance.

`--config` and legacy `--mode` are mutually exclusive. Legacy `fresh_base`/`continued` keep their original meaning; the [legacy configuration map](../configs/README.md) identifies defaults. A new scientific condition gets a new run ID, not a resume of completed compute.

## Detached execution and status

Keep the laptop awake until the launcher returns `detached_supervisor_and_watchdog_verified: true`. For supported mixed/paired runs, `verify-detached` additionally observes stage/epoch/event progress and advancing heartbeats across fresh, closed SSH connections. The remote worker, backups and independent watchdog then operate without the laptop, browser, terminal or Codex session.

```powershell
python -m model_collapse status RUN_ID
```

A running Pod is not proof of healthy training. Completion requires the expected stages, terminal/selected checkpoints, data lineage, successful private backup and provider-confirmed stopped compute. The watchdog stops on completion, failure or budget exhaustion and retries transient API failures; provider outages can still delay shutdown.

Resume only an unfinished, stopped run with compatible source/settings, verified completed stages and retained storage:

```powershell
python -m model_collapse resume RUN_ID
```

Resume can provision paid compute. Never resume an active/completed run or use resume to change its scientific condition. A new provider session has a separately accounted deadline; restarting a process within a session does not reset it. Sealed completed jobs retain their original registered execution identities and are skipped. Unsealed partial jobs are conservatively forecast as whole jobs, even when their completed stages can be reused.

## Fetch and report

```powershell
python -m model_collapse fetch RUN_ID --metadata-only
python -m model_collapse report RUN_ID
python -m model_collapse report RUN_ID --out reports/RUN_ID/generated/local-review
```

Metadata-only fetch leaves weights/training arrays in the cloud. Full `fetch RUN_ID` can download several GB. Recovery receipts distinguish metadata-only, probe-verified and bulk-verified scope; a valid index does not verify absent bulk objects.

Reporting is offline: it uses saved measurements and writes a new output directory with input/output hashes, analysis revision and environment. Authored reports, historical CSVs and raw records are protected. Independent comparison audits remain separate from presentation generation. See [analysis tools](../scripts/analysis/README.md) for figure/report reproduction and acceptance export.

## Retention and cleanup

```powershell
python -m model_collapse cleanup RUN_ID
```

Cleanup deletes external resources and requires matching ownership plus adequate recovery. Paired cleanup requires both arms; campaign cleanup requires all three jobs. Metadata/probe-only full-run recovery does not authorize deleting its volume. Temporary smoke cleanup can use its explicit verified remote bulk-restore audit when supported. An interrupted campaign with jobs never created needs separately recorded recovery of every existing artifact and ownership-checked operator cleanup; it cannot claim successful three-job acceptance.

Historical volumes and the active run's volume are listed in [current state](current-state.md#completed-work-and-recovery-obligations). Preserve them during repository cleanup. Storage continues billing after compute stops; read live prices before financial decisions. Verify provider read-back and close the storage ledger interval after actual deletion.

The [manual shutdown-timeout probe](../scripts/operations/README.md) also rents a GPU. It must never run implicitly during installation, tests or reporting. Source archives, private control records and recovery workspaces are not disposable caches.
