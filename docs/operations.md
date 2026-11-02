# Operations

Use [current state](current-state.md) for recorded run/authorization, [AGENTS.md](../AGENTS.md) for stable rules, and the [catalogue](../experiments/catalog.yaml) for exact configurations. This is a shared runbook; historical commands and receipts are not new execution authority.

## Environments and credentials

Python 3.12 is tested. Follow the [README installation](../README.md#install-and-check-offline) for local CPU PyTorch and pinned dependencies. Local inspection/reporting does not need full checkpoint downloads. Remote environments reuse the official image's GPU PyTorch; do not replace it with a CPU wheel.

Keep `RUNPOD_API_KEY` in ignored `.env`; Hugging Face uses `HF_TOKEN` or its saved token. Optional `HF_MODEL_REPO`/`HF_DATASET_REPO` select private repositories. Never expose credentials in commands, logs, source bundles or reports.

```powershell
python -m model_collapse preflight --probe
```

Preflight is external: it reads live account/stock and exercises a private upload/download/checksum probe. It is not an offline test. Resolve pinned source commits, artifact-index hashes and per-file checksums before renting compute.

## Authorization, budget and ownership

- The latest [authorization](validation/2026-11-02_real-only-budget-authorization.json) raises the cumulative ceiling to USD 50 for the explicitly requested real-only control, retaining the USD 2 reserve and existing setup/validation caps. Earlier USD 25/30/45 receipts remain historical; account balance alone never increases authorization. The real-only GPU validation session is limited to USD 0.25.
- Standalone, paired and campaign launches now honor an explicitly recorded cumulative ceiling in the private ledger. Standalone/paired legacy ledgers without that field retain the original default; campaigns still require explicit authorization. The original default and [historical USD 30 launch override](validation/2026-10-31_mixed-50-full-launch.json) retain their recorded meanings. This admission fix changes runtime identity and requires new GPU acceptance; a ledger-only future ceiling update does not require a runtime edit.
- Ledger costs are conservative estimates, distinct from account debits. Preserve historical receipts. Stop compute accrual at verified shutdown, close storage only at deletion and avoid duplicate intervals across resumes.
- `.state/registry/<run_id>/resources.json` and `.state/ledger.json` bind run/launch/Pod/volume ownership. A fetched historical control file must not replace them. Act only on task-owned resources; at most one GPU Pod at a time.
- Never buy credits, enable automatic/backup payments or purchase Hugging Face compute. Keep the independent watchdog and private recovery route operational.

## Validation and full launch

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

Historical volumes and the active run's volume are listed in [current state](current-state.md#completed-work-and-recovery). Preserve them during repository cleanup. Storage continues billing after compute stops; read live prices before financial decisions. Verify provider read-back and close the storage ledger interval after actual deletion.

The [manual shutdown-timeout probe](../scripts/operations/README.md) also rents a GPU. It must never run implicitly during installation, tests or reporting. Source archives, private control records and recovery workspaces are not disposable caches.
