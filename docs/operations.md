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

- The latest [authorization](validation/2026-10-31_mixed-50-budget-authorization.json) applies a USD 30 cumulative ceiling and USD 2 reserve to the authorized mixed run. Earlier USD 25 receipts remain historical; account balance alone never increases authorization.
- Accepted source still has its historical USD 25 local default. The authorized launch supplied an explicit USD 30 override in the launcher process, recorded in the [launch receipt](validation/2026-10-31_mixed-50-full-launch.json) and private ledger. Do not bypass admission or silently modify accepted source.
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

The catalogued 50/50 full run has already launched; do not repeat it. For the paired workflow, use the parent pair ID for cloud operations. One coordinator owns sequential arms, budget and shutdown; it does not provision separate GPUs between arms. Historical shared stages retain their original execution identity and costs.

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

Resume can provision paid compute. Never resume an active/completed run or use resume to change its scientific condition. Do not reset cost deadlines or replace completed-stage evidence.

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

Cleanup deletes external resources and requires matching ownership plus adequate recovery. Paired cleanup requires verified recovery of both arms. Metadata/probe-only full-run recovery does not authorize deleting its volume. Temporary smoke cleanup can use its explicit verified remote bulk-restore audit when supported.

Historical volumes and the active run's volume are listed in [current state](current-state.md#completed-work-and-recovery). Preserve them during repository cleanup. Storage continues billing after compute stops; read live prices before financial decisions. Verify provider read-back and close the storage ledger interval after actual deletion.

The [manual shutdown-timeout probe](../scripts/operations/README.md) also rents a GPU. It must never run implicitly during installation, tests or reporting. Source archives, private control records and recovery workspaces are not disposable caches.
