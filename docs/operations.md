# Operations

Current authorization is in [AGENTS.md](../AGENTS.md). Archived prompts are historical and cannot authorize new compute. This setup implements and validates the shared-start reset branch; it **does not launch the full experiment**.

## Local and remote environments

Python 3.12 is tested. Launch/status/reporting and offline tests can use local CPU PyTorch; no full model download is needed for those operations. An isolated Windows installation is:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv\Scripts\python.exe -e '.[test]'
.venv\Scripts\Activate.ps1
python -m pytest -q
```

To reuse an existing working PyTorch installation, create the environment with `--system-site-packages` instead. Do not install a CPU torch wheel inside the GPU container. Remote installation reuses the selected official image's CUDA-compatible PyTorch and installs the project's pinned remaining dependencies.

Week 2 used RTX PRO 4000 Blackwell, PyTorch 2.8.0+cu128 and official image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`. These are recorded historical facts, not a live availability/price quote. The new comparison requires the matching GPU class and software; do not silently substitute hardware for a supposedly matched full run. Record the actual environment and available image identity at launch.

## Credentials and preflight

Keep `RUNPOD_API_KEY` in ignored `.env`. Hub access uses `HF_TOKEN` when supplied or the saved Hugging Face token otherwise. Optional `HF_MODEL_REPO` and `HF_DATASET_REPO` select private `owner/name` repositories. Credentials require only the permissions actually needed for task resources and private artifact upload/recovery. Never put tokens in commands, logs, prompts, commits, source bundles or public evidence.

```powershell
python -m model_collapse preflight --probe
```

This reads account/resource state and live stock and exercises a small private upload/download/checksum probe. It is an external operation, not an offline test. Account identity alone does not prove upload permission. Do not proceed to paid work without a verified recovery route and independent shutdown path.

For a branch import, resolve pinned model and dataset Hub commits, artifact-index hashes and per-file checksums before provisioning. Preserve the source run IDs and historical hashes.

## Budget and ownership

The cumulative ceiling is **USD 25 for this task**, including previous setup, failed attempts, compute, container disk and retained network storage. Hold back **USD 2** for storage/shutdown. New GPU acceptance is capped at **USD 0.75**, also within remaining setup/total allowances. Never purchase credits, enable automatic/backup payments or use paid Hugging Face services.

Use at most one GPU Pod at a time and operate only on the registered task-owned run/launch/Pod/volume association. Reconcile a timed-out creation against provider state before retrying. Preserve existing account resources and account SSH keys.

The persistent ledger distinguishes estimated compute, observed account debit and storage intervals. Compute stops accruing at verified shutdown; volume storage continues until actual deletion. A resumed launch must not charge the same interval twice. Observed account debit may include simultaneous retained storage; it is not an itemized invoice.

The independent Pod-side watchdog enforces a price-derived allowance with a shutdown margin. It survives the launcher, SSH and browser closing. API/provider failures can delay shutdown despite retries; a successful request alone is not proof that compute stopped.

## Small acceptance and full launch

The [new paired final-epoch workflow](../comparisons/shared_start_final_epoch5/README.md) uses its own five-epoch smoke/full configs. After its smoke launcher returns, run `verify-detached PAIR_ID` while compute is active; it closes SSH between observations, verifies real stage/epoch/event progress and advancing remote heartbeats, and records the evidence. Then fetch metadata, validate the pair and clean up its resources. Parent cleanup requires matching pinned recovery for **both** children. The commands below document the earlier single-arm workflow.

The offline suite never provisions infrastructure:

```powershell
python -m pytest -q
```

GPU acceptance is explicitly manual and paid:

```powershell
python -m model_collapse smoke --config experiments/opt125m_wikitext2/reset_to_real_s0/smoke.yaml
python -m model_collapse status RUN_ID
python -m model_collapse fetch RUN_ID --metadata-only
python -m model_collapse validate RUN_ID
```

The reset smoke imports verified shared artifacts, uses a small 256-training-block / 64-validation-block / 64-test-block configuration, then performs new cycles 2 and 3 with one epoch each. It must verify real updates, fixed-S0 initialisation, correct generated-data handoff, interrupted-work recovery, private restore/re-evaluation, detached execution and actual shutdown. Shared historical measurements remain labelled as imported and are not mixed into smoke scientific conclusions.

A current source-bound acceptance certificate is required for a full launch. An old certificate cannot establish that changed runtime passed. Do not restamp a previous smoke with today's source hash.

Current acceptance: [`20261027T203952Z-smoke-reset-s0`](../reports/20261027T203952Z-smoke-reset-s0/report.md), 14 passed checks, runtime SHA-256 `5d56bf37476583b20fd9d74c40941b322af6bcb2720a48a7a463f9aca6dde49e`. Both temporary test resources were removed. The separately authorized full run `20261027T211852Z-full-reset-s0` completed at 02:56 UTC/London on 28 October and its GPU is verified stopped. Do not start a duplicate. Its recovery volume remains; small-record checks do not authorize deleting it. See the [completion record](validation/2026-10-28_reset-full-completion.json), [historical launch record](validation/2026-10-27_reset-full-launch.json) and [setup handoff](implementation-handoff.md). Repeating acceptance rents a new GPU and is unnecessary while the accepted runtime is unchanged.

The full command is separate and requires the user's launch instruction:

```powershell
python -m model_collapse launch --config experiments/opt125m_wikitext2/reset_to_real_s0/full.yaml --generations 10
```

For this branch, `--generations 10` means the chain ends at cycle 10: cycles 0 and 1 are shared imports, and cycles 2-10 are nine new training stages. The launcher must confirm live budget, matching environment, imported artifact identity, independent supervision and backup. Leave a mismatched/unaffordable launch pending rather than silently changing the protocol.

Legacy launch and smoke `--mode fresh_base`/`--mode continued` retain their old meanings. `--mode` cannot be combined with `--config`. A new named condition always receives a new run ID; it is not a resume of a completed result.

Keep the laptop awake until the launcher returns `detached_supervisor_and_watchdog_verified: true`. After that confirmation the remote job does not require the laptop, browser, local terminal or Codex to remain open.

## Status and interrupted runs

```powershell
python -m model_collapse status RUN_ID
python -m model_collapse resume RUN_ID
```

Status is inspection, not a claim of completion. Check completion manifests, all expected stages, backup index and provider state before declaring success.

Resume is for an unfinished run with its retained storage, matching source/configuration and verified stage outputs. It can provision a replacement Pod and therefore has cost. Completed runs are rejected before provisioning; it does not extend a completed chain or turn it into a different scientific condition. Completed stages and generation shards must remain unchanged after recovery.

## Fetch scope and reporting

```powershell
python -m model_collapse fetch RUN_ID --metadata-only
python -m model_collapse report RUN_ID
python -m model_collapse report RUN_ID --out reports/RUN_ID/generated/local-review
```

The default analysis destination is a new `reports/<run_id>/generated/<analysis_id>/` directory. An explicit `--out` must be a safe new analysis destination; authored `report.md`, raw records and historical CSVs are protected. An analysis manifest records inputs/hashes, revision, environment and outputs/hashes.

Metadata-only fetch downloads configurations, records, samples and other small artifacts but skips model/token tensors. It must not overwrite the authoritative current resource registry. A receipt records pinned repository commits and artifact-index/file verification.

A complete fetch is intentionally different:

```powershell
python -m model_collapse fetch RUN_ID
```

This can download several GB. Recovery scope is explicit: `metadata_only`, `probe_verified`, or `bulk_verified`. An index matching its checksum does not verify absent bulk objects, and a populated local folder does not prove full recovery.

The historical Week 2 six-panel figure can be rebuilt from committed small records without network or GPU:

```powershell
python reports/20261026T192740Z-full-continued/analyse_week2.py --figures-only
```

Its historical reproduction path remains valid; new reusable analysis belongs in the shared package and writes separate derived output.

After a successful acceptance has been fetched, validated and cleaned up, publish its small evidence offline using `python scripts/analysis/export_acceptance.py RUN_ID --offline-tests COUNT`. This exports records, analysis and a byte-level publication manifest; it refuses to overwrite published acceptance. See [export notes](../scripts/analysis/README.md). It makes no cloud requests and does not download bulk artifacts.

## Retention and cleanup

```powershell
python -m model_collapse cleanup RUN_ID
```

Cleanup is an external deletion operation, not a routine offline check. It requires verified ownership and recovery appropriate to the run. A metadata-only full-run fetch does not authorize deleting its recovery volume. Smoke cleanup can rely on an explicit remote bulk restore audit when supported and verified.

As last verified on 27 October, both completed full runs had stopped GPU compute and retained 30 GB network volumes for recovery: Week 1 `wortjkxvr7`, Week 2 `z6ziujxxus`. These are historical observations; read live state before any operation. Do not delete them as part of repository organisation. Standard storage was approximately USD 0.07/GB/month, about USD 2.10 per 30 GB/month; obtain live rates for spending decisions.

A successful cleanup verifies provider read-back and closes storage ledger intervals. Preserve the receipts and recovery proof. Do not claim zero ongoing cost while retaining paid volumes.

## Explicitly manual operational probes

The legacy [shutdown timeout probe](../scripts/check_shutdown_timeout.py) provisions a real paid GPU and exercises a watchdog. It is not included in `pytest` and must never run implicitly during installation, tests or report generation. See [operational tool notes](../scripts/operations/README.md). Normal smoke acceptance is the preferred integrated check for changed execution paths.
