# Repository architecture

The project has one scientific implementation and several named experimental conditions. Calendar weeks identify reports; they do not select training behaviour.

The standalone `gradient_diagnostic` workflow uses `science/gradients.py` for FP32 source-mean gradients and isolated AdamW steps, `artifacts/gradient_sources.py` for pinned selective source recovery, and a dedicated guarded cloud launcher. Its data and reports are separate from recursive-training runs. It reuses the shared model/loss helpers, resource registry, ledger, supervision, watchdog and artifact publication; it never changes completed training stages.

## Identities and responsibility

| Identity | Meaning | Home |
| --- | --- | --- |
| Condition | Scientific rules independent of a particular execution | `experiments/<model_dataset>/<condition>/` |
| Comparison | Arms, shared stages, matching requirements and known confounders | `comparisons/<comparison>/` |
| Run | One seeded realization with fixed resolved configuration | `runs/<run_id>/`, plus published `reports/<run_id>/` |
| Launch | One provisioning/resume event, with a returned `launch_id` | Private resource registry and cost ledger |
| Stage | Training inputs, generator, initializer and selected output | Hashed per-cycle records and completion manifests |
| Analysis | A reproducible derivation from recorded evidence | A new `generated/<analysis_id>/` directory |

An experiment definition is not a run; a directory is not proof of completion; a launch is not a completed result. Importing shared evidence does not create a second measurement event or a second charge.

## Package boundaries

The root-level package and `python -m model_collapse` CLI remain stable. These are the actual implementation homes:

| Package | Implementation and responsibility |
| --- | --- |
| `science/` | `engine.py`: shared model IO, training, generation and evaluation; `decoding.py`: explicit training decoders and stable sampling seeds; `wikitext.py`: split packing/arrays; `metrics.py`: output statistics; `sampling.py`: pure mixture counts; `identity.py`: legacy int32 token hashes |
| `workflow/` | `wikitext.py`, `fresh_real.py`, `completion.py`: scientific loops; `paired.py` and `mixed_campaign.py`: coordinators; selection/optimization/configuration: policies; lineage/branch/initial_data: verified inputs; registry: lazy CLI worker/report/validation routing |
| `cloud/` | `execution.py`: guarded launch, SSH, resource registry and operations; `runpod.py`: API client; `supervisor.py`: detached processes and stdlib-only watchdog; `ownership.py`: states/session association; `ledger.py`/`budget.py`: transitions/costs; parallel/S3/private-capacity adapters |
| `artifacts/` | `hub.py`: private backups, pinned fetch and restoration; `deployment.py`: immutable source bundles; `paths.py`: safe artifact destinations |
| `analysis/` | Historical schema readers, forecasts, single-run/FineWeb reports and the paired/campaign renderers; input/source/environment/output provenance |

### Compatibility and frozen data builders

The SmolLM2 portability profile lives in `workflow/smollm.py`; its bounded acceptance lives in `workflow/smollm_validation.py`. They reuse `completion.py` and the shared engine. `science/completion_bank_v2.py` is the versioned native-tokenizer builder; it does not replace the byte-preserved OPT builders. Profile-specific validation and storage policies do not redefine historical condition meanings.

`engine.py`, `data.py`, `pipeline.py`, `remote.py`, `runpod_api.py`, `supervisor.py` and `hub.py` at package root are compatibility aliases. Each shares its canonical implementation's module object, preserving established imports and patched globals. Add implementation to its subpackage rather than a second engine or launcher at the old path.

`completion_data.py` and `fresh_data.py` remain at root with their original bytes. The completion-bank identity includes its builder source hash; relocating that implementation without a versioned legacy reader would invalidate a pinned bank. `completion_mixing.py` and `mixing.py` remain distinct scientific algorithms: FineWeb replaces targets at fixed prompt rows; WikiText samples independent block pools. They share only pure count and token-hash utilities.

The existing `continued.py`/`validate.py` and `workflow/*validation.py` retain acceptance logic. There is no separate validation package yet. `stages.py` exposes local stages; `common.py` and `paths.py` provide shared small IO/workspace primitives. Training-dependency imports stay out of report and watchdog CLI startup.

The supported installation is an **editable repository checkout**. Configurations, scripts and comparison definitions are repository resources; a bare wheel is not a standalone deployment. Deployment archives enumerate package files and `pyproject.toml`; resolved configurations travel separately. Tests exercise nested bundle contents and legacy imports.

### Coordinators and manual adapters

Named coordinators preserve different scientific lineage/seal requirements. Shared helpers cover mechanics without merging the protocols. Root compatibility exports remain for existing callers; do not copy training engines into condition folders.

`scripts/operations/` contains explicitly manual and often date-specific adapters for historical authorizations and accepted bundles. Some still patch legacy globals; retain their recovery compatibility and restrictions. They are not templates for new workflows. Ordinary and staged high-fraction launches share the same admission lock; staged parallel startup releases admission after deployment before independent progress observation.

`workflow/mixed_campaign.py` coordinates three flat sibling jobs under one parent provider owner. Each child has its own scientific configuration and artifacts; the parent owns the Pod, volume, ledger session and watchdog. Completion seals pin backup commits and indexes before handoff. A resumed campaign preserves sealed prior-session children and binds active work to its current registered session; failure or pause prevents later jobs from starting.

## Scientific lineage

Every training stage distinguishes four objects:

1. **Initializer:** checkpoint and tensor hash from which training starts.
2. **Training data:** token dataset, file checksum and token-content hash.
3. **Generator:** selected checkpoint that produced those tokens.
4. **Output:** checkpoint selected by the configured policy, tensor hash and selected/executed epochs.

These identities can differ intentionally. In a shared-start reset branch, cycle 2 consumes D2 generated by S1 but starts training from S0. Historical `fresh_base` means original pretrained OPT; `reset_s0` means fixed real-data-trained S0. Mixed training separately records the full synthetic source pool and actual real/synthetic training targets. Preserve historical meanings.

Shared S0/S1 records retain their source run, original measurement environment, timings and cost provenance. A new run imports them using pinned Hub commits and checksums, then starts new work at cycle 2. Completion and branch manifests prevent cached files from being mistaken for completed training.

## Evidence, derived output and private state

```text
experiments/                         scientific definitions and configurations
comparisons/                         explicit comparison definitions and analyses
docs/                                methodology, operations, audits and history
model_collapse/                      shared runtime and CLI
tests/<responsibility>/             science, workflow, cloud, artifacts, analysis and repository checks
scripts/operations/                  explicitly manual and historical operational adapters
reports/<historical_run_id>/         published paths preserved verbatim
reports/<new_run_id>/records/        immutable evidence snapshots
reports/<new_run_id>/analysis/       frozen published derived analysis
reports/<run_id>/generated/<id>/     regenerated tables/figures and analysis manifest
reports/<run_id>/report.md           authored interpretation, protected from automatic overwrite
runs/<run_id>/                       ignored execution/recovery workspace
.state/                             ignored authoritative operational state and source archives
literature/                         tracked citations/notes; local papers/ ignored
docs/project/                       local proposal ignored; tracked location guide
```

Old artifact prefixes, CSV schemas and indexed bytes remain unchanged. Readers adapt historical schemas for comparisons rather than rewriting the evidence. An analysis records input hashes, source revision, execution environment, metric scope and output hashes. It must run from small recorded metrics without loading model weights when those metrics suffice.

Private control state is separate from fetched scientific evidence. A historical backup of `resources.json` must not replace the authoritative current run/launch/Pod/volume association. Ownership checks bind that complete tuple before stopping or deleting resources.

Recovery records distinguish **metadata-only**, **probe-verified** and **bulk-verified** scope. Having a local run folder or a verified artifact index does not establish that all model weights and datasets have been independently recovered. Existing private state, source archives, document-delivery receipts and reference documents are retained during reorganisation.

## Runtime identity and deployment

Scientific configuration, source bundle identity and execution environment are recorded separately. A changed runtime requires fresh acceptance; adding unrelated documentation or a new condition must not falsely certify changed runtime or silently rewrite historical hashes. Deployment uses explicit project paths and an enumerated source bundle; credentials and bulk research artifacts are excluded.

The independent budget watchdog must bootstrap without the training dependencies. Supervisor, worker and watchdog do not depend on the launcher remaining open. Compute and storage have separate billing intervals, and completed runs are rejected before any resume provisioning.

## Maintaining these boundaries

- Add named scientific settings beside their condition and register references in the catalogue. Keep complete resolved YAML snapshots; do not silently introduce inherited defaults into historical files.
- Keep reports offline. Renderers live in `analysis/`, which is included in analysis provenance; workflow wrappers are compatibility entry points.
- Tests use one network/credential/state isolation fixture. See the [test map](../tests/README.md) and [repository checker](../scripts/maintenance/README.md). Windows CI checks publication bytes with line-ending conversion enabled.
- Keep one current-state snapshot; append execution history to dated receipts. Stable documentation describes protocols instead of duplicating mutable run statuses.
- Existing runtime archives and acceptance remain immutable. Current runtime-changing cleanup needs a fresh separately authorized acceptance before new paid training. Do not deploy it into a running or historically pinned run.

See the [audit](audits/2026-10-27_repository_audit.md) for the baseline findings, [methodology](methodology.md) for scientific rules, and [operations](operations.md) for the actual lifecycle.
