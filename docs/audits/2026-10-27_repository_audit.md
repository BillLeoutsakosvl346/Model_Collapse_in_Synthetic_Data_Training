# Repository audit and organisation proposal

Audit date: 27 October 2026. Baseline commit: `a76d77d`.

This is an audit and implementation proposal, not a completed refactor or a new GPU launch. Five fresh, independent agent reviews covered experiment architecture, cloud infrastructure, results/analysis, documentation/tests, and the whole repository. The primary agent reconciled the findings, inspected the relevant implementation, inventoried the tracked tree, and ran the existing local tests. No cloud APIs, paid probes, bulk model downloads, credential contents, or remote resources were used for this audit.

## Decision

Keep the existing Python package, command interface, run IDs and published report paths. Add explicit experiment definitions and a catalogue around the existing implementation. Make report generation non-destructive and scientific lineage explicit before adding the next condition. Extract reusable code gradually, after its boundaries have tests; do not begin with a wholesale `src/` conversion or renaming of historical artifacts.

The repository has a useful small implementation, not an unmanageable amount of code. Its main problem is that protocol selection, operational state, evidence, analysis and publication do not yet have consistently separate responsibilities.

## Audit coverage and baseline checks

The tracked baseline contains **238 files / 6,786,935 working-tree bytes**:

| Area | Files | Review |
| --- | ---: | --- |
| Root | 8 | Instructions, installation, packaging, ignore/byte-preservation policy |
| `model_collapse/` | 14 | All package modules, scientific and cloud control paths |
| `configs/` | 4 | Both full and smoke configurations for both existing modes |
| `tests/` | 5 | Scientific, continuation, recovery, epoch-resume and cloud-safety tests |
| `scripts/` | 1 | Manual paid shutdown acceptance probe |
| `reports/` | 206 | All four run inventories, principal reports, analysis scripts and indexed public evidence |

Automated offline format checks passed for 23 Python files (AST), 169 JSON files, 4 YAML files, 6 CSV files and 1 JSONL file. The existing test suite passed: **26 passed, 3 dependency deprecation warnings, 27.75 seconds**. This is a local CPU/offline baseline, not GPU acceptance of any proposed changes.

The results review verified SHA-256 agreement for all 37 indexed Week 1 public records, 83 indexed Week 2 public records and 28 indexed continued-smoke public records. Some operational receipts deliberately fall outside their historical indexes. A full tracked-file snapshot with role classification and working-tree hashes is in [2026-10-27_tracked_inventory.json](2026-10-27_tracked_inventory.json). These counts refer to the baseline, before adding this audit.

Ignored `runs/` and `.state/` were reviewed as inventories and selected non-secret records, not as complete archives. They include operational state, retained source archives, recovery caches and document-delivery proofs. Secrets and bulk checkpoint contents were excluded. Existing cloud resource status was not refreshed during this audit.

## Findings to address before the next paid experiment

| Priority | Finding and evidence | Required action |
| --- | --- | --- |
| High | Generic `report RUN_ID` writes into the published directory and overwrites `report.md` and `metrics.csv`: [`__main__.py:74`](../../model_collapse/__main__.py#L74), [`report.py:33`](../../model_collapse/report.py#L33), [`report.py:114`](../../model_collapse/report.py#L114). For Week 2 it replaces authored prose and the expanded CSV with a different schema. | Generate into a separate destination by default. Preserve published reports and their original schemas. |
| High | Fresh-base validation stamps the current archive hash without comparing it with the source that actually ran on the smoke Pod: [`validate.py:95`](../../model_collapse/validate.py#L95). Full launch trusts that certificate at [`remote.py:199`](../../model_collapse/remote.py#L199). Continued validation already rejects a source mismatch. | Require tested-source identity in both validators; never certify changed code by re-stamping old acceptance. |
| High | Cleanup estimates GPU cost through deletion time even after compute stopped: [`remote.py:382`](../../model_collapse/remote.py#L382). Retained-volume accrual also remains open after removal: [`remote.py:55`](../../model_collapse/remote.py#L55). | Separate and close compute and storage billing intervals. Preserve settled historical values; distinguish estimates from observed account debits. |
| High | Resume has no early completed-run rejection before provisioning: [`remote.py:177`](../../model_collapse/remote.py#L177), [`remote.py:228`](../../model_collapse/remote.py#L228). | Reject completed runs before creating resources. A scientific branch must receive a new run ID, not resume Week 2. |
| High | Current policy permits only pretrained reset or previous-model continuation: [`pipeline.py:138`](../../model_collapse/pipeline.py#L138). Continued mode requires generator weights to equal training initializer: [`pipeline.py:177`](../../model_collapse/pipeline.py#L177), [`stages.py:35`](../../model_collapse/stages.py#L35). An explicit checkpoint is always labelled `continued`: [`engine.py:193`](../../model_collapse/engine.py#L193). | Add a fixed-checkpoint policy and verified branch import. Record generator and training initializer separately. Do not redefine historical `fresh_base`. |
| Medium | Existing `metrics.json` short-circuits remaining stage checks: [`pipeline.py:151`](../../model_collapse/pipeline.py#L151). Dataset lineage checks occur after new training at lines 172-178. | Verify inputs, stage outputs, hashes and completion status before training or reusing a stage, especially imported artifacts. |
| Medium | Fetch can overwrite current `resources.json` with a historical backup: [`hub.py:50`](../../model_collapse/hub.py#L50), [`hub.py:90`](../../model_collapse/hub.py#L90). Ownership checks do not bind the complete run/launch/Pod/volume tuple: [`remote.py:138`](../../model_collapse/remote.py#L138). | Separate authoritative local control state from fetched scientific evidence; validate full ownership associations before destructive actions. |
| Medium | Recovery fetches mutable Hub defaults without pinned revisions: [`hub.py:86`](../../model_collapse/hub.py#L86). Existing source archives are private local state, while resume messages mention a recorded Git commit not present in the run's basic source record. | Add a recovery receipt with resolved model/data repository commits and index hashes. Record source commit, source bundle and protocol identities separately. Keep old hashes unchanged. |
| Medium | `repetition_penalty` can differ in YAML while generation always executes `1.0`: [`common.py:64`](../../model_collapse/common.py#L64), [`engine.py:239`](../../model_collapse/engine.py#L239). Several length/beam/epoch assumptions are hardcoded. | Execute validated settings or reject unsupported settings. Keep protocol constraints explicit and versioned. |
| Medium | Root detection, config lookup, bundle creation and remote extraction depend on today's layout: [`common.py:9`](../../model_collapse/common.py#L9), [`remote.py:74`](../../model_collapse/remote.py#L74), [`remote.py:181`](../../model_collapse/remote.py#L181), [`remote.py:266`](../../model_collapse/remote.py#L266). | Introduce explicit paths and a tested bundle manifest before moving package/config files. Preserve the stdlib-only watchdog bootstrap. |
| Medium | Calendar weeks stand in for protocols, and instructions have stale execution text. `WEEK1_CLOUD_TASK.md:3` still says the continued full run needs authorization; old automatic-execution text remains elsewhere despite current AGENTS supersession. | Keep current operational authority short. Move historical task prompts into documented history with forwarding links and completion notices. |
| Medium | Byte preservation is hardcoded to three dated `records/` trees. Week 1's original cloud report has CRLF checkout bytes; LF normalization in memory reproduces its indexed cloud hash. | Generalize preservation by artifact role, test checkout bytes and Git blobs, and never silently normalize indexed evidence. This is a representation issue, not evidence of altered measurements. |

These are observed code paths and limitations. They do not imply that a listed failure occurred during the completed experiments, or invalidate their measured results. The audit did not exercise destructive operations.

Other improvements: directly declare runtime imports currently obtained transitively (`filelock`, `requests`, `cryptography`); document an isolated CPU installation separately from inheriting remote CUDA PyTorch; make an image digest/environment record part of future execution provenance. The shutdown probe in `scripts/check_shutdown_timeout.py` creates a paid Pod and must remain a clearly manual tool outside routine test commands.

## The identities we need

| Entity | Meaning | Example |
| --- | --- | --- |
| Experiment/condition | Named scientific rules, independent of calendar week | `pretrained_reset`, `continued_from_real_s0`, `reset_to_real_s0` |
| Comparison | Explicit arms, shared prefix, branch point and matched factors | `shared_s0_s1_reset_vs_continue` |
| Run | A concrete seeded realization with fixed resolved configuration | Existing timestamp run IDs, preserved verbatim |
| Launch | A provisioning or resumed execution of a run | Existing `launch_id`; do not introduce a competing synonym |
| Model/data stage | Separate initializer, training dataset, generator and selected checkpoint | Cycle 2 can use S0 weights with D2 generated by S1 |
| Analysis | A derivation from recorded inputs | Input hashes, analysis revision/environment and output hashes |

Use Week 1/Week 2 as presentation aliases. They must not determine behavior or identity. In particular, historical `fresh_base` means reset to untouched pretrained OPT; reset to real-data-trained S0 is a new condition.

## Recommended layout

This tree is the target organisation, not a claim that the folders and commands below have been implemented. Add the experiment catalogue and documentation first. Internal package subdivision is incremental, only where needed.

```text
README.md                         concise entrypoint, experiment list, common commands
AGENTS.md                         current scientific/resource rules and authority
pyproject.toml                    installation and dependency metadata

docs/
  architecture.md                 boundaries, record formats, run lifecycle
  methodology.md                  shared data/training/evaluation protocol
  metrics.md                      definitions, units, sample scope and limitations
  operations.md                   install, launch, status, recovery and spending
  history/                        old task prompts and setup instructions
  audits/                         this review and tracked-file inventory

experiments/
  catalog.yaml                    protocols, run IDs, status and result links
  opt125m_wikitext2/
    pretrained_reset/             existing Week 1 scientific definition
      README.md
      full.yaml
      smoke.yaml
    continued_from_real_s0/       existing Week 2 scientific definition
      README.md
      full.yaml
      smoke.yaml
    reset_to_real_s0/             proposed new arm
      README.md
      full.yaml
      smoke.yaml
      branch.yaml                 immutable source S0/S1/D2 references

comparisons/
  shared_s0_s1_reset_vs_continue/
    comparison.yaml               arms, shared prefix, matched factors, confounders
    README.md
    analyses/<analysis_id>/        derived tables, figures, interpretation, manifest

model_collapse/                   retain this package root and python -m interface
  __main__.py                     stable entrypoint
  common.py                       compatibility while shared helpers are extracted
  science/                        data, model IO, training, generation, evaluation, metrics
  workflow/                       policies, lineage, orchestration and local stages
  cloud/                          provider, provisioning, SSH, budget, registry, supervision
  artifacts/                      manifests, Hub transport, imports, recovery
  analysis/                       reusable tables, figures, audits and comparison readers
  validation/                     local invariants and explicit GPU acceptance

configs/                          keep existing files during compatibility migration
tests/                            retain existing tests; add focused regression tests
scripts/operations/               clearly manual operational probes

reports/<existing_run_id>/        preserve every published path
  records/                        existing immutable evidence
  report.md                       existing authored interpretation
  ...                             existing exports, figures and reproduction scripts
reports/<new_run_id>/
  records/                        immutable evidence snapshots
  generated/<analysis_id>/         regenerated tables/figures and analysis manifest
  report.md                       authored publication, never overwritten by default

runs/<run_id>/                    ignored working/recovery data; completeness recorded
.state/                          ignored private operational state; no casual moves
references/                      optional ignored local papers/proposal, migrated carefully
```

Do not create a new training engine for each experiment. The new folders describe scientific conditions and reuse the same train/generate/evaluate functions. Keep old YAMLs at their paths as versioned historical definitions until explicit configuration lookup and bundling support the new locations.

There is no immediate need for a database, tracking service, workflow framework or `src/` conversion. A versioned YAML catalogue, immutable JSON records and CSV analysis tables fit the current scale.

## Where the current implementation belongs

| Current files | Responsibility after extraction |
| --- | --- |
| `__main__.py` | Stable CLI dispatcher; optional later `cli.py` extraction |
| `common.py` | Atomic files/hashes, schema validation and explicit path helpers; keep compatibility imports |
| `data.py` | `science/data.py`; preserve exact tokenizer/packing behavior |
| `engine.py` | Shared model IO, training, generation, evaluation and diversity modules in `science/` |
| `pipeline.py`, `stages.py` | `workflow/`; initialization policy and data/model lineage belong here |
| `continued.py` | Split pinned checkpoint import into `artifacts/` and acceptance checks into `validation/` |
| `remote.py` | Gradually split provisioning/SSH, budget ledger and resource registry into `cloud/` |
| `runpod_api.py`, `supervisor.py` | `cloud/`; retain independent shutdown and bootstrap behavior |
| `hub.py` | `artifacts/`; revision-pinned recovery separated from current operational registry |
| `report.py` | `analysis/`; plotting must not require cloud pricing or import the worker |
| `validate.py` | `validation/`; acceptance bound to the actual tested source and protocol |
| Three report-local reproduction scripts | Keep old paths as historical scripts or compatibility wrappers; extract reusable logic for new analyses |
| `scripts/check_shutdown_timeout.py` | Manual operational acceptance tool, documented as paid |

## Evidence, cache and publication rules

1. Preserve existing run IDs, relative artifact paths, source archives, checkpoint/token hashes, pinned initial checkpoint references, and report URLs. Do not rename remote artifact prefixes to match a nicer directory tree.
2. Raw evidence, copied cloud exports, generated analysis and authored interpretation have different roles. Preserve original schemas and introduce readers/adapters for normalized cross-run tables rather than rewriting old CSVs.
3. Every future analysis records its source inputs and hashes, code revision, environment, metric definitions/scope and output hashes. A plot should be reproducible without calling Runpod or loading model weights when recorded metrics suffice.
4. Record recovery scope: `metadata_only`, `probe_verified`, or `bulk_verified`. A local run directory is not proof that all checkpoint/data files have been recovered.
5. Bulk models and token arrays remain in private Hugging Face backups/network storage, outside ordinary Git. Credentials and the user's proposal are never included in source bundles or public evidence. Moving local references requires updating ignore rules before the move.
6. Operational control state is private and durable. Separate it conceptually from disposable caches and document-render intermediates; preserve delivery receipts and source documents before removing render caches. No cleanup is part of this audit.
7. Preserve scientific distinctions: corpus PPL versus mean passage PPL; passage SD versus seed uncertainty; empirical token entropy versus predictive entropy; frozen-reference NLL versus KL; 256-output probe versus full training dataset; executed epochs versus inherited selected epochs.

## Specification for the next shared-start comparison

The common prefix is the existing continued run `20261026T192740Z-full-continued`:

| Shared artifact | Recorded content hash |
| --- | --- |
| Real-data S0 weights | `dd12f1e10e953a352299108fc013fa8c5b6d2afcd99addfc41e8df5ddc586b10` |
| First synthetic-trained S1 weights | `5cbccf4740806deeb6d972426309982ae9713651fa1b41602d81cb0ff421f91f` |
| D2 continuation token content | `4ed3a1b44b031cad1e2a6ba72cb368c0e2efb7a848237e262ce744a3de255e42` |

Pin both model and dataset repository commits and serialized-file checksums when resolving the import. Verify them before renting a GPU. The recorded shared evaluations remain references to the original measurement event; identical weights alone do not justify merging measurements taken on different hardware.

```text
Shared: S0 evaluated -> generates D1 -> train S0 on D1 -> S1 evaluated -> generates D2

Existing continued arm: train S1 on D2 -> B2 -> generate DB3 -> train B2 on DB3 -> ...
New reset arm:         train S0 on D2 -> A2 -> generate DA3 -> train S0 on DA3 -> ...
```

Both curves reference the identical saved S0 and S1, so they share the first jump by construction. The first new reset training is cycle 2; nine new training stages reach cycle 10. Import common stages with explicit provenance, not invented training durations or duplicate synthetic datasets. The first new reset stage has different generator and initializer identities intentionally.

Hold fixed the real split arrays, prompts, model/tokenizer revisions, data size, decoding, optimizer restart, scheduler, batch sizes, seed, precision, validation selection and metric definitions. Match the existing Blackwell GPU/software where feasible. Record any execution mismatch; if a clean match is unavailable, a freshly executed paired study sharing one prefix would be stronger than claiming a perfectly controlled comparison with the old arm.

This is a new scientific condition; it neither replaces Week 1 nor overwrites Week 2. A full implementation needs a new initializer policy, branch-import records and tests. The audit has not implemented those or started a paid smoke/full run.

## Migration and verification sequence

### Phase 1: navigation and historical preservation

- Add a catalogue linking all completed runs to explicit scientific conditions and recovery status; list the new condition as planned.
- Shorten README to installation, experiment index and safe operations; extract shared methodology and metric definitions.
- Archive outdated automatic-execution prompts with forwarding notices at their old paths. Keep current AGENTS instructions authoritative and preserve published report links.
- Preserve the audit's baseline hashes and ignored operational/source state before any moves. Generalize evidence byte-preservation rules after checking both Git blobs and Windows working-tree bytes.

### Phase 2: narrow correctness fixes

- Protect published reports from default overwrite; add explicit output destinations and an analysis manifest.
- Correct source-bound acceptance, completed-run rejection, cost-interval accounting, control/evidence separation and complete resource-ownership checks.
- Verify generation settings against resolved configuration and verify stage inputs/completion before reuse.
- Add offline regression tests that reproduce these concrete failure modes; do not create tests that merely assert a folder exists.

### Phase 3: explicit protocol and branch support

- Separate initial state, per-cycle weight initialization, dataset-generator identity and optional branch origin.
- Add a fixed-S0 initializer and verified S0/S1/D2 import using the existing training primitive.
- Record immutable source repository commits, file/content hashes and lineage; use a new run ID.
- Test a tiny local shared-prefix fork: identical S0/S1 and D2, reset arm initialized from S0, continued arm initialized from S1, same evaluation arrays, correct output sizes, incompatible imports rejected and completed stages unchanged after resume.

### Phase 4: incremental module/config extraction

- Introduce explicit paths and a source-bundle manifest first. Keep the CLI and old imports/commands compatible during extraction.
- Move one responsibility at a time, running its meaningful tests, installation/CLI checks and bundle checks. Do not rewrite old manifests or pretend old GPU acceptance covers changed source.
- Parameterize analysis by catalogue/comparison inputs. Preserve old reproduction commands, generate into scratch output and verify numerical agreement and source hash preservation.

### Phase 5: prepare and validate the new run

- Resolve exact artifacts and compatible runtime, check the persistent task ledger and storage reserve, and estimate only the remaining work.
- Perform the small GPU acceptance appropriate to the changed workflow before a long paid launch. A folder cleanup alone does not justify GPU work, and no paid action occurred in this audit.
- Confirm detached worker/watchdog, recoverable artifacts, exact branch handoff and shutdown. Only then describe the new run as ready or running.

The next implementation should begin with the catalogue, report protection, source acceptance and ledger fixes, then the shared-prefix branch. Broad package moves can wait until those boundaries are established. This order improves scientific clarity and operational reliability without putting completed evidence at risk.
