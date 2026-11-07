# Project instructions

Cambridge Engineering IIB research on recursive synthetic-data training with OPT-125M. Reproduce explicitly named scientific conditions; novel mitigation requires a separate request.

## Start here

- [Current state and authorization](docs/current-state.md): one dated snapshot, spending limits and recovery obligations. Recorded status is not a live check.
- [Architecture](docs/architecture.md): module responsibilities and compatibility paths.
- [Catalogue](experiments/catalog.yaml) and [reports](reports/README.md): conditions, runs and evidence.
- [Methodology](docs/methodology.md), [metrics](docs/metrics.md), [operations](docs/operations.md): scientific and operational contracts.
- [Preserved instructions](docs/history/2026-11-07_pre-cleanup-agent-instructions.txt): exact earlier chronology. Historical forecasts and superseded statuses grant no new authority.

## Scientific rules

- Reuse the shared training engine and pinned configuration. Preserve seed, packing, prompts, evaluation arrays, decoding and condition meanings; prevent held-out leakage.
- Distinguish original-pretrained reset, fixed real-trained S0 reset and continued weights. Record generator, initializer, training targets, optimizer parent and selected-checkpoint identities separately.
- Record executed epochs, selected epoch and selection policy. WikiText restarts optimizer/scheduler each cycle; FineWeb fresh-real/completion arms carry optimizer, scheduler and training RNG. Named configurations define each protocol.
- Distinguish full synthetic pools from actual mixed targets. FineWeb mixtures replace continuations at their original prompt rows using that arm's preceding checkpoint.
- Use token-weighted corpus NLL/perplexity. Distinguish pooled diversity, within-output repetition and formatting sensitivity. Frozen-S0 scores describe predictability of outputs, not model distance or independent quality.
- One seed gives descriptive trajectories. Never invent measurements, significance, completion or recovery coverage. Preserve unexpected results and separate observations from explanations.

## Evidence and private state

- Preserve published paths, raw records, historical schemas and indexed bytes. New analyses use new destinations; never overwrite authored reports automatically.
- Credentials, local PDFs/proposal, checkpoints and bulk arrays stay out of Git. Local research lives in ignored `literature/papers/` and `docs/project/`.
- `.state/registry/` and `.state/ledger.json` are authoritative; `runs/` holds recovery workspaces. Housekeeping must not move, delete or replace private state. Fetched historical controls cannot replace current ownership.
- Preserve historical recovery storage until adequate recovery is verified. Metadata/probe checks do not establish full checkpoint/dataset restoration; stopped storage continues billing.
- Preserve source archives, certificates and exact legacy data-contract algorithms. Compatibility aliases keep old imports working; canonical implementation belongs in the documented subpackage.

## Cloud boundaries

- Act only on task-owned resources, verifying the full run/launch/Pod/volume association. Default to one GPU; exceptions require their exact recorded authorization. The three named high-fraction jobs have an explicit parallel exception in current state.
- Do not duplicate active/completed runs or resume completed compute. Recheck stock, pricing, credentials, budget, private recovery and independent shutdown before separately authorized paid work.
- Never purchase credits, enable automatic/backup payments or use paid Hugging Face compute. Enforce recorded cumulative, allocation, reserve and setup limits together.
- Acceptance binds the actual runtime and scientific protocol. Changed execution code needs fresh GPU acceptance before paid training; old certificates remain historical evidence. A cleanup does not itself authorize cloud validation or launch.
- Offline tests/reporting must not provision resources. Keep the watchdog stdlib-only and independent of SSH, laptop, browser and agent lifetimes.

Keep this file stable and concise. Put dated execution chronology in receipts/history and refresh the single current-state snapshot.
