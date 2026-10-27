# Organise the repository and prepare the shared-start experiment

Accepted implementation plan, 27 October 2026. See the [repository audit](../audits/2026-10-27_repository_audit.md) for findings and the module responsibility map.

## Outcome and limits

Organise scientific conditions, shared execution infrastructure, immutable evidence and analysis. Implement and GPU-validate a new reset-to-real-S0 branch using the saved Week 2 shared beginning. Preserve all previous results, published paths and artifact hashes. Deliver tested code and a launch command; **do not start the full experiment**.

Maintain the cumulative USD 25 Runpod ceiling, existing setup allowance and USD 2 shutdown/storage reserve. New GPU acceptance is capped at USD 0.75 within the remaining allowances. Never purchase credits or enable automatic payments.

## Organisation and interfaces

- Add an experiment catalogue and named pretrained-reset, continued-from-real-S0 and reset-to-real-S0 definitions with methodology and full/smoke configurations. Weeks remain historical aliases.
- Add a comparison definition identifying the arms, exact shared prefix, matched settings, confounders and derived analyses.
- Keep the root-level `model_collapse` package and its command interface. Gradually separate science, workflow/lineage, cloud controls, artifacts, analysis and validation, retaining compatibility imports and one training engine.
- Provide concise entrypoint documentation, shared methodology, metric definitions, operations and historical instructions with forwarding notices.
- Preserve existing dated report paths. New raw records, generated analysis and authored publication have separate destinations. Keep ignored operational state, source archives, credentials and recovery caches in place.
- Add `smoke --config PATH`, `launch --config PATH --generations 10` and `report RUN_ID --out PATH`. Configuration and legacy mode selection are mutually exclusive. Existing commands retain their historical meaning. Reporting defaults to a generated-analysis directory and cannot overwrite published evidence or interpretation.
- Analysis records input hashes, revision, environment and output hashes. Adapt old CSV schemas through readers; never rewrite historical data to normalize it.

## Correctness fixes

- Bind smoke acceptance to the source that actually ran; reject stale acceptance after runtime changes.
- Separate and close compute/storage billing intervals, avoid duplicate resumed-launch charging, and preserve historical receipts and estimates versus account debits.
- Reject completed-run resume before provisioning. Branches get new run IDs.
- Separate current resource registry from fetched historical evidence; validate run/launch/Pod/volume ownership before destructive actions.
- Pin backup repository commits, index hashes and object checksums without rewriting old indexes.
- Execute validated generation settings or reject unsupported values; metadata must match execution.
- Verify inputs, lineage and completed-stage outputs before training or reuse.
- Introduce explicit project paths and a tested source bundle before module moves. Preserve the independent stdlib-only watchdog bootstrap. Separate runtime identity from protocol configuration.

## Shared-start branch

Import S0, S1, exact D2 and shared evaluation records from `20261026T192740Z-full-continued`, pinned by repository revisions and file/content hashes before provisioning.

```text
Shared: S0 -> D1 -> train S0 on D1 -> S1 -> D2
Existing continued: train S1 on D2 -> B2 -> DB3 -> train B2 on DB3 -> ...
New reset:         train S0 on D2 -> A2 -> DA3 -> train S0 on DA3 -> ...
```

Add an explicit fixed-S0 initialization policy. Do not redefine historical `fresh_base`. Record training initializer, training dataset, data generator, selected checkpoint, selected epoch and branch origin separately. First new training is cycle 2, with nine new training stages through cycle 10. Imported stages retain their original costs/timings and measurement identities.

Preserve Week 2 packing, fixed prompts/evaluation, token budgets, decoding, optimizer restart, scheduler, batch sizes, seed, precision and validation selection. Target RTX PRO 4000 Blackwell with the recorded software. If matching execution is unavailable, leave the full run pending rather than silently claim matched hardware.

## Verification and publication

1. Publish this plan, the audit and inventory to `origin/main` first.
2. Retain the 26-test baseline. Add narrow regressions for the audited operational failures, CLI compatibility, source bundle contents, byte preservation and non-destructive analysis.
3. Test a tiny local common-prefix fork, wrong-artifact rejection, exact model/data handoffs, fixed evaluation, interrupted-work recovery and unchanged shared stages.
4. Regenerate figures/tables into scratch output; verify original evidence and publication bytes are unchanged.
5. After offline checks pass, execute a small actual reset-branch GPU smoke within the limits. Verify checkpoint import, real weight updates, lineage, recovery, private-backup restore, detached supervision and automatic shutdown. Fetch the small report and confirm no task-owned GPU remains running.
6. Publish implementation in focused verified commits, staging explicit files and excluding credentials, private state, bulk artifacts and local reference documents.
7. Deliver exact validation results, spend, artifact locations, measured full-run estimate, limitations and tested launch command. Leave the full experiment unlaunched.
