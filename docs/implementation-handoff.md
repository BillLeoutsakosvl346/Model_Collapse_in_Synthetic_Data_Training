# Shared-start experiment: implementation handoff

**Subsequent full run:** after this setup handoff, the user explicitly authorized `20261027T211852Z-full-reset-s0`. It launched at 21:18 UTC/London on 27 October and completed at 02:56 on 28 October, with actual GPU shutdown verified. [Completion record](validation/2026-10-28_reset-full-completion.json) · [Historical launch record](validation/2026-10-27_reset-full-launch.json). The text below records the earlier setup-completion state. Do not repeat its launch command for the completed run.

27 October 2026. **Setup complete; the full reset experiment is unlaunched.** Historical Week 1/2 reports, configurations, source archives and private state were preserved. The [accepted plan](plans/repository-and-shared-start.md) and [audit](audits/2026-10-27_repository_audit.md) describe the scope.

## Repository and runtime

Named conditions and their full/smoke profiles live in [experiments](../experiments/catalog.yaml); shared artifacts and matching requirements live in [comparisons](../comparisons/shared_s0_s1_reset_vs_continue/comparison.yaml). Shared computation, workflow, cloud budgeting, artifact paths and analysis have reusable package boundaries, with existing imports and CLI meanings retained. The root package remains in place. [Architecture](architecture.md) maps the files; [methodology](methodology.md), [metrics](metrics.md) and [operations](operations.md) cover the protocol and execution.

Focused fixes cover source-bound acceptance, separate compute/storage accounting, completed-run resume rejection, authoritative resource ownership, pinned recovery, decoding fidelity, stage integrity, immutable analysis and source-bundle deployment. Historical CSVs are normalized by readers. Runtime identity is separate from scientific configuration.

## What was verified

- **86 offline tests passed**, retaining the original 26-test suite and adding regressions for the audit findings, tiny shared-prefix forks, resume, reporting protection, CLI compatibility and launch admission. The paid launcher is mocked in offline tests.
- Editable installation, package/source-bundle contents and documentation links were checked. All **210 protected historical report/configuration files** matched the audit's working-tree hashes. Publication manifests and staged Git blobs were checked for byte preservation on Windows.
- Final GPU acceptance [`20261027T203952Z-smoke-reset-s0`](../reports/20261027T203952Z-smoke-reset-s0/report.md) passed **14 checks** on RTX PRO 4000 Blackwell with PyTorch 2.8.0+cu128. It imported pinned S0/S1/D2, trained two new cycles from S0, recovered after two injected worker exits, reused a partial generation shard and preserved the completed stage.
- Private backup recovery checked **129 files / 2,019,179,284 bytes remotely**, then reloaded/evaluated the final checkpoint. Direct shifted-NLL calculation discrepancy was **0**. Locally, **105 metadata files / 14,873,728 bytes** were recovered; model weights were not downloaded to the laptop.
- Independent supervision continued without a persistent SSH session. The independent guard stopped actual compute; provider read-back confirmed `EXITED`. Both acceptance attempts' temporary Pods and volumes were deleted after recovery checks. The two historical full-run Pods remain `EXITED` and their recovery volumes remain intact.

Accepted runtime SHA-256: `5d56bf37476583b20fd9d74c40941b322af6bcb2720a48a7a463f9aca6dde49e`; source commit `3d9e35898a4823407eeda8683033f041f6f56bb9`. Earlier-source acceptance [`20261027T202525Z-smoke-reset-s0`](../reports/20261027T202525Z-smoke-reset-s0/report.md) is preserved and explicitly superseded. Later documentation/evidence commits do not alter this runtime identity. Changed runtime requires a new acceptance.

Smoke results validate execution only: new stages use 256 training blocks, 64 validation blocks and 64 test blocks for one epoch. Imported S0/S1 metrics retain their original full-data scope, so their plotted values are not directly comparable to tiny-smoke evaluations. No collapse conclusion follows from these test points.

## Costs and forecast

Across both acceptance attempts, the observed account balance changed by **USD 0.1421921924** (23.0240973694 to 22.8819051770 through the recorded check). This includes concurrent historical storage and billing timing; it is not an itemized compute invoice. The final attempt's earlier cleanup snapshot alone recorded USD 0.0723899962. Do not sum overlapping/lagging snapshots as a precise bill. [Budget/provider verification](validation/2026-10-27_shared-start-budget.json).

The conservative cumulative ledger estimate was **USD 8.03**, with setup estimated at **USD 0.43**, below the cumulative USD 25, setup USD 3 and new acceptance USD 0.75 limits. The USD 2 reserve remains protected. No credits or automatic payments were purchased/enabled. Ongoing historical storage was USD 0.006/hour at the recorded check; no GPU compute was running.

Warmed production-batch throughput was about **34,530 training target tokens/s** and **1,304 generated tokens/s** with five beams. The remaining full reset branch is estimated at **8.26 hours / USD 4.75**, including a 1.5x compute margin and setup/checkpoint/upload allowances. This counts **nine new training stages and eight full generation stages plus the final probe**, excluding imported S0/S1 work. Storage retained after completion and prior task spend are separate. Live price/stock, contention and upload time can change this estimate.

## Launch later

From the project directory with its local environment active:

```powershell
python -m model_collapse launch --config experiments/opt125m_wikitext2/reset_to_real_s0/full.yaml --generations 10
```

This command is prepared and its admission path tested; it has **not** been executed for a full run. It creates a new run ID, imports the exact Week 2 S0/S1/D2, trains S0 on D2 at cycle 2, then repeatedly trains fixed S0 on the preceding selected model's new continuations through cycle 10. Shared evaluations remain the same saved measurement events. Each new cycle has five epochs and fresh optimizer/scheduler state; validation selects the checkpoint.

The launcher requires current-source acceptance, pinned/checksummed imports, sufficient remaining budget and matching RTX PRO 4000 Blackwell/software. Unavailable matching hardware leaves launch pending. The laptop can close **after** the launcher returns `detached_supervisor_and_watchdog_verified: true`.

```powershell
python -m model_collapse status RUN_ID
python -m model_collapse fetch RUN_ID --metadata-only
python -m model_collapse report RUN_ID --out reports/RUN_ID/generated/review
```

Reports and small evidence are in Git; private Hugging Face backups retain bulk checkpoints and datasets under each run ID. A full fetch downloads several GB; metadata-only fetch does not. Do not delete historical recovery volumes during this refactor. A completed run cannot be resumed into a new experiment.

The future comparison is one seeded pair with a historical continued arm. Matching settings and hardware do not establish statistical significance or guarantee bitwise cross-launch determinism. The baseline pretrained-reset condition remains distinct from this real-trained-S0 reset.
