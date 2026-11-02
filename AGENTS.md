# Project instructions

Cambridge Engineering IIB research on recursive synthetic-data training with OPT-125M and WikiText-2. Reproduce and compare explicitly named scientific conditions; novel mitigation methods require a separate request.

## Start here

- [Current state](docs/current-state.md): latest recorded run, authorization and outstanding recovery work. A recorded launch is not a live status check.
- [Experiment catalogue](experiments/catalog.yaml) and [reports index](reports/README.md): conditions, run IDs, results and verification scope.
- [Methodology](docs/methodology.md), [metrics](docs/metrics.md) and [operations](docs/operations.md): scientific rules and supported commands.
- [Historical agent chronology](docs/history/2026-10-31_agent-context.md): the complete previous instructions, preserved byte-for-byte. Historical prompts, forecasts and superseded statuses do not grant new execution authority.

## Current authorization

The real-only control **`20261102T154833Z-full-continued` is running**, provider-verified on 2 November after explicit authorization. Pod/disk `0kj0h4jzxtkrbi` use the accepted RTX PRO 4500/persistent-Pod profile; two new finite full-data epochs and detached supervision/watchdog passed. See [launch receipt](docs/validation/2026-11-02_real-only-full-launch.json) and current state. Do not duplicate or resume this active compute; preserve its disk for recovery. The latest cumulative ceiling is USD 50 with USD 2 reserved.

On 2 November the user explicitly authorized the real-only continued control (ten cycles, five epochs each, retaining epoch 5) and raised the cumulative ceiling to **USD 50**, using existing credit only. The **USD 2 reserve**, total setup allowance and validation caps remain. See the [authorization receipt](docs/validation/2026-11-02_real-only-budget-authorization.json). This supersedes the earlier USD 45 ceiling; The [real-only GPU acceptance](docs/validation/2026-11-02_real-only-gpu-acceptance.json) now passed on RTX PRO 4500 Blackwell Server Edition with explicit persistent Pod storage. Its temporary resources were recovered and removed; full-launch status belongs in current state. Accepted standalone runtime is `08586783f0ff72e5b52e76e3bcd2022832637721a85d6a891456664f392da56d`, protocol `0ffc95fa20209096b3e52003d55b25f5c522989025b77aba1377d0c3ff5298da`. Preserve historical ledger snapshots, receipts and recovery volumes.

The authorized ten-cycle 50/50 continued-training run `20261031T163428Z-full-continued` completed at 22:51:45 UTC/London on 31 October 2026. All fifty epochs retained epoch 5; provider shutdown was verified at 22:55:32, reason `complete`. Its [analysis and Google Doc delivery](docs/validation/2026-11-01_mixed-50-analysis-and-doc.json) compares it with the final-epoch-5 synthetic-only continued arm: final test PPL 62.71 versus 107.37, with mixed diversity results and residual formatting failures. All metadata and ten new probes were recovered; bulk recovery/re-evaluation remains separate. Preserve its recovery volume. Do not provision a duplicate or resume completed compute.

The user authorized implementation and small GPU validation of the sequential mixed-data campaign on 1 November: 50/50 reset, then 20/80 continued, then 20/80 reset. **Setup is complete; the subsequent full launch is explicitly authorized below.** Apply the selected **USD 45 cumulative ceiling**, including setup, failed attempts and storage, with **USD 2 reserved** and **at most USD 0.75 across all new validation attempts** within the existing USD 3 setup allowance. Preserve earlier USD 25/30 receipts and original ledger bytes as historical evidence. Use existing credit only: never purchase credits, enable automatic/backup payments or use paid Hugging Face compute. The setup authorization excluded full compute; the subsequent explicit launch instruction is recorded below. See the [current authorization](docs/validation/2026-11-01_mixed-campaign-budget-authorization.json).

Campaign setup is complete: `20261101T022210Z-smoke-mixed-campaign` passed 345 offline tests, independent review and all ten parent GPU/provider checks, executing six new stages / thirty epochs with epoch 5 retained. Both automatic handoffs, immutable mixtures, worker/coordinator recovery, three private checkpoint restores/re-evaluations and actual watchdog shutdown passed. Final and preliminary temporary resources were removed after recovery; historical volumes remain. [Setup receipt](docs/validation/2026-11-01_mixed-campaign-setup.json), [handoff](docs/mixed-campaign-handoff.md). Full training was unlaunched at that setup check. As of that receipt, the 24.09-hour / USD 13.93 running-cost forecast and padded watchdog coverage fit the authorized ceiling; later launches must recheck stock, credit and accrued storage.

Full campaign **`20261101T030306Z-full-mixed-campaign` launched at 03:03 UTC/London on 1 November**, after the user explicitly requested it. Pod `faocwszff0eaq6`, 60 GB volume `11hvbmunz5`, matching RTX PRO 4000 Blackwell in EU-RO-1 at USD 0.57/hour. Accepted runtime is unchanged; detached supervision, independent watchdog and two finite new cycle-2 epochs passed startup checks. This supersedes the setup-only unlaunched state: **do not launch a duplicate or resume active compute**. [Launch receipt](docs/validation/2026-11-01_mixed-campaign-full-launch.json). Live GPU benchmark forecasts 26.40 hours / USD 15.27 conservatively, slightly beyond the funded 25.64-hour watchdog window; it may pause if slow. The USD 45 ceiling/USD 2 reserve remain unchanged; deadline is 2 November 04:41 UTC. No completion claim.

## Scientific and reporting rules

- Reuse the shared training engine and pinned configuration. Keep seed, packing, prompt/evaluation arrays and decoding consistent; avoid validation/test leakage.
- Distinguish original pretrained reset, fixed real-trained S0 reset and continued weights. Do not relabel historical conditions or silently change their meaning.
- Record executed epochs, selected epoch and selection policy. Historical runs executed five epochs but selected epoch 1; final-epoch conditions retain epoch 5. Optimizer/scheduler restart each cycle.
- Distinguish a full synthetic source pool from the mixed dataset actually used for training. Preserve generator, initializer, data and selected-checkpoint hashes separately.
- Use token-weighted corpus NLL/perplexity. Keep pooled diversity, within-output repetition and formatting sensitivity separate. Frozen-S0 scores measure predictability of generated text, not model distance or independent quality.
- One seed gives descriptive trajectories. Never invent measurements, significance, completion or recovery coverage. Preserve unexpected/non-monotonic results and distinguish observation from explanation.

## Evidence, private state and cloud work

- Preserve published report paths, raw records, historical schemas and indexed bytes. Regenerate into new output directories; never overwrite authored reports or evidence automatically.
- Credentials, local PDFs/proposal, checkpoints and bulk arrays stay out of Git. Local research is in ignored `literature/papers/` and `docs/project/*.docx`.
- `.state/registry/` and `.state/ledger.json` are authoritative operational records; `runs/` contains recovery workspaces. Do not delete, move or replace private state as repository housekeeping. Fetching historical evidence must not replace current resource ownership.
- Modify only task-owned cloud resources, checking the full run/launch/Pod/volume association. Provision at most one GPU Pod at a time. Verify live pricing, matching hardware/software, credentials, private recovery and independent shutdown before paid work.
- Keep all historical recovery volumes until adequate recovery is verified. Metadata/probe verification does not establish full checkpoint/dataset restoration. Retained storage continues to bill after GPU shutdown.
- GPU acceptance must bind the runtime and scientific protocol that actually ran. The sequential campaign accepts runtime `35c89c3b9fb78c7967c4f62f07972739eabbe57b353a0360749eba71d1af2171` and protocol `770a627c02821b1301f3194268e65fb9635eccdc6a43e639477d886f26dac377`. Historical 50/50 acceptance covers its recorded `266afa984a2a51b63f850390232a12b3ec98a93975ad27f9401983d1afc060c5` runtime. Do not interchange workflow certificates. Runtime changes require fresh GPU acceptance; documentation, standalone report tools and tests do not change that bundle.
- Offline tests/reporting must never provision cloud resources. Preserve the watchdog's independent startup without training dependencies. Budget protection must survive SSH, browser, laptop and agent disconnection.

Record material changes and checks in the appropriate dated receipt/report. Keep this file concise; put new execution chronology in the catalogue/current-state record and historical evidence rather than appending another long narrative here.
