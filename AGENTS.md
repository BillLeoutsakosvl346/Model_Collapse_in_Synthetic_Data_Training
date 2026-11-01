# Project instructions

Cambridge Engineering IIB research on recursive synthetic-data training with OPT-125M and WikiText-2. Reproduce and compare explicitly named scientific conditions; novel mitigation methods require a separate request.

## Start here

- [Current state](docs/current-state.md): latest recorded run, authorization and outstanding recovery work. A recorded launch is not a live status check.
- [Experiment catalogue](experiments/catalog.yaml) and [reports index](reports/README.md): conditions, run IDs, results and verification scope.
- [Methodology](docs/methodology.md), [metrics](docs/metrics.md) and [operations](docs/operations.md): scientific rules and supported commands.
- [Historical agent chronology](docs/history/2026-10-31_agent-context.md): the complete previous instructions, preserved byte-for-byte. Historical prompts, forecasts and superseded statuses do not grant new execution authority.

## Current authorization

The authorized ten-cycle 50/50 continued-training run `20261031T163428Z-full-continued` completed at 22:51:45 UTC/London on 31 October 2026. All fifty epochs retained epoch 5; provider shutdown was verified at 22:55:32, reason `complete`. Its [analysis and Google Doc delivery](docs/validation/2026-11-01_mixed-50-analysis-and-doc.json) compares it with the final-epoch-5 synthetic-only continued arm: final test PPL 62.71 versus 107.37, with mixed diversity results and residual formatting failures. All metadata and ten new probes were recovered; bulk recovery/re-evaluation remains separate. Preserve its recovery volume. Do not provision a duplicate or resume completed compute.

Apply the explicitly authorized **USD 30 cumulative Runpod ceiling**, including setup, failed attempts and storage, with **USD 2 reserved** for storage/shutdown and no fixed training time limit beyond the budget guard. Preserve the earlier USD 25 receipts as historical evidence. Use existing credit only: never purchase credits, enable automatic/backup payments or use paid Hugging Face compute. New paid experiments need explicit user authorization; smoke validation is also paid. See the [authorization receipt](docs/validation/2026-10-31_mixed-50-budget-authorization.json).

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
- GPU acceptance must bind the runtime and scientific protocol that actually ran. The latest 50/50 acceptance covers runtime `266afa984a2a51b63f850390232a12b3ec98a93975ad27f9401983d1afc060c5`. Runtime changes require fresh GPU acceptance; documentation, standalone report tools and tests do not change that bundle.
- Offline tests/reporting must never provision cloud resources. Preserve the watchdog's independent startup without training dependencies. Budget protection must survive SSH, browser, laptop and agent disconnection.

Record material changes and checks in the appropriate dated receipt/report. Keep this file concise; put new execution chronology in the catalogue/current-state record and historical evidence rather than appending another long narrative here.
