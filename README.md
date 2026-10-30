# Model collapse in synthetic-data training

Cambridge Engineering IIB: reproducible OPT-125M / WikiText-2 experiments on recursive synthetic-data training. The project follows [Shumailov et al. (2024)](https://doi.org/10.1038/s41586-024-07566-y), with explicit reconstruction choices and separately named weight-initialisation conditions. Novel mitigation is out of scope.

## Experiments and results

| Condition | Training initialisation after real-data baseline | State |
| --- | --- | --- |
| [Pretrained reset](experiments/opt125m_wikitext2/pretrained_reset/README.md) | Original pretrained OPT at every synthetic cycle | [Completed Week 1](reports/20261020T211121Z-full/report.md) |
| [Continued from real S0](experiments/opt125m_wikitext2/continued_from_real_s0/README.md) | Previous selected checkpoint | [Completed Week 2](reports/20261026T192740Z-full-continued/report.md) |
| [Reset to real S0](experiments/opt125m_wikitext2/reset_to_real_s0/README.md) | Fixed real-data-trained S0, after an imported shared first cycle | [Completed and analysed](reports/20261027T211852Z-full-reset-s0/report.md) |
| [Matched final epoch 5](comparisons/shared_start_final_epoch5/README.md) | New shared S1, then continued weights versus fixed S0 | [GPU validated; full pair unlaunched](reports/20261030T172428Z-smoke-paired-final5/report.md) |

The [experiment catalogue](experiments/catalog.yaml) records run IDs and recovery scope. The [shared-start comparison](comparisons/shared_s0_s1_reset_vs_continue/README.md) uses the **same saved S0, S1 and D2** from Week 2, then compares resetting to S0 with continuing from the previous checkpoint. It preserves both earlier experiments. Week labels are presentation aliases, not configuration modes.

The completed runs use one seed and different GPU architectures. Their results are preliminary: perplexity need not rise at every generation, and the diversity measurements are mixed. Read the reports before interpreting a curve as monotonic collapse.

The [implementation handoff](docs/implementation-handoff.md) records **86 offline tests and 14 GPU acceptance checks**, recovery and automatic shutdown. Both test Pods and volumes were deleted. The separately authorized full reset run completed cycles 2-10 at **02:56 UTC/London on 28 October**, after **5.62 hours**. [Completion evidence](docs/validation/2026-10-28_reset-full-completion.json).

The [shared-start analysis](reports/20261027T211852Z-full-reset-s0/report.md) compares the exact same S0/S1 beginning. Reset ends at test PPL **42.10**, versus **55.56** for continued weights, but has more repeated 4-grams (**57.66% versus 52.36%**) and lower token entropy. It preserves real-text prediction better in this seeded comparison while several output-diversity metrics are worse. Metadata and small probes were verified; bulk checkpoint recovery/re-evaluation remains separate.

## Install locally

Python 3.12 is the tested version. Local launch, analysis and offline tests need no large model download or local GPU training. In a new isolated environment, install CPU PyTorch first:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv\Scripts\python.exe -e '.[test]'
.venv\Scripts\Activate.ps1
python -m pytest -q
```

An existing working CUDA PyTorch installation can instead be inherited with `uv venv --system-site-packages .venv`; do not replace it unnecessarily. The remote official Runpod image supplies GPU PyTorch separately. See [operations](docs/operations.md) for credentials, acceptance, recovery and spending controls.

## Configure, validate, then launch separately

The new [five-epoch comparison](comparisons/shared_start_final_epoch5/README.md) retains epoch 5 in both arms; historical Week 2 executed five epochs but selected epoch 1. Its paired configuration runs continued then reset on one detached GPU owner. Use its dedicated commands and acceptance report; it does not reuse the historical S1/D2 or the old acceptance certificate.

Its [handoff](docs/final-epoch5-handoff.md) records 131 passed offline tests, four real five-epoch GPU stages, private restores, recovery and verified shutdown/cleanup for $0.1544 observed account debit. Full compute remains unlaunched. The current conservative $25 ledger gate has $5.64 available versus a $10.39 forecast, so budget reconciliation is required before a later full launch; account credit alone does not override the ceiling.

The setup task prepares the next condition and validates it with a small GPU test. **It does not start the full experiment.** A configuration file is the preferred interface; legacy `--mode fresh_base` and `--mode continued` retain their existing meaning. `--mode` and `--config` are mutually exclusive.

```powershell
python -m model_collapse preflight --probe
python -m model_collapse smoke --config experiments/opt125m_wikitext2/reset_to_real_s0/smoke.yaml
python -m model_collapse status RUN_ID
python -m model_collapse fetch RUN_ID --metadata-only
python -m model_collapse validate RUN_ID
```

Smoke, preflight with `--probe`, and cleanup have real external effects; the smoke command rents compute. They are not part of the offline test suite. Acceptance must match the source and protocol that actually ran. Check the resulting validation report before describing a changed workflow as ready.

The full command below was used for the separately authorized launch. **The run `20261027T211852Z-full-reset-s0` is complete; do not launch a duplicate.** Use status/fetch/report to inspect it. Future launches require new user authorization, current-source acceptance and live budget/environment checks:

```powershell
python -m model_collapse launch --config experiments/opt125m_wikitext2/reset_to_real_s0/full.yaml --generations 10
```

Only after the launcher confirms `detached_supervisor_and_watchdog_verified: true` can the laptop, browser and local terminal close. The remote worker, backup and budget watchdog then run independently.

## Inspect and recover

```powershell
python -m model_collapse status RUN_ID
python -m model_collapse fetch RUN_ID --metadata-only
python -m model_collapse report RUN_ID --out reports/RUN_ID/generated/local-review
```

Metadata-only fetch leaves bulk weights and training arrays in private cloud storage. A full `fetch RUN_ID` downloads several GB. Report generation writes new analysis output; it must not overwrite published evidence or authored reports. Interrupted unfinished runs can use `resume RUN_ID`; completed runs cannot be resumed into new experiments. `cleanup RUN_ID` removes only registered task resources after the required recovery verification.

The cumulative Runpod ceiling is **USD 25**, including setup, failed attempts and storage, with **USD 2 reserved** for shutdown/storage. New setup acceptance is capped at **USD 0.75** within the remaining allowances. Retained network volumes continue to bill after GPU shutdown. No credits, automatic payments or paid Hugging Face compute may be purchased.

## Find the right files

- [Architecture and repository layout](docs/architecture.md): shared implementation, experiment definitions, control state and evidence.
- [Methodology](docs/methodology.md): data packing, training, decoding, selection and protocol differences.
- [Metric definitions](docs/metrics.md): scoring denominators, probe scope and interpretation limits.
- [Operations](docs/operations.md): install, launch, monitoring, recovery, cleanup and manual paid tools.
- [Audit](docs/audits/2026-10-27_repository_audit.md) and [implementation plan](docs/plans/repository-and-shared-start.md): reasons for the organisation and safety fixes.
- [Historical instructions](docs/history/README.md): archived setup prompts, preserved as history rather than current execution authority.

Small reports, figures and provenance belong in Git. Credentials, local reference documents, private control state, model weights and bulk token arrays do not.
