# Model collapse in synthetic-data training

Cambridge Engineering IIB research with OPT-125M and WikiText-2: recursively generate training data, train successive checkpoints, and measure prediction loss, repetition and output diversity. The experiments reconstruct selected ideas from [Shumailov et al. (2024)](https://doi.org/10.1038/s41586-024-07566-y); each condition records its own methodological differences.

## Experiments and results

| Condition | Scientific difference | Recorded result/status |
| --- | --- | --- |
| [Pretrained reset](experiments/opt125m_wikitext2/pretrained_reset/README.md) | Reset to original pretrained OPT each cycle | [Week 1 report](reports/20261020T211121Z-full/report.md) |
| [Continued weights](experiments/opt125m_wikitext2/continued_from_real_s0/README.md) | Continue from the preceding selected checkpoint | [Week 2 report](reports/20261026T192740Z-full-continued/report.md) |
| [Shared-start reset versus continue](comparisons/shared_s0_s1_reset_vs_continue/README.md) | Same S0/S1/D2; fixed-S0 reset versus continued weights | [Matched Week 2 report](reports/20261027T211852Z-full-reset-s0/report.md) |
| [Final epoch 5 comparison](comparisons/shared_start_final_epoch5/README.md) | Retain epoch 5 in both shared-start arms | [Week 3 report](reports/20261030T180755Z-full-paired-final5/report.md) |
| [Continued 50/50 mixture](experiments/opt125m_wikitext2/continued_mixed_50_final5/README.md) | Equal real/synthetic targets; continued epoch-5 weights | [Matched synthetic-only comparison](reports/20261031T163428Z-full-continued/report.md) |
| [Mixed-fraction campaign](comparisons/mixed_fraction_final5/README.md) | 50/50 reset, then matched 20/80 continued/reset; retain epoch 5 | [Setup handoff](docs/mixed-campaign-handoff.md); full campaign unlaunched |

Use the [current-state record](docs/current-state.md) for the latest recorded authorization and status, the [catalogue](experiments/catalog.yaml) for exact run IDs/configurations, and the [reports index](reports/README.md) for results, validation and recovery scope. Week labels identify reports, not runtime modes. Existing runs use one seed; comparisons must account for their recorded hardware, checkpoint policy and shared artifacts.

## Install and check offline

Python 3.12 is tested. Local launch/analysis needs no local GPU or full checkpoint download. For a new Windows environment:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv\Scripts\python.exe -e '.[test]'
.venv\Scripts\Activate.ps1
python -m pytest -q
python scripts/maintenance/check_repository.py
```

Remote GPU environments reuse their image's CUDA PyTorch. See [operations](docs/operations.md) for credentials and paid execution; acceptance and full launch are separate authorized actions. The recorded 50/50 run must not be launched again.

## Inspect saved work

```powershell
python -m model_collapse status RUN_ID
python -m model_collapse fetch RUN_ID --metadata-only
python -m model_collapse report RUN_ID
```

Status/fetch access the cloud. Metadata-only fetch leaves bulk checkpoints and token arrays there; full fetch can download several GB. Reporting is offline and writes a new `reports/<run>/generated/<analysis>/` destination. Resume is only for unfinished stopped runs; cleanup requires verified ownership and adequate recovery.

## Find the right files

- [Documentation index](docs/README.md): methodology, metric definitions, operations, architecture and history.
- [Experiments](experiments/README.md) and [comparisons](comparisons/README.md): scientific definitions and matched settings.
- [Reports](reports/README.md): preserved evidence, published analyses and authored interpretation.
- [Literature](literature/README.md): paper citations, local PDF locations and research context.
- [Analysis tools](scripts/analysis/README.md): reproducible figures and report builders with protected output destinations.
- [Legacy configurations](configs/README.md): compatibility paths; prefer named experiment configurations.

Small reports and provenance belong in Git. Credentials, private control/recovery state, local reference documents, weights and bulk training arrays stay private. [AGENTS.md](AGENTS.md) contains the stable working rules.
