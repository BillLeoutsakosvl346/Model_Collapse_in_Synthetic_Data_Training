# Model collapse in synthetic-data training

Cambridge Engineering IIB research with OPT-125M, WikiText-2 and FineWeb-Edu: recursively generate training data, train successive checkpoints, and measure prediction loss, repetition and output diversity. The experiments reconstruct selected ideas from [Shumailov et al. (2024)](https://doi.org/10.1038/s41586-024-07566-y); each condition records its own methodological differences.

## Experiments and results

| Condition | Scientific difference | Recorded result/status |
| --- | --- | --- |
| [SmolLM2-135M fraction sweep](comparisons/smollm2_135m_fineweb_completion64_fractions/README.md) | Seven native-tokenizer continuation mixtures; fast two-update compatibility acceptance | GPU validated; seven-condition sweep running |
| [Matched FineWeb completions](comparisons/fineweb_real_vs_synthetic_completion64/README.md) | Same fresh prompts, real versus self-generated continuation targets, one epoch per chunk | [Completed comparison and analysis](docs/validation/2026-11-07_fineweb-four-fraction-completion-analysis.json) |
| [FineWeb-Edu real-only pilot](experiments/opt125m_fineweb_edu/fresh_real_one_epoch/README.md) | Fresh documents, one epoch per cycle, continuous AdamW and global cosine schedule | [Completed pilot and shutdown](docs/validation/2026-11-06_fineweb-full-completion-status.json); authored analysis pending |
| [Pretrained reset](experiments/opt125m_wikitext2/pretrained_reset/README.md) | Reset to original pretrained OPT each cycle | [Week 1 report](reports/20261020T211121Z-full/report.md) |
| [Continued weights](experiments/opt125m_wikitext2/continued_from_real_s0/README.md) | Continue from the preceding selected checkpoint | [Week 2 report](reports/20261026T192740Z-full-continued/report.md) |
| [Shared-start reset versus continue](comparisons/shared_s0_s1_reset_vs_continue/README.md) | Same S0/S1/D2; fixed-S0 reset versus continued weights | [Matched Week 2 report](reports/20261027T211852Z-full-reset-s0/report.md) |
| [Final epoch 5 comparison](comparisons/shared_start_final_epoch5/README.md) | Retain epoch 5 in both shared-start arms | [Week 3 report](reports/20261030T180755Z-full-paired-final5/report.md) |
| [Continued 50/50 mixture](experiments/opt125m_wikitext2/continued_mixed_50_final5/README.md) | Equal real/synthetic targets; continued epoch-5 weights | [Matched synthetic-only comparison](reports/20261031T163428Z-full-continued/report.md) |
| [Mixed-fraction campaign](comparisons/mixed_fraction_final5/README.md) | 50/50 reset, then matched 20/80 continued/reset; retain epoch 5 | [Completed analyses](reports/README.md); [execution record](docs/current-state.md) |

Use the [current-state record](docs/current-state.md) for the latest recorded authorization and status, the [catalogue](experiments/catalog.yaml) for exact run IDs/configurations, and the [reports index](reports/README.md) for results, validation and recovery scope. Week labels identify reports, not runtime modes. Existing runs use one seed; comparisons must account for their recorded hardware, checkpoint policy and shared artifacts.

## Install and check offline

Use an editable repository checkout; standalone wheel deployment is not supported. Python 3.12 is tested. Local launch/analysis needs no local GPU or full checkpoint download. For a new Windows environment:

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

## Code structure

- `model_collapse/science/`: shared training, evaluation, generation, WikiText preparation, metrics and pure sampling/hash helpers.
- `model_collapse/workflow/`: named experiment loops, lineage, checkpoint/optimization policies and lazy CLI dispatch.
- `model_collapse/cloud/`: provider client, launch/SSH lifecycle, ownership, budget, storage and detached supervision.
- `model_collapse/artifacts/`: pinned backup, recovery and safe paths.
- `model_collapse/analysis/`: offline readers, forecasts, tables, plots and provenance.

Root-level compatibility modules preserve established imports. FineWeb data builders retain their original locations because their source is part of the frozen data contract. See [architecture](docs/architecture.md) before moving them.

## Find the right files

- [Documentation index](docs/README.md): methodology, metric definitions, operations, architecture and history.
- [Experiments](experiments/README.md) and [comparisons](comparisons/README.md): scientific definitions and matched settings.
- [Reports](reports/README.md): preserved evidence, published analyses and authored interpretation.
- [Literature](literature/README.md): paper citations, local PDF locations and research context.
- [Analysis tools](scripts/analysis/README.md): reproducible figures and report builders with protected output destinations.
- [Legacy configurations](configs/README.md): compatibility paths; prefer named experiment configurations.

Small reports and provenance belong in Git. Credentials, private control/recovery state, local reference documents, weights and bulk training arrays stay private. [AGENTS.md](AGENTS.md) contains the stable working rules.
