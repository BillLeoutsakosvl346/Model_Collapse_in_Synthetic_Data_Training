# Model collapse in synthetic-data training

Cambridge Engineering IIB, Week 1. A recursive OPT-125M / WikiText-2 experiment following [Shumailov et al. (2024)](https://doi.org/10.1038/s41586-024-07566-y).

Evaluate untouched pretrained OPT, fine-tune G0 on real text, then train G1-G10 on the previous model's continuations. **Every generation reloads the identical original pretrained model.** Evaluation text remains fixed. The remote worker, backups, reports and GPU shutdown operate independently of the laptop.

## Install on Windows

Python 3.10+ and PyTorch are required locally. The official Runpod image supplies remote GPU PyTorch.

```powershell
uv venv --system-site-packages .venv
uv pip install --python .venv\Scripts\python.exe -e '.[test]'
.venv\Scripts\Activate.ps1
python -m pytest -q
```

Put `RUNPOD_API_KEY` in the ignored `.env`. Hugging Face uses `HF_TOKEN` if set, otherwise the existing saved Hub token. Optional `HF_MODEL_REPO` and `HF_DATASET_REPO` select PRIVATE owner/name repositories. Defaults are `iib-model-collapse-week1-models` and `iib-model-collapse-week1-data` under the authenticated user. The token needs repository write/create access. Never commit credentials. GitHub authentication stays local.

## Validate, then launch

```powershell
python -m model_collapse preflight --probe
python -m model_collapse smoke
```

The smoke test uses 256 training blocks, 64 validation/test blocks each, one epoch and G0-G2. It deliberately exits after a saved generation shard and after G1, resumes each, checks G1 files are unchanged, uploads artifacts, builds plots, then exercises actual budget-guard shutdown. **Smoke results validate plumbing; they are not evidence of collapse.**

The full experiment is a separate command:

```powershell
python -m model_collapse launch --generations 10
```

Use `--generations 5` for a shorter chain. G10 means ten synthetic generations after real-data G0. The launcher checks live stock/prices, the measured benchmark and remaining budget. Keep the laptop awake until it returns `detached_supervisor_and_watchdog_verified: true`. Then no browser, SSH, Codex session or local process needs to stay open.

## Monitor and retrieve

Replace `RUN_ID` with the ID returned by the launcher.

```powershell
python -m model_collapse status RUN_ID
python -m model_collapse fetch RUN_ID
python -m model_collapse report RUN_ID
python -m model_collapse cleanup RUN_ID
```

Fetch downloads all indexed artifacts from private Hugging Face repositories and checks SHA-256 hashes. Report rebuilds figures and CSV. Cleanup removes only the registered task Pod and network volume, and requires successful recovery first. The retained volume continues to bill after GPU shutdown: 30 GB Standard storage is approximately $2.10/month. Seven days of storage are reserved in the budget. No additional credits, automatic payments or paid Hugging Face services are used.

For interrupted work with its retained network volume:

```powershell
python -m model_collapse resume RUN_ID
```

Use the original source version and configuration. Resume reuses verified completed generations/shards and the last saved epoch of unfinished training. It provisions a new Pod beside the retained volume. It does not extend the configured generation count or overwrite completed stages.

## Protocol and artifacts

`configs/full.yaml` uses all original WikiText-2 raw splits and pinned model/tokenizer/dataset revisions; five epochs; AdamW LR 2e-5, betas (0.9,0.999), epsilon 1e-8, weight decay 0.01; cosine decay by epoch towards 1e-6; effective batch 128; seed 42; BF16 autocast with FP32 parameters. Best real validation epoch supplies generation. Five-beam deterministic decoding generates exactly 64 new tokens after each fixed 64-token real training prompt, repetition penalty 1.0 and no repetition constraint. Only continuation token IDs train the next model.

Preserve source row strings including newlines/blank rows, add OPT default special tokens per row, concatenate once within each split and divide into non-overlapping 64-token blocks. Discard each split's final incomplete block. The authors group map batches instead, so this packing is a documented deviation. Their released generation call hardcodes repetition penalty 3.0; this run uses 1.0 for the paper's main condition. Explicit fresh-base initialization, mixed precision, corrected corpus perplexity, modern libraries and one seed also limit exact numerical comparability. Published perplexities are reference context, not acceptance targets.

Corpus perplexity = exp(total NLL / scored shifted targets). Each unpadded 64-token block scores 63 targets. Mean sequence perplexity and its actual standard deviation are separately labelled. Diversity compares continuations for the same first 256 training prompts, excluding prompt tokens. Frozen G0 scores evaluate each continuation alone with 63 targets.

`runs/RUN_ID/` contains config/environment/resource records, manifests, frozen token arrays, generation shards, selected model checkpoints, per-sequence losses, samples and report. Private Hub repositories mirror recoverable artifacts. Large files, caches, credentials and local proposal/paper stay out of Git. CSV and PNG plots cover perplexity, diversity/repetition, G0 reference scores and timing.

The modular implementation separates preparation (`data.py`), training/generation/evaluation (`engine.py`), recursion (`pipeline.py`), reporting (`report.py`), lifecycle (`remote.py`) and independent supervision (`supervisor.py`). Completed stages are verified before reuse; unfinished training resumes at epoch boundaries. Fixed seeds do not guarantee bitwise equality across different GPU/library versions.

The cumulative ceiling is $25, including the $3 setup allocation; $2 is held back for shutdown/storage. The independent watchdog enforces a price-derived deadline that does not reset across trainer restarts. A provider/API outage can delay shutdown despite retries; validation proves the tested path, not a provider billing guarantee. Partial or budget-paused chains are reported honestly. Mitigation methods and extra seeds are outside this setup.
