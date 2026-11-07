# Dataset preparation and cross-corpus evaluation

The shared [corpus bank](../../corpora/README.md) records source revisions, split membership, selections and exclusions.

- `readers.py` supplies pinned source streams and native split adapters.
- `registry.py` freezes document selections, tokenization, masks, leakage checks and preview excerpts.
- `prepare.py` prepares or verifies an existing bank using CPU tokenization.
- `evaluate.py` evaluates two existing checkpoints with the shared inference engine and writes a square comparison figure, CSV/JSON records and analysis hashes.

```powershell
python -m scripts.data.prepare --config corpora/cross_corpus_v1.yaml --out data/corpus_bank/cross_corpus_v1

# Regenerate a fetched evaluation report into a new directory
python -m scripts.data.evaluate --root runs/RUN_ID --report-only --out reports/RUN_ID/analysis
```

Paid provisioning is confined to the explicitly invoked [operational launcher](../operations/cross_corpus_evaluation.py). Offline tests and report generation have no provisioning side effects. Dataset arrays and original source texts stay in ignored private directories.
