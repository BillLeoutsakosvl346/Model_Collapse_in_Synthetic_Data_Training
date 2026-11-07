# Cross-corpus data bank

The `cross_corpus_v1` bank prepares five text sources for fixed held-out evaluation and future training mixtures. All sources use the pinned OPT-125M tokenizer and the same variable-length, masked sequence policy, with at most 512 tokens including BOS/EOS.

## Frozen collections

| Source | Training source tokens | Validation source tokens | Test source tokens | Split membership |
|---|---:|---:|---:|---|
| FineWeb-Edu | 2,000,000 | 1,000,439 | 978,319 | Exact previous held-out collections, training drawn from the previous training selections |
| WikiText-103 raw | 2,000,000 | 128,000 | 128,000 | Official train, validation and test articles |
| English C4 | 2,000,000 | 128,000 | 128,000 | Official training pool, URL/text-hash halves of official validation for validation/test |
| PG-19 books | 2,000,000 | 128,000 | 128,000 | Official book-disjoint train, validation and test lists |
| AG News | 2,000,000 | 128,000 | 128,000 | Official test, with a deterministic validation partition of official training |

The training bank totals 10 million source tokens. These are finite subsets of the published datasets. WikiText-103 supplies the single Wikipedia source. Its held-out raw files are also the WikiText-2 held-out files, so those two would duplicate evaluation coverage.

Source revisions and selection defaults are in [cross_corpus_v1.yaml](cross_corpus_v1.yaml). The [preparation receipt](../docs/validation/2026-11-06_cross-corpus-bank.json) records the selected document counts, exclusions and bank identity.

## Selection and separation

- Seed 42 drives source-specific bounded stream shuffles and deterministic split partitions. Read files and coverage are recorded. This sampling is reproducible, with finite read coverage rather than uniform access over every document in the complete source.
- Reserve validation and test collections before allocating any training pools.
- Exclude shared document identities, normalized URLs/text, canonical Wikipedia titles and exact sliding 64-token passages across collections. Every held-out collection also excludes the earlier FineWeb pilot's training documents/passages.
- PG-19 holds out whole books. A long foreign-source document contributes at most a deterministic 16,384-token window. Each PG-19 evaluation set therefore contains eight book windows.
- Preserve text, split automatically into sequences, insert BOS at sequence starts and EOS only at genuine document endings. Padding is excluded from attention, targets and denominators.
- Final selected passages are trimmed to meet source-token quotas. Scored target counts also include genuine EOS tokens, so they can exceed source-token counts.
- Original OPT pretraining exposure and semantic/shorter approximate overlaps are outside the verified separation scope.

FineWeb training passages were previously used in the pilot. A later experiment starting from an adapted checkpoint must account for that history. The frozen training banks support a future mixture protocol, whose cycle allocations still need to be specified.

## Local artifacts and commands

```powershell
python -m scripts.data.prepare --config corpora/cross_corpus_v1.yaml --out data/corpus_bank/cross_corpus_v1
```

The ignored `data/corpus_bank/cross_corpus_v1/` directory contains approximately 421 MB. Each source/split has token arrays, padding masks, original document records, source locations and a checksum manifest. The root `manifest.json` seals the complete bank. `preview.html` and `examples.json` show fixed seeded excerpts.

An existing completed bank is verified before reuse. Configuration or preparation-code changes require a new output directory. Original documents and bulk arrays remain outside Git.

## One-epoch comparison

The manual [evaluation launcher](../scripts/operations/cross_corpus_evaluation.py) compares the existing pretrained checkpoint and cycle-1 checkpoint from `20261106T151257Z-full-fresh-real`. That checkpoint has already completed one epoch on 640,000 FineWeb-Edu source tokens. The launcher performs zero training updates.

```powershell
# Explicit paid evaluation-only job, capped at USD 0.50
python -m scripts.operations.cross_corpus_evaluation --execute
```

Both checkpoints are evaluated on identical arrays, using token-weighted corpus NLL and `perplexity = exp(NLL)`. Compare changes within each corpus. Different corpora have different intrinsic predictability, lengths and content, so their absolute perplexities require that context.

The launcher verifies private source revisions, budget, live stock and the official image. A separate watchdog controls shutdown. After completion it recovers checksum-verified small artifacts and removes only its temporary evaluation resources. Historical recovery resources are preserved.
