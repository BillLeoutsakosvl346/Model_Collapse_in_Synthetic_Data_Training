# Continued final-epoch-5 training: 50/50 versus synthetic-only

Compare `20261031T163428Z-full-continued` with the **continued** arm of `20261030T180755Z-full-paired-final5`. Both execute ten cycles / fifty epochs, retaining epoch 5; initialization is always the preceding selected checkpoint. The reset arm and historical best-validation-selected checkpoints are outside this comparison.

Shared: exact real-trained S0 and historical D1, pinned OPT-125M/WikiText-2, fixed prompts/evaluation, 37,946 64-token training blocks, seed, optimizer/scheduler restarts, decoding and recorded GPU/software. Deliberate difference: each new mixed training dataset samples 18,973 real and 18,973 synthetic blocks; the comparator uses all 37,946 synthetic blocks. S1 differs intentionally. Real rows are resampled from the same finite training split, not new external text.

```powershell
python comparisons/continued_mixed50_vs_synthetic_final5/analyse.py --out reports/20261031T163428Z-full-continued/generated/review
```

The normal path is offline: checks saved records/probes, independently recalculates statistics, reconstructs selection indices from seed and compares their hashes, then writes new tables/figures/provenance. Add `--recover` only to retrieve missing pinned private metadata and ten small probe shards; it never rents compute or downloads full token arrays/model weights. Existing evidence cannot be overwritten.

Primary metrics include all tokens. Newline exclusion, standalone-space concentrations and decoded whitespace-only counts are explicit sensitivity/formatting diagnostics. The fixed first 256 prompts are not a representative random sample. This is one seeded trajectory per condition, not an independently replicated significance test. See the [report](../../reports/20261031T163428Z-full-continued/report.md) and [methodology](../../docs/methodology.md).
