# Independent audit of the FineWeb synthetic-fraction experiments

11 November 2026. This is an audit of completed experiments, not a new training run. Five fresh reviewers examined data/mixing, training mathematics, executed provenance, metrics/reporting and repository publication. No training or paid compute was launched.

**No training, mixing or numerical graph fault was found in the inspected evidence.** The 99% curve really flattens near 83 within the measured ten-cycle window. One derived-table provenance field has ambiguous checksum scope; a new explicit source map resolves it without rewriting the published results. Two new tests also depended on ignored local recovery files; they were fixed before publication. These repository issues do not change the recorded experiments.

## What actually ran

Every condition starts independently from the same original pretrained OPT-125M checkpoint. It processes the same ten fresh, document-separated FineWeb-Edu chunks, each containing 10,000 examples. An example is BOS + 64 real prompt tokens + 64 continuation tokens; only the continuation labels contribute to loss.

For each cycle, a deterministic seeded permutation selects the rows whose continuations become synthetic. That arm's preceding checkpoint generates their targets with five deterministic beams, exactly 64 new tokens and repetition penalty 1.0. Other rows retain their original real targets. Generated targets are scattered into their original prompt positions; each prompt contributes one training example. The frozen dataset trains its own preceding checkpoint for one terminal epoch. Weights, AdamW moments, training RNG and the global cosine scheduler continue into the next cycle.

| Synthetic labels | Synthetic / real examples per cycle | Final real-test perplexity | GPU |
| --- | --- | ---: | --- |
| 0% | 0 / 10,000 | 28.0529 | RTX PRO 4000 |
| 20% | 2,000 / 8,000 | 28.3713 | RTX PRO 4000 |
| 50% | 5,000 / 5,000 | 29.0698 | RTX PRO 4000 |
| 80% | 8,000 / 2,000 | 31.0178 | RTX PRO 4000 |
| 85% | 8,500 / 1,500 | 32.0951 | RTX PRO 4500 |
| 90% | 9,000 / 1,000 | 33.6830 | RTX PRO 4500 |
| 95% | 9,500 / 500 | 38.5688 | RTX PRO 4500 |
| 99% | 9,900 / 100 | 82.6773 | RTX PRO 4500 |
| 99.9% | 9,990 / 10 | 258.6782 | RTX PRO 4500 |
| 100% | 10,000 / 0 | 342.0113 | RTX PRO 4000 |

All begin at test PPL 29.1763. Each arm receives 6.4 million scored training targets and 790 optimizer updates. Primary evaluation always scores the same real held-out continuations: 457,664 test targets and 468,608 validation targets. The ten-condition aggregate is preserved in [records](records/aggregate_metrics_110.json); seven principal conditions additionally have independent raw-stage/probe verification.

## What was checked

- **Executed source, not only today's code:** six original runtime archives reproduce their recorded digests. Evaluation and generation semantics match; later engine refactoring only changes import paths and extracts an equivalent persistent-optimizer policy. Critical mixer/lineage functions match across the seven mixture executions. The private archive hashes remain in [source review](review/provenance-review.json).
- **Actual training:** seventy saved stages across 0/50/80/95/99/99.9/100% have correct generator, initializer, preceding optimizer, fresh chunk, selected-terminal checkpoint and update counts. Every stage changes its weight hash. Forty mixture stages independently reproduce 160 serialized permutation/index/label/row-map hashes.
- **Direct 99% array reconstruction:** five checksum-pinned cycle-10 files, totaling 20,575,040 bytes, were recovered read-only. All 100 real continuation rows are unchanged; all 9,900 synthetic rows contain the correctly mapped generated answers. Full training tokens, attention and continuation masks reconstruct byte-for-byte. See [scatter evidence](review/scatter-review.json) and the independent [data review](review/data-review.json).
- **Full frozen bank:** an independent CPU rerun reconstructed all 100,000 training pairs and all usable held-out pairs from saved text and offsets. Source identities and exact sliding 64-token passages are disjoint across collections. Probe document stratification and all indexed file hashes pass.
- **Perplexity:** all 77 selected published rows match the indexed original stage metrics. Thirty available per-example loss arrays independently reproduce corpus NLL, PPL and P95; maximum NLL discrepancy is 8.88e-16. These are token-weighted corpus scores, not means of passage perplexities.
- **Diversity graphs:** seventy trained 256-by-64 probe arrays match source checksums. All nineteen diversity/repetition/formatting statistics recompute with zero difference. Plot code uses the recorded values directly, without smoothing or fitted curves.

These checks establish source/data/metadata consistency and saved-loss arithmetic. They do not substitute for independently loading and re-evaluating every historical checkpoint. This audit used no GPU inference; historical bulk weights and optimizer states remain privately preserved.

## The real issue found: mixed provenance scope

In the seven-fraction table, `source_metrics_sha256` refers to two different kinds of source. Forty-four inherited 0/50/80/100% rows carry the checksum of the earlier five-fraction aggregate table; newer rows carry the checksum of their individual stage's `metrics.json`.

Both checksums refer to genuine inputs, and the numerical values match the original stage records. The field is nevertheless ambiguous. The [new explicit provenance map](generated/verified-20261111/corrected_provenance.csv) records the actual per-stage file path/hash and separately identifies the inherited aggregate checksum. The original reports, indexed bytes and published curves remain unchanged.

## How to interpret the surprising result

The recorded 99% trajectory is:

`29.18 → 43.56 → 61.29 → 73.75 → 82.85 → 84.41 → 83.06 → 82.99 → 83.22 → 83.32 → 82.68`

- **The real anchor is small, but repeatedly refreshed.** At 99%, each cycle supplies 100 fresh real examples / 6,400 real target tokens. Across ten disjoint chunks, that is 1,000 examples / 64,000 target tokens spanning 858 source documents. It is not one real example repeated a thousand times.
- **A label fraction is not a gradient fraction.** Easy, highly confident self-generated answers can contribute small corrective gradients. Harder real answers may keep pulling predictions toward the source distribution. A late balance of these effects is a plausible explanation for flattening; no source-specific gradient decomposition was measured, so it is not a proven mechanism.
- **Learning rate keeps decreasing.** All conditions use cosine 5e-6 to 1e-6. At cycle 5 it is still 3e-6 and weights continue changing. The same schedule does not prevent 99.9% or 100% from worsening, but it remains a reason not to extrapolate the ten-cycle flattening into permanent stability.
- **50% is close on primary PPL, not every quality measure.** Its final PPL 29.0698 is near the original 29.1763, but local repeated four-grams rise to about 47.7%, versus roughly 39% initially. Secondary PG-19 and AG News PPL rise approximately 9.3% and 5.5%, respectively. Predictive likelihood, repetition and topic transfer measure different properties.
- **The result depends strongly on decoding.** These mixtures used deterministic five-beam targets. The later [decoder comparison](../20261110T234038Z-full-completion64-decoding/report.md) ends at PPL 31.88 for 100% pure-sampling targets, versus 342.01 for historical 100% beam targets. There is no universal statement here that 1% real data is necessary or sufficient to prevent collapse for all generators.

## Other design limitations

- Percentages concern **continuation labels**. Even 100% synthetic retains 64 real conditioning tokens per prompt: 6.4 million real prompt tokens across ten cycles.
- GPU type is partly confounded with percentage. Both are Blackwell, but hardware-dependent numerical/beam differences cannot be assumed zero. The 99% versus 99.9% contrast shares the 4500; 0/20/50/80/100% share the 4000.
- The bank is a bounded-shuffle sample from 24,950 rows of one shuffled FineWeb shard, not uniform random addressing throughout all 10B tokens. It contains varied source documents but is one finite selected bank.
- There is one seed and ten cycles. There are no training-seed uncertainty estimates, permanent-stability test or universal threshold estimate. Exact-passage/document exclusions do not prove semantic deduplication, human authorship or no overlap with OPT pretraining.
- The final 16-example partial optimizer batch is normalized correctly over its own tokens. It creates ordinary finite-batch weighting variation; reconstructed real-label coefficients differ only slightly around 1%, and this is not evidence of hidden real-data oversampling.

## Reproduce the public numerical audit

The small `records/` tree preserves scientific JSON and seventy fixed probe arrays; it excludes credentials, complete training banks, model weights and optimizer blobs. Use a new output directory:

```powershell
python -m scripts.analysis.audit_fineweb_fraction_records --out reports/20261111_fineweb_fraction_audit/generated/reproduce
```

This performs no cloud calls, training or inference. It verifies the raw metrics, probe statistics, available example-loss arrays and explicit provenance map. Private source-archive and full-bank checks have separate recorded scopes. Review receipts and [verification summary](verification.json) record the test and publication evidence.

The most useful follow-up would preserve matched hardware and data while testing multiple seeds, a constant LR/longer horizon, and real-versus-synthetic loss/gradient contributions separately. Those are proposed experiments; none was launched by this audit.
