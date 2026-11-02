# Continued real-only control versus synthetic mixtures

This comparison adds the completed real-only control `20261102T154833Z-full-continued` to the five-epoch/final-epoch continued runs at 100%, 50% and 20% synthetic data. It is an analysis definition, not a launch configuration. All full runs are complete; do not duplicate them.

| Synthetic fraction | Run / evidence |
| --- | --- |
| 100% | `20261030T180755Z-full-paired-final5`, continued arm |
| 50% | `20261031T163428Z-full-continued` |
| 20% | `20261101T030306Z-full-mixed-campaign-continued-mixed20` |
| 0% | `20261102T154833Z-full-continued` |

All share the exact real-trained S0, frozen evaluation arrays, model/data revisions, seed and five-epoch/final-checkpoint policy. Each dataset has 37,946 blocks of 64 tokens. The real-only arm selects the complete real training pool each cycle, changing its order rather than its contents. It generates only the fixed 256-output probes. Historical synthetic-containing arms generate full synthetic source pools before mixture sampling. Initial D1 remains imported and unused for real-only targets.

The real-only GPU is RTX PRO 4500 Blackwell Server Edition; comparators used RTX PRO 4000. Recorded software/driver settings match, and the archived training, packing, metric and selection code is identical. The hardware difference remains a confounder; this is a one-seed descriptive comparison.

The [report](../../reports/20261102T154833Z-full-continued/report.md), [analysis](../../reports/20261102T154833Z-full-continued/analysis/) and [timing audit](../../reports/20261102T154833Z-full-continued/timing_audit.md) preserve distinct scopes. Small metadata/probe verification does not establish independent bulk checkpoint restoration or re-evaluation.

Reproduce in a new destination, without cloud access or training:

```powershell
python comparisons/continued_real_only_vs_mixtures/analyse.py --out reports/20261102T154833Z-full-continued/generated/audit-review
python scripts/analysis/build_real_only_comparison_report.py --out reports/20261102T154833Z-full-continued/generated/report-review
```

The analysis validates 44 metric observations, 200 epoch entries and 41 unique probes, verifies historical schemas through readers, reconstructs deterministic real-only selection hashes, and recalculates corpus NLL/PPL and output statistics. Seven figure families use square panels; the final four-panel summary places cycle-10 perplexity, entropy, distinct-2 and within-output repetition against synthetic fraction. Joined points are not a fitted optimum or independent-run uncertainty estimate.

`--recover` explicitly reads the pinned private Hugging Face backup recorded in the completion receipt; normal execution uses the committed selected records/probes only. Neither tool edits Google Docs, provisions resources or loads model weights.
