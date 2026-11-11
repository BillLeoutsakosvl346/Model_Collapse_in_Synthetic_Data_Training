# Repeated-data overfitting versus recursive synthetic training

Read the [report](report.md), [metrics and reproduction instructions](analysis/four-condition/README.md), and [completion/delivery receipt](../../docs/validation/2026-11-08_repeated-chunk-completion-analysis-and-doc.json). Google Doc pages 85–109 contain fourteen square figures, nine native tables and two fixed same-prompt comparisons. The prior 84 pages were preserved.

| Condition | Final real-test perplexity |
| --- | ---: |
| Fresh real | 28.0529 |
| Repeated real | 28.6100 |
| Fresh synthetic | 342.0113 |
| Repeated synthetic | 215.3885 |

Both new arms completed ten one-epoch stages. Runpod stopped automatically at 20:48:57 UTC on 8 November 2026; no duplicate or resume is needed. Repeated real shows mild late overfitting, not catastrophic deterioration. Recursive synthetic training deteriorates much more, and repeating its prompts produces a different trajectory from using fresh prompts. These are descriptive, single-seed results under the same decreasing learning rate and beam decoder.

The full Pod lifetime was approximately 3 hours 14 minutes. Conservative setup-plus-full cost is approximately USD 2.45; provider bills are still partial. Another seven days of the two retained 60 GB Standard volumes is approximately USD 1.93 under the recorded estimate. The new USD 10 allocation remains sufficient; no credits were purchased. See [operations](operations.json) for exact billing reads, scope and retained resource IDs.
