# Greedy versus sampling in recursive synthetic training

Read the [report](report.md), [44-row metrics and reproduction guide](generated/decoding-20261111/README.md), [operations](operations.json), and [delivery verification](document_delivery.json).

| Condition | Initial test perplexity | Cycle-10 test perplexity |
| --- | ---: | ---: |
| Real targets (historical) | 29.1763 | 28.0529 |
| Five-beam synthetic targets (historical) | 29.1763 | 342.0113 |
| Greedy synthetic targets | 29.1763 | 390.9933 |
| Pure-sampling synthetic targets | 29.1763 | 31.8782 |

Both new arms completed ten continued cycles and stopped automatically. Sampling markedly reduces deterioration in this one-seed protocol, but test perplexity still rises at every cycle. The common five-beam probe, decreasing LR and historical hardware difference are explicit throughout the report.

Google Doc pages 110–130 contain nine square figures, seven native tables and two fixed same-prompt examples. All preceding 109 pages are pixel-identical to the pre-edit export, and native content/styles were unchanged. The 492 small JSON records contain the original scientific schemas and pinned indexes; no credentials, full checkpoints, optimizer blobs or bulk token arrays are committed. Recovery completeness is qualified in the analysis guide and receipt. Retained storage continues billing.
