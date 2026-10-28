# Illustrated metric-reading guide

Authored in the user's existing Google Doc on 28 October 2026. The 11-page guide
groups the existing results by metric, with each graph family and equation above
its explanation, a small intuitive example and the relevant measured results.

It covers real NLL/perplexity, distinct-1/2/3, repeated four-grams, exact duplicate
continuations, empirical entropy, frozen-reference scoring, passage distributions
and P95, epoch selection, and data/runtime/resource accounting. The final page
connects the measurements and maps them to the original report panels.

`guide.json` is the native-document authoring plan. `manifest.json` records the
25 immutable inputs, source revision, script hash, environment and asset hashes.
`google_doc_delivery.json` records native readback and rendered-PDF checks.

Regenerate in a new directory without a GPU, network, model downloads or cloud
resource changes:

```powershell
python scripts/analysis/build_metric_guide.py --out analysis/generated/metric-guide-review
```

The program refuses a non-empty destination. All source experiment reports,
metrics, figures, artifact indexes and historical delivery receipts are preserved.
Illustrations replot existing measurements; they are not new experimental runs.
