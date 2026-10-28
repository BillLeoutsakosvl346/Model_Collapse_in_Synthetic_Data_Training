# Acceptance evidence export

`export_acceptance.py` publishes a verified reset-branch smoke test after metadata fetch, local acceptance and temporary-resource cleanup have succeeded:

```powershell
python scripts/analysis/export_acceptance.py RUN_ID --offline-tests COUNT
```

It reads local records, verifies their checksums, exports small evidence and regenerates figures. It does not make cloud requests or copy model/token tensors. `--offline-tests` records an already completed suite; it does not run tests. `--superseded` labels successful acceptance of an earlier runtime which must not authorize changed code.

Published evidence is immutable: the exporter refuses to overwrite an existing acceptance. Every exported role has a checksum in `publication_manifest.json`; Git attributes preserve those bytes on Windows. General analysis uses `python -m model_collapse report RUN_ID [--out OUTPUT_DIRECTORY]` and writes a new destination.

See [operations](../../docs/operations.md) and the [validated setup handoff](../../docs/implementation-handoff.md).

## Report figures and interpretation

`week1_report_figure.py` rebuilds the six-panel Week 1 presentation with explicit common histogram bins. `continued_only_figure.py` renders the standalone continued experiment. `build_metric_guide.py` renders the metric cards; `formatting_diagnostics_figure.py` renders the supplementary audit diagnostic. All take explicit new output destinations and preserve prior publications.

The [28 October correction revision](../../reports/presentation_revisions/2026-10-28_audit-corrections/README.md) contains complete commands, verified input statistics, the current authored guide and document-delivery receipts. Supplementary formatting exclusions are sensitivity checks with unequal retained sample sizes, not replacements for the historical raw-token metrics.
