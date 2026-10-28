# Audit corrections and interpretation

This presentation revision updates the main Google Doc and its illustrated metric guide after the [independent audit](../../../docs/audits/2026-10-28_graph_and_protocol_audit.md). It preserves all dated run reports, measurements, raw evidence and accepted runtime identities. No new inference, training or paid cloud action was performed.

Changes:

- Correct the final-arm versus S0 comparison and distinguish observations from possible mechanisms.
- Explain nonmonotonic perplexity, mixed diversity, pooled versus within-output entropy and validation-selected epoch inheritance.
- Add a supplementary formatting diagnostic: final reset/continued equality-token fractions are 22.82%/11.00%; diagnostic exclusion changes the entropy gap from 0.620 to 0.122 nats. Exclusion changes retained sample sizes and does not replace primary metrics or measure semantic quality.
- State that the first 256 prompts span six consecutive articles, and explain forced-length EOS suppression.
- Move legends away from data. Replace the Week 1 presentation figure with a fully scripted version using 35 explicit common histogram bins from 1 to 8 nats; preserve the previous raster in `archive/`.

`guide.json` at this directory's root is the complete current authored reading guide, including its new Section 5a. `editorial_changes.json`, `editorial_followup.json` and `supplementary_sections.json` record the document changes. The builder's `figures/guide/guide.json` is its original template output, not the authored document text; only the root plan represents this delivered revision.

`inputs/` contains verified block-loss exports and supplementary diagnostic summaries. These are measured statistics, not full training text, model weights or credentials. Input provenance, output hashes, environment and source identities are recorded in manifests and the delivery receipt.

Reproduce the figures into new destinations:

```powershell
python scripts/analysis/week1_report_figure.py --block-losses reports/presentation_revisions/2026-10-28_audit-corrections/inputs/week1_block_losses.csv --out .state/review-week1.png
python scripts/analysis/continued_only_figure.py --out .state/review-continued.png
python scripts/analysis/build_metric_guide.py --out .state/review-guide-figures
python scripts/analysis/formatting_diagnostics_figure.py --inputs reports/presentation_revisions/2026-10-28_audit-corrections/inputs --out .state/review-formatting.png
```

Choose unused output paths: the scripts refuse to overwrite published assets. The root authoring plan references the generated cards and the supplementary figure. Live-document verification and PDF layout results are in `google_doc_delivery.json`; authenticated exports and native snapshots remain private under `.state/2026-10-28_report-corrections/`.
