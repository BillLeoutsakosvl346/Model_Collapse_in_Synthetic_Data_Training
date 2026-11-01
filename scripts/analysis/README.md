# Offline analysis and report tools

These tools read saved evidence. They do not train, provision resources or edit Google Docs. Use new output directories; original reports/raw records remain preserved. Each rebuilt report records input/output hashes, script identity, Git revision and environment.

## Mixed-data acceptance report

Rebuild the 50/50 smoke report and figures from its committed publication:

```powershell
python scripts/analysis/build_mixed_acceptance.py --out reports/20261031T153536Z-smoke-continued/generated/review
```

`--source` can name another copy of that preserved publication. The builder verifies its publication manifest, mixture/selection alignment, final-epoch policy and finite NLL/perplexity arithmetic. It reads historical costs/provider facts from `validation_summary.json`; it does not query today's ledger, cloud or tokenizers. Saved examples are reused without decoding/downloading a model. It explicitly distinguishes recorded historical bulk recovery from checks repeated during this offline rebuild.

This replaces reliance on the original private `.state/build_mixed_acceptance.py` for report construction. That original operational helper is preserved locally as history; its live provider checks and financial reconciliation are dated acceptance evidence, not part of offline publication. The new builder produces a self-contained report/figures/preview and `publication_reproduction.json` in a new destination.

## Week 3 authored report and native-Docs draft

The matched 50/50 reset/continued publication can be independently recalculated and drafted without cloud access:

```powershell
python comparisons/mixed_fraction_final5/analyse_reset50.py --out reports/20261101T030306Z-full-mixed-campaign-reset-mixed50/generated/audit-review
python scripts/analysis/build_reset50_comparison_report.py --out reports/20261101T030306Z-full-mixed-campaign-reset-mixed50/generated/report-review
```

The first command verifies pinned source records, exact shared S0/S1/D2/M2, deterministic selection hashes, all epoch histories, output probes and metric arithmetic. The second builds prose/tables and a native-Docs draft from the frozen analysis; it does not edit Google Docs. Each requires a new output directory. Full checkpoint restoration/model re-evaluation are outside these checks. The analysis script's explicit `--recover` option reads the sealed Hugging Face snapshot only and requires the private saved source proof; normal reproduction is entirely offline.

For the later mixed-data comparison, `python scripts/analysis/build_mixed_comparison_report.py --out reports/20261031T163428Z-full-continued/generated/report-review` rebuilds a separate authored report and native-Docs draft from its frozen analysis. It requires a new destination and does not edit the Google Doc. The [comparison analysis](../../comparisons/continued_mixed50_vs_synthetic_final5/README.md) independently recalculates the statistics/probes.

```powershell
python scripts/analysis/build_week3_report.py --out reports/20261030T180755Z-full-paired-final5/generated/report-review
```

Reads the frozen paired audit, analysis, examples and figures; writes a new `report.md`, `draft.json` and provenance manifest. Relative links point to the preserved source evidence. It requires `--out`, refuses existing/non-empty or published destinations, and no longer overwrites either the authored publication or the private Google Docs editing draft. Producing `draft.json` does not write to Google Docs.

Independent calculation remains separate:

```powershell
python comparisons/shared_start_final_epoch5/analyse_evidence.py --out .state/week3-audit-review
python comparisons/shared_start_final_epoch5/analyse.py --out .state/week3-analysis-review
```

## Other report/figure entrypoints

| Tool | Role |
| --- | --- |
| `python -m model_collapse report RUN_ID [--out NEW_DIRECTORY]` | Shared policy-aware analysis from fetched records, with a new default generated destination |
| [Week 1 reproduction](../../reports/20261020T211121Z-full/reproduce_overview.py) | Historical figure entrypoint; preserve its published path |
| [Week 2 analysis](../../reports/20261026T192740Z-full-continued/analyse_week2.py) | Historical consistency audit/figures; `--figures-only` avoids cloud reads |
| `week1_report_figure.py` / `continued_only_figure.py` | Presentation-specific figures with explicit destinations |
| `build_metric_guide.py` / `formatting_diagnostics_figure.py` | Metric explanation cards and supplementary formatting diagnostics |

The [28 October correction revision](../../reports/presentation_revisions/2026-10-28_audit-corrections/README.md) records commands, input statistics and document-delivery evidence. Formatting exclusions are sensitivity checks with unequal retained samples, not replacement quality metrics.

## Historical reset acceptance export

```powershell
python scripts/analysis/export_acceptance.py RUN_ID --offline-tests COUNT
```

This legacy exporter is specifically for the reset-S0 one-epoch smoke profile. Run only after metadata fetch, validation and temporary-resource cleanup. It exports checksummed small records/figures and refuses an existing publication; it never copies bulk weights/arrays. `--offline-tests` records a previously completed count; it does not run tests. `--superseded` scopes an earlier source certificate. Paired/mixed conditions use their own preserved evidence and builders above; do not apply this reset-specific authored methodology to them.

See [operations](../../docs/operations.md), [reports](../../reports/README.md) and [repository checks](../maintenance/README.md).

## Mixed campaign acceptance export

```powershell
python scripts/analysis/export_campaign_acceptance.py runs/CAMPAIGN_ID --out .state/campaign-acceptance-export
```

Run after campaign metadata recovery and local GPU/provider validation. This offline exporter requires a completed six-stage / thirty-epoch smoke campaign and binds its acceptance receipt to the exact run, manifest runtime/configuration, required campaign protocol and all three child audits. It checks saved epoch-5 selection, NLL/PPL arithmetic, mixture counts/selections and shared cycle-2 identities, then copies allowlisted scientific JSON into a new destination. It preserves three sibling job directories and reuses the campaign/per-job reporters for tables and figure families. It does not copy checkpoints, token arrays, tokenizer contents, credentials or resource registries, and performs no cloud access or model evaluation.

The export contains a report, per-stage and mixture CSVs, selected records, analysis provenance and a checksummed publication manifest. Source indexes also reference uncopied bulk objects; the export is explicitly selected metadata rather than a new full restoration. Spending and current resource state remain separate dated operational evidence. Never overwrite a historical publication with regenerated output.
