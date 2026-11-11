# Offline analysis and report tools

These tools read saved evidence. They do not train, provision resources or edit Google Docs. Use new output directories; original reports/raw records remain preserved. Each rebuilt report records input/output hashes, script identity, Git revision and environment.

## Repeated versus fresh FineWeb completions

Reproduce the completed four-condition comparison from verified local metadata:

```powershell
python -m scripts.analysis.build_repeated_chunk_comparison --root runs/COMPARISON_ID --reference runs/20261106T193918Z-full-completion64 --out reports/COMPARISON_ID/generated/four-condition-review
```

The builder requires matching bank, primary evaluation, initial weights, hardware and scientific settings. It keeps fresh/repeated conditions separate at each synthetic fraction, verifies 44 checkpoint rows and 790 updates per arm, and saves square figures plus input/output hashes. Fixed-real-anchor diagnostics apply to the new repeated arms. Incomplete comparisons and nonempty outputs are rejected. No training or cloud access occurs.

The [completed publication](../../reports/20261108T173505Z-full-completion64/analysis/four-condition/README.md) also provides a fully offline command using its committed small records, fourteen figure families and independent output/lineage audits.

## FineWeb cycle-10 endpoints

```powershell
python -m scripts.analysis.build_fineweb_endpoint_summary --source reports/20261107T214406Z-full-completion64-mixed99p9 --out reports/20261107T214406Z-full-completion64-mixed99p9/generated/reproduce-cycle10-endpoints
```

Verifies the existing publication, selects exactly the seven terminal cycle-10 rows, and plots final perplexity, pooled/within-output entropy, distinct-2/3 and within-output repetition against actual numeric percentages. [The published appendix](../../reports/20261107T214406Z-full-completion64-mixed99p9/analysis/cycle10-fraction-summary/README.md) records explicit zoom ranges and the no-fit, single-seed scope. The output directory must be new; no cloud/model work occurs.

## FineWeb real-only validation

```powershell
python scripts/analysis/build_fresh_real_validation.py --out .state/fineweb-report-review
```

Verifies the [published small-record manifest](../../reports/20261106T143315Z-smoke-fresh-real/publication_manifest.json) and regenerates CSV/JSON, five square figures and a short report into a new directory. It uses frozen records only. The [validation report](../../reports/20261106T143315Z-smoke-fresh-real/report.md) distinguishes smoke execution checks from the later full-data stability assessment. Model weights, optimization state, token arrays and original document text remain in private backups.

## Mixed-data acceptance report

Rebuild the 50/50 smoke report and figures from its committed publication:

```powershell
python scripts/analysis/build_mixed_acceptance.py --out reports/20261031T153536Z-smoke-continued/generated/review
```

`--source` can name another copy of that preserved publication. The builder verifies its publication manifest, mixture/selection alignment, final-epoch policy and finite NLL/perplexity arithmetic. It reads historical costs/provider facts from `validation_summary.json`; it does not query today's ledger, cloud or tokenizers. Saved examples are reused without decoding/downloading a model. It explicitly distinguishes recorded historical bulk recovery from checks repeated during this offline rebuild.

This replaces reliance on the original private `.state/build_mixed_acceptance.py` for report construction. That original operational helper is preserved locally as history; its live provider checks and financial reconciliation are dated acceptance evidence, not part of offline publication. The new builder produces a self-contained report/figures/preview and `publication_reproduction.json` in a new destination.

## Week 3 authored report and native-Docs draft

The completed real-only control and its four-fraction continued comparison reproduce offline:

```powershell
python comparisons/continued_real_only_vs_mixtures/analyse.py --out reports/20261102T154833Z-full-continued/generated/audit-review
python scripts/analysis/build_real_only_comparison_report.py --out reports/20261102T154833Z-full-continued/generated/report-review
```

The [comparison definition](../../comparisons/continued_real_only_vs_mixtures/README.md) records the shared start, fixed arrays, full real selection, hardware difference and recovery scope. Analysis checks 44 metric rows, 200 epoch entries and 41 unique probes; the builder produces a separate Markdown/native-Docs draft. Seven square-panel figure families include the requested endpoint-versus-synthetic-fraction summary. Normal reproduction has no cloud/model execution; `--recover` is an explicit pinned private-backup read. Published paths remain protected.

Reproduce the completed 20/80 reset comparison and its three-reset-fraction context offline:

```powershell
python comparisons/mixed_fraction_final5/analyse_reset20.py --out reports/20261101T030306Z-full-mixed-campaign-reset-mixed20/generated/audit-review
python scripts/analysis/build_reset20_report.py --out reports/20261101T030306Z-full-mixed-campaign-reset-mixed20/generated/report-review
```

The audit verifies exact shared S0/S1/D2/M2, both initializer policies, mixtures/selection hashes, final-epoch histories, saved output probes and NLL/PPL arithmetic. It checks old reset publications separately for the descriptive fraction comparison. All six figure families use square panels. The report builder makes a separate Markdown/native-Docs draft without editing Google Docs. Output directories must be new; the published evidence stays unchanged. The explicit `--recover` option uses the privately saved pinned Hub source proof; normal reproduction needs no cloud access.

The three-fraction continued comparison and the larger square 50/50 presentation revision reproduce offline:

```powershell
python comparisons/mixed_fraction_final5/analyse_continued_fractions.py --out reports/20261101T030306Z-full-mixed-campaign-continued-mixed20/generated/audit-review
python scripts/analysis/build_continued_fraction_report.py --out reports/20261101T030306Z-full-mixed-campaign-continued-mixed20/generated/report-review
python scripts/analysis/build_square_reset50_report.py --out reports/20261101T030306Z-full-mixed-campaign-reset-mixed50/generated/square-review
```

These require new destinations and make no Runpod calls. The first rechecks the preserved synthetic-only/50% audit, then adds all ten 20% stages and source-selection hashes; the report builders create drafts without editing Google Docs. The square revision preserves the original report/figure hashes. `square_panel_figures.py` supplies explicitly square plotting boxes; figure families are split across more pages for legibility.

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

- `build_decoding_comparison.py`: offline 44-row greedy/sampling versus historical beam/real analysis, matched scientific contract with declared GPU difference, lineage/loss audit and thirteen square figure families. Optional private arrays enable probe and final-target recomputation. See the [reproduction guide](../../reports/20261110T234038Z-full-completion64-decoding/generated/decoding-20261111/README.md).

- `audit_fineweb_fraction_records.py`: offline independent check of 77 published fraction rows against committed indexed metrics, 70 saved probe arrays and 30 example-loss arrays; exports a provenance map that distinguishes inherited aggregate-table checksums from per-stage checksums. See the [audit](../../reports/20261111_fineweb_fraction_audit/README.md).
