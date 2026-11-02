# Reports and preserved evidence

Full experiments measure scientific trajectories; smoke experiments validate the pipeline. All full conditions currently use one seed. A report's recorded recovery scope is not proof of live cloud state or complete bulk restoration. See [current state](../docs/current-state.md) for the latest recorded execution.

## Full experiments

| Run / presentation | Condition and checkpoint policy | Evidence/interpretation | Recovery recorded |
| --- | --- | --- | --- |
| `20261020T211121Z-full` / Week 1 | Original pretrained reset; five executed epochs, epoch 1 selected | [Report](20261020T211121Z-full/report.md), [metrics](20261020T211121Z-full/metrics.csv) | Metadata audited; bulk backups private |
| `20261026T192740Z-full-continued` / Week 2 | Continued weights; five executed epochs, epoch 1 selected | [Report](20261026T192740Z-full-continued/report.md), [consistency audit](20261026T192740Z-full-continued/consistency_audit.json) | Metadata plus eleven probe arrays; full-run bulk recovery separate |
| `20261027T211852Z-full-reset-s0` / matched Week 2 extension | Exact historical S0/S1/D2; reset cycles 2-10 to S0; epoch 1 selected | [Matched report](20261027T211852Z-full-reset-s0/report.md), [analysis](20261027T211852Z-full-reset-s0/analysis/) | Metadata plus nine new probes; historical shared probes reused |
| `20261030T180755Z-full-paired-final5` / Week 3 | New shared epoch-5 S1/D2; continued and reset arms retain epoch 5 | [Report](20261030T180755Z-full-paired-final5/report.md), [independent audit](20261030T180755Z-full-paired-final5/analysis/audit/audit.json), [derived analysis](20261030T180755Z-full-paired-final5/analysis/results/) | Selected metadata and twenty unique probes; full-run bulk restoration separate |
| `20261031T163428Z-full-continued` / 50/50 mixture | Continued weights; half real/half synthetic targets; retain epoch 5 | [Matched synthetic-only report](20261031T163428Z-full-continued/report.md), [analysis](20261031T163428Z-full-continued/analysis/), [delivery receipt](../docs/validation/2026-11-01_mixed-50-analysis-and-doc.json) | All metadata and ten new probes verified; shared S0/comparator evidence reused; bulk recovery separate |
| `20261101T030306Z-full-mixed-campaign-reset-mixed50` / matched 50/50 reset | Exact mixed S0/S1/D2/M2; reset cycles 2–10 to S0; retain epoch 5 | [Matched continued/reset report](20261101T030306Z-full-mixed-campaign-reset-mixed50/report.md), [analysis](20261101T030306Z-full-mixed-campaign-reset-mixed50/analysis/), [delivery receipt](../docs/validation/2026-11-01_mixed50-reset-analysis-and-doc.json) | 1,421 metadata files and nine new probes verified; shared S0/S1 reused; bulk recovery separate. This child is complete; no parent-campaign completion claim. |
| `20261101T030306Z-full-mixed-campaign-continued-mixed20` / 20% synthetic continued | Exact S0/D1; 80% real / 20% synthetic; fifty epochs, retain epoch 5 | [Three-fraction report](20261101T030306Z-full-mixed-campaign-continued-mixed20/report.md), [analysis](20261101T030306Z-full-mixed-campaign-continued-mixed20/analysis/), [delivery receipt](../docs/validation/2026-11-01_continued20-analysis-and-square-doc.json) | 1,560 metadata files and ten new probes verified; source checkpoints/full arrays not restored. Child completion does not establish parent completion. |
| `20261101T030306Z-full-mixed-campaign-reset-mixed20` / matched 20/80 reset | Exact 20% S0/S1/D2/M2; reset cycles 2–10 to real-trained S0; retain epoch 5 | [Matched report and reset-fraction context](20261101T030306Z-full-mixed-campaign-reset-mixed20/report.md), [analysis](20261101T030306Z-full-mixed-campaign-reset-mixed20/analysis/), [delivery receipt](../docs/validation/2026-11-02_reset20-analysis-and-doc.json) | 1,421 metadata files / 53,710,269 bytes and nine new probes verified; shared S0/S1 reused. Full checkpoint/training-array restoration remains separate. |

The Week 3 pair's logical arms are `<pair-id>-continued` and `<pair-id>-reset-s0`; use the parent ID for cloud operations. Shared measurements are reused evidence, not independent replicate runs. Historical cross-GPU comparisons differ from the later explicitly matched-environment comparisons.

## GPU validation

| Acceptance | Scope | Report and recovery |
| --- | --- | --- |
| `20261020T193206Z-smoke` | Original G0-G2 setup | [Week 1 validation](week1_validation.md); historical bulk restore/re-evaluation |
| `20261026T182856Z-smoke-continued` | Two continued training cycles and recovery | [Report](20261026T182856Z-smoke-continued/report.md); remote bulk restore/re-evaluation |
| `20261027T202525Z-smoke-reset-s0` | Earlier-source reset acceptance | [Report](20261027T202525Z-smoke-reset-s0/report.md); superseded source certificate |
| `20261027T203952Z-smoke-reset-s0` | Reset-S0 stages 2/3; one epoch in the smoke profile | [Report](20261027T203952Z-smoke-reset-s0/report.md); remote bulk restore/re-evaluation |
| `20261030T172428Z-smoke-paired-final5` | Four unique stages / twenty actual epochs, paired handoff | [Report](20261030T172428Z-smoke-paired-final5/report.md); both private child restores/re-evaluations |
| `20261031T153536Z-smoke-continued` | Two mixed stages / ten actual epochs | [Report](20261031T153536Z-smoke-continued/report.md), [dataset preview](20261031T153536Z-smoke-continued/dataset_preview.json); both mixtures reconstructed and remote bulk restoration checked |
| `20261101T022210Z-smoke-mixed-campaign` | Three sequential jobs, six stages / thirty actual epochs, matched cycle-2 mixtures | [Report](20261101T022210Z-smoke-mixed-campaign/report.md), [preview](20261101T022210Z-smoke-mixed-campaign/dataset_preview.json), [local reconstruction](20261101T022210Z-smoke-mixed-campaign/local_mixture_reconstruction.json), [setup receipt](../docs/validation/2026-11-01_mixed-campaign-setup.json); three private restores/re-evaluations, detached handoffs and actual watchdog shutdown |

The [real-only acceptance](20261102T152509Z-smoke-continued/report.md) passed two continued stages / ten actual epochs, exact all-real mixtures, interruption recovery, private checkpoint restoration/re-evaluation, detached progress and actual watchdog shutdown on explicitly recorded fallback hardware/storage. Its temporary Pod/disk were removed.

Each acceptance covers its recorded runtime/protocol. It must not certify later code or a different mixture/selection policy. These temporary resources were removed after recorded verification; that does not authorize deleting historical full-run recovery volumes.

The [preliminary mixed-campaign recovery](../docs/validation/2026-11-01_mixed-campaign-preliminary-recovery.json) records an interrupted earlier-runtime smoke and the faults fixed before final acceptance. Its saved stages/backups were recovered and temporary resources removed; it received no campaign acceptance certificate. Smoke measurements validate the pipeline and do not establish a mixed-data scientific result.

## Editorial revisions and navigation

[Presentation revisions](presentation_revisions/) preserve report/Google Doc corrections and delivery evidence; they are not new training experiments. The [28 October correction index](presentation_revisions/2026-10-28_audit-corrections/README.md) explains metric-interpretation and presentation updates. [Original validation JSON](validation.json) and [shutdown probe receipt](shutdown_receipt_fault.json) remain historical supporting evidence.

The [1 November square-panel revision](presentation_revisions/2026-11-01_square-reset50/report.md) enlarges the 50/50 reset figures and reorganizes their pages without replacing the original report, measurements or figures. The newer 20/80 comparison uses the same square-panel layout.

Existing dated paths, CSV schemas, token probes, figures and hashes are preserved. Future publications should use:

```text
reports/<run-id>/
  records/             selected immutable input evidence
  analysis/            frozen published tables, figures and provenance
  report.md            authored interpretation
  generated/<id>/      ignored, reproducible scratch analysis
```

Small `.npy` files already committed here are fixed scientific probes, not bulk training datasets. Do not remove them by extension. New analysis must cite its input hashes and write a new destination. [Analysis tools](../scripts/analysis/README.md) document reproduction without training or cloud access.
