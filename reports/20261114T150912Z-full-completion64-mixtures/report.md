# SmolLM2 SGD: how real and synthetic updates balance

Both ten-cycle runs completed. Final fixed real-test perplexity is **18.586 at 50% synthetic** and **19.707 at 80%**, from pretrained **17.958**. The matched real-only, 99% and 100% controls end at 17.726, 164.413 and 1,408.607. The 80% trajectory overshoots then partly recovers; finite-horizon apparent stability does not establish a permanent equilibrium.

At cycle 10, real examples supply **93.3% of mean batch loss in 50/50** and **87.5% in the 80%-synthetic condition**. Directly measured source gradients oppose in every new update. Real inclusion improves the fixed held-out probe and synthetic inclusion worsens it in all 790 updates per arm. The conditional effects have a substantial nonlinear interaction: symmetric local allocations provide an explicit additive accounting, with near-balanced late effects. Scalar loss share, gradient geometry and immediate held-out effects are separate measurements.

- [Complete authored analysis](generated/authored-source-effects-20261114-v3/section.md): methodology, equations, ten square figures, seven tables, full/tail qualifications and limitations.
- [Google Doc](https://docs.google.com/document/d/1WyQia9SBCOtT9_X4X56BpTihz-X9q9LWZFVapFEB4uk/edit): appended pages 170–183; all previous 169 pages and native content preserved.
- [Per-update CSV](generated/source-effects-20261114-v2/per_update.csv), [per-cycle source accounting](generated/source-effects-20261114-v2/cycle_source_summary.csv), [held-out effects and document intervals](generated/source-effects-20261114-v2/cycle_effects.csv), [early/late and full/tail comparisons](generated/source-effects-20261114-v2/phase_summary.csv).
- [Full test trajectory](generated/source-effects-20261114-v2/test_trajectory.csv), [output statistics](generated/source-effects-20261114-v2/output_metrics.csv), [tail records](generated/source-effects-20261114-v2/tail_batches.csv).
- [Scientific/runtime compatibility](generated/source-effects-20261114-v2/compatibility.json), [numerical checks](generated/source-effects-20261114-v2/verification.json), [analysis provenance and source hashes](generated/source-effects-20261114-v2/analysis_manifest.json), [document preservation](generated/authored-source-effects-20261114-v3/document_verification.json).

The final 16-example update contributes about 75% of cycle-10 probe improvement at 80%; late tails worsen predictions on average, while late full batches still show near balance. Generated-output repetition remains higher than real-only despite much better likelihood than the high-synthetic controls. This is one seed, one model/corpus and a 128-pair repeatedly measured diagnostic probe; bootstrap bands describe document-sample variability, not seed uncertainty.

Recovery covers 1,580 new sealed update records/probe arrays and 2,370 historical controls. Retained-volume bytes were checked against sealed private-backup indexes, including all generator/initializer/optimizer chains and deterministic source selections. The 21 unique new generated probe arrays were remeasured. No new training/inference or full bulk checkpoint cold restoration was performed. Provider shutdown is recorded at 14 November 2026 20:45:56 UTC; working and backup volumes remain preserved.

## Reproduce the offline analysis

Run from the repository root with authorized private recovery access and the historical diagnostic workspace/source archives retained. Output destinations must be new.

```powershell
python -m comparisons.smollm2_sgd_greedy_mixtures.recover_updates
python -m comparisons.smollm2_sgd_greedy_mixtures.analyse_updates --out reports/20261114T150912Z-full-completion64-mixtures/generated/NEW_ANALYSIS_ID
python scripts/analysis/compose_sgd_mixture_report.py --source reports/20261114T150912Z-full-completion64-mixtures/generated/NEW_ANALYSIS_ID --out reports/20261114T150912Z-full-completion64-mixtures/generated/NEW_AUTHORED_ID
```

These commands read retained evidence and use CPU analysis only; they do not provision resources. Model tensors, optimization states, raw diagnostic arrays and credentials remain private. The publication manifest covers only the small explicitly published files, not unpublished local drafts.
