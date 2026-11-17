# When a model becomes its own teacher

[Open the Google Slides presentation](https://docs.google.com/presentation/d/1Y4hktQamSbHQcP5rRulqlTvYoPv051ROcUR7HMEA_vM/edit).

[Verified PDF snapshot](presentation.pdf) includes the main talk and all backup pages.

This edition rebuilds the earlier presentation around the mechanism investigated in Week 5. The main talk is paced at **9 minutes 35 seconds**, leaving a little room within ten minutes. Its **21 visible slides include successive graph builds**; **11 skipped backup slides** preserve discussion material.

## Main story

1. Synthetic feedback can degrade predictions, while real-only overfitting can also raise perplexity. Earlier OPT controls motivate matched fresh-prompt experiments.
2. SmolLM2-135M receives the same ten FineWeb chunks, with real or self-generated continuations. Plain SGD uses constant LR 0.01, FP32 arithmetic and greedy generation.
3. Click-through builds add 0%, 100%, 99%, 80% and 50% synthetic curves. Axes and placement stay fixed. The main vertical series is explicitly **log10(test perplexity)**; raw endpoint PPL is shown beside it. The linear-scale detail is in backup B3.
4. In the 99% arm, real rows supply **49.1% of mean batch loss by cycle 10**, despite being 1% of examples. Corpus-token weighting gives 47.5%. This scalar loss share is distinct from gradient strength or generalization benefit.
5. Measured temporary source-removal updates show directional held-out effects. Real inclusion helps all 563 updates containing real rows in that trajectory; synthetic inclusion worsens 778 of 790.
6. Whole-epoch forks at cycles 3, 6 and 10 remove each source at its original strength. Saved targets, batch order and denominators remain fixed. The copies then evolve independently. The cycle-3 graph reveals the additional branches on successive clicks; all three cases retain common axes.
7. Output entropy/repetition provide another view. The conclusion retains the one-seed, finite-horizon and final-batch qualifications.

## Presenter use

Start at slide 1 and advance normally. Curve builds are consecutive native slides, so they need no animation controls. Google Slides skips B1–B11 during the main slideshow. Speaker notes contain pacing, methodology, interpretation and references.

The source-ablation panels show a **128-pair real probe**, while the fraction trajectories use the **complete fixed test**. Backup B10 gives full-test ablation endpoints. A sharp final correction can involve one real example in a 16-example batch; that sensitivity is visible in the main talk and explained in B6.

## Reproducible evidence

- `data.json`: every plotted numerical series, including the log10 transformation and arithmetic-mean reference.
- `manifest.json`: original source identities and input checksums; all historical CSVs remain unchanged.
- `outline.json`: narrative roles, timing, exemplar choices and media replacement maps.
- `notes.json`: native speaker-note text.
- `exemplars.json`: compact archival native design reference, with transient media URLs and old notes removed.
- `verification.json`: final native revision, PDF checksum, independent evidence review and visual checks.

The [builder](../../scripts/presentations/build_research_update.py) generates data and native Sheets/Slides requests offline:

```powershell
python scripts/presentations/build_research_update.py `
  --template-json presentations/2026-11-17_research-update/exemplars.json `
  --sheet-id 10TD0TdcnURG4kpOmOAyOFp9DnAdmhC_GpWETOpLCky0 `
  --out .state/presentation-rebuild
```

This command makes no API calls. Request application is a separate authoring operation on a working copy containing the named native exemplars. The delivered deck was edited in place, using a private copy of the earlier editable backing spreadsheet. All 21 new charts remain native linked Sheets charts; text, equations and tables remain native Slides objects.

Four research-history/evidence agents and one post-code reviewer informed the edit. The final reviewer independently checked 652 plotted values and all eleven input hashes. All 32 final pages were exported and inspected after one consolidated repair pass; the structural checker reports zero issues. Local PDF rendering used PyMuPDF because Poppler was unavailable. The export helper retained PDF checksums and slide-order/page-count validation.

No new training, inference, cloud resource changes or Google Doc changes were made for this presentation. The main [research report](https://docs.google.com/document/d/1WyQia9SBCOtT9_X4X56BpTihz-X9q9LWZFVapFEB4uk/edit) remains the longer account.
