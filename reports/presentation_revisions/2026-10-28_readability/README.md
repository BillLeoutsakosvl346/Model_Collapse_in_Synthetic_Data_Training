# Google Doc presentation revision, 28 October 2026

The live project report now presents the original Week 2 continued-training
experiment on its own. Its figure shows only that condition. The subsequent
shared-S0/S1 comparison remains below it with both reset and continued curves.
Week 1 and both Week 2 sections use short introductions and labelled native
bullets for methodology, findings and limitations.

All 11 native tables, their measurements and styles, and the other three graphs
were preserved. The original four-week plan is unchanged. The exported report
was rendered and visually reviewed at 13 pages.

This directory contains the replacement figure, its input/source/environment
manifest and the native-document delivery check. Original dated reports, CSVs,
figures, artifact indexes and prior delivery receipts remain unchanged.

Reproduce the standalone figure into a new path:

```powershell
python scripts/analysis/continued_only_figure.py --out analysis/generated/continued-only-review.png
```

The figure uses the frozen continued-run metrics and saved training histories;
it does not load model checkpoints, launch compute or alter cloud resources.
