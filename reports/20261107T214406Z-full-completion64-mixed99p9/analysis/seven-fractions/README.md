# Seven-fraction FineWeb completion analysis

The [authored report](../../report.md) and Google Doc pages 67–78 cover 0%, 50%, 80%, 95%, 99%, 99.9% and 100% synthetic continuation targets. They contain seven square figures, three native tables and exact same-prompt examples. The 77 checkpoint rows are pinned; all 70 trained fixed-probe arrays were recovered and their production diversity/formatting metrics recomputed. The complete private raw evidence remains in the ignored generated analysis directory recorded in the delivery receipt.

## Reproduce figures offline

From the repository root, choose a new output directory:

```powershell
python reports/20261107T214406Z-full-completion64-mixed99p9/analysis/seven-fractions/plot.py --out reports/20261107T214406Z-full-completion64-mixed99p9/generated/reproduce-seven-fractions
```

NumPy and Matplotlib are required. This command reads the frozen local metrics and writes only to the new output directory; it never provisions cloud resources or reads credentials. `input_provenance.json` records immutable private snapshot commits, index/file hashes, source paths and the scientific contract. `figure_manifest.json` records analysis environment, revision, script and image hashes. The enclosing publication manifest freezes this publication's exact bytes.

## Interpretation limits

- Primary perplexity scores fixed real continuations; empirical entropy/distinctness describe generated text on a fixed 256-prompt validation probe.
- Pooled and within-output entropy have different scopes. None measures factual accuracy or predictive/logit entropy.
- All conditions share declining cosine LR and continued optimizer state. Ten-cycle flattening is not proof of permanent stability.
- 95/99/99.9% used RTX PRO 4500 Blackwell; the other selected conditions used RTX PRO 4000. Identical pretrained weights generated slightly different baseline probe text across these devices, and both baseline measurements are preserved.
- One seed per condition; no significance claim, universal mixture threshold or new constant-LR experiment.
- This is small-artifact recovery and analysis, not full trained-checkpoint restoration/re-evaluation. All cloud recovery resources remain preserved.
