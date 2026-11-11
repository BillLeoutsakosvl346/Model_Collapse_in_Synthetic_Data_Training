# Final metrics versus synthetic percentage

Six square figures show the cycle-10 real-test perplexity, pooled token entropy, mean within-output entropy, distinct-2, distinct-3 and mean within-output repeated four-gram rate. Fractions are 0%, 50%, 80%, 95%, 99%, 99.9% and 100%, matching the authored seven-fraction comparison. All points retain actual numeric percentage spacing. Insets expand dense conditions; dashed joins are visual guides rather than fits.

- [Explanations](report.md)
- [Seven endpoint rows](cycle10_metrics.csv)
- [Input, environment, script and output hashes](publication_reproduction.json)

Reproduce into a new directory from the repository root:

```powershell
python -m scripts.analysis.build_fineweb_endpoint_summary --source reports/20261107T214406Z-full-completion64-mixed99p9 --out reports/20261107T214406Z-full-completion64-mixed99p9/generated/reproduce-cycle10-endpoints
```

This is offline analysis of existing measurements. No Runpod, training, checkpoint selection or new evaluation occurs. The original report and publication hashes remain unchanged. Single-seed, learning-rate-schedule, hardware and metric-scope limitations remain as recorded in the main report.
