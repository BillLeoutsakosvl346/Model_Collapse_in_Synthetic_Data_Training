# Decoder comparison: offline analysis

The [authored report](../../report.md) compares new greedy and pure-sampling recursive targets with historical five-beam synthetic targets and a historical real-only reference. Forty-four checkpoint rows cover baseline plus ten continued cycles in each condition; thirteen square figure families are supplied, nine used in Google Docs.

## Reproduce without cloud access

From the project root, choose a new, empty generated destination:

```powershell
python -m scripts.analysis.build_decoding_comparison --records reports/20261110T234038Z-full-completion64-decoding/records --out reports/20261110T234038Z-full-completion64-decoding/generated/reproduce-decoding
```

This command never trains, runs inference, provisions resources or reads credentials. The 31 core metrics/example/summary/figure outputs reproduced byte-for-byte in the recorded environment. To additionally recompute output diagnostics from private arrays, provide `--bulk PRIVATE_RECOVERY_ROOT --bank data/fineweb_completion64`; the accepted directory layout follows the pinned per-run indexes. Do not put bulk arrays in Git.

## Evidence and scope

- `metrics.csv` / `metrics.json`: all 44 logical checkpoints, primary/secondary likelihood, local/pooled output statistics, steps, LR, training time and peak memory.
- `audit.json`: source-index hashes, continued initializer/generator/optimizer chains for all forty training stages (twenty new, twenty historical), 64 scored targets per example and direct arithmetic from saved example NLLs.
- All 44 logical probe rows were recomputed from 42 unique-stage arrays. Each comparison shares its baseline probe between children. Statistics agree within numerical tolerance; the historical and new baseline probe outputs can differ slightly across GPUs despite identical initial tensor hashes.
- `training_target_statistics.json`: all three synthetic cycle-10 target pools were checksum-recovered and their actual target-content hashes verified. Their generator is M9. The real comparator uses original chunk-10 targets. These 10,000-output populations are distinct from 256-output checkpoint probes.
- All generated-output trajectory curves use the common five-beam probe, irrespective of the training-target decoder. Sampling targets are pure temperature-1 draws with top-k=0 and top-p=1. The shared LR remains cosine 5e-6 to 1e-6.
- New arms used RTX PRO 4500 Blackwell; historical real/beam used RTX PRO 4000. Remaining comparison-contract fields match. The saved 44-row comparison is one seed, not a significance test or a hardware-controlled beam contrast.
- Twenty historical nested probe event logs were absent from their old index. Scientific metrics, probes and the referenced model/state checksums are available; this omission is preserved in the audit and never claimed as full historical seal restoration.
- This task recovered metadata/probes/final target pools, not full models or optimizer states, and performed no new GPU evaluation. Earlier GPU acceptance cold-restoration evidence is separate. Full-run checkpoints remain in private pinned Hub/S3 backups and retained volumes.

Input hashes, the analysis script/revision, local environment and output hashes are recorded in `analysis_manifest.json`. Historical source records retain their original schemas and bytes. The publication manifest freezes this new bundle.
