# Four-condition FineWeb completion comparison

[Authored report](../../report.md): fresh/repeated prompts crossed with real/recursive synthetic continuations. The 44 logical checkpoint rows produce fourteen square figure families. Both new arms continue weights and optimizer, never reset. Synthetic targets are regenerated from the preceding checkpoint every epoch.

## Reproduce offline

From the repository root, with the project installed, choose a new output directory:

```powershell
python -m scripts.analysis.build_repeated_chunk_comparison --root reports/20261108T173505Z-full-completion64/records/20261108T173505Z-full-completion64 --reference reports/20261108T173505Z-full-completion64/records/20261106T193918Z-full-completion64 --out reports/20261108T173505Z-full-completion64/generated/reproduce-four-conditions
```

This reads the committed small records and never contacts cloud services, trains or reads credentials. CSV/JSON, all PNGs and PDFs reproduced byte-for-byte in this recorded environment (33 files). Input paths/bytes, script revision/hash, environment and output hashes are in `analysis_manifest.json`; the publication manifest freezes the final bundle. Historical reports and raw schemas remain unchanged.

## Verification and scope

- `lineage_audit.json`: twenty new stages, continued model/optimizer chains, unchanged repeated prompts, actual regenerated synthetic targets, 790 updates and 6.4 million target exposures per arm. Saved example losses independently reproduce corpus NLL/PPL and median/P95.
- `probe_audit.json`: 42 stored 256-by-64 token arrays represent 44 logical curve rows, because baselines are shared within each comparison. Saved diversity/repetition/formatting statistics recomputed within 1e-12. Some stored arrays share identical contents across the matched first steps.
- `target_stability.json`: all ten regenerated 10,000-by-64 training-target pools and their generator identities were verified; assembled input tokens, attention and continuation masks were reconstructed and their saved file hashes matched. For adjacent aligned arrays, token agreement is `mean(Tk == Tprev)` and whole-answer agreement is `mean(all(Tk == Tprev, axis=1))`. These are training-pool diagnostics, not held-out quality scores.
- Both new terminal stages were independently cold-restored privately on the GPU: 3,037,798,371 bytes including models and optimization state, with zero reproduced test-loss error. Other historical bulk checkpoints were not independently restored by this task.
- Primary PPL scores 457,664 fixed real continuation targets; secondary 512-token full-text evaluations are separate. Pooled entropy, local entropy, phrase variety and local repetition are different scopes, not accuracy scores.

The bank, full checkpoints, optimizer states and bulk arrays remain private on retained recovery storage. The `records/` tree contains 454 byte-preserved small JSON records (about 10.7 MB), including the complete per-example test losses, configuration and provenance. Authoritative resource controls and credentials are excluded. [Operations and cost](../../operations.json) distinguish lagging billed amounts from conservative estimates.
