# Repeated FineWeb chunk: real versus recursive synthetic targets

**Completed:** `20261108T173505Z-full-completion64` finished both ten-epoch arms and stopped automatically on 8 November. The [report](../../reports/20261108T173505Z-full-completion64/report.md) and [completion/delivery receipt](../../docs/validation/2026-11-08_repeated-chunk-completion-analysis-and-doc.json) record all four trajectories, terminal recovery and Google Doc pages 85–109. Do not duplicate or resume this completed execution. The commands below describe the implemented interface, not an instruction to relaunch it.

This comparison extends the completed fresh-chunk real/synthetic experiment with two new conditions. Both reuse the frozen bank's first 10,000 prompt rows for ten one-epoch stages. The real arm always uses the same original continuations; the synthetic arm regenerates all continuations with its own preceding selected checkpoint before each epoch. Both start independently from original pretrained OPT-125M.

All existing scientific settings remain fixed, including continuation-only loss on BOS + 64 prompt + 64 target tokens, seed 42, persistent AdamW/RNG state, batch 128, and the global cosine learning rate from 5e-6 to 1e-6 over 790 updates. Each arm receives 6.4 million scored target-token exposures. The frozen validation/test collections and 256-prompt output probe are unchanged. Each epoch uses the existing seeded training shuffle.

`training_chunk_policy: repeat_fixed` and `fixed_chunk: 1` describe source reuse independently from the stage/epoch number. Omitted policy retains fresh chunks. Every stage records initializer/generator/optimizer identities, source-collection and input hashes. The first real chunk is also scored at baseline and every trained checkpoint as a fixed diagnostic; it is training data, not an extra held-out set.

The parent owns one GPU, a 60 GB working volume, a private S3 bulk-backup store, one ledger allocation and one independent deadline. Arms run sequentially. Smoke uses 256 repeated pairs, three epochs per arm, 64 validation/test pairs and a 64-prompt probe, exercising worker/generation/handoff interruptions and both private restores. Full execution also restores each terminal stage and re-evaluates its test loss before shutdown. Terminal-stage restoration does not establish recovery of every earlier bulk checkpoint.

```powershell
$comparison = "comparisons/fineweb_completion64_repeated_chunk"
python -m model_collapse smoke --config "$comparison/smoke.yaml"
python -m model_collapse launch --config "$comparison/full.yaml" --generations 10
python -m model_collapse status COMPARISON_ID
python -m model_collapse fetch COMPARISON_ID --metadata-only
python -m model_collapse report COMPARISON_ID
```

Execution is explicitly authorized by the [8 November receipt](../../docs/validation/2026-11-08_repeated-chunk-authorization.json), subject to fresh acceptance and live admission. Never duplicate a launched/completed comparison. See [operations](../../docs/operations.md) and [current state](../../docs/current-state.md) for live-check and recovery requirements.

The final analysis compares these new arms with the preserved fresh-real/fresh-synthetic results from `20261106T193918Z-full-completion64`. Falling training loss together with worsening held-out loss supports overfitting; repetition/diversity and real-text degradation describe synthetic feedback. Neither outcome is assumed. One seed, finite training and decreasing learning rate limit interpretation.
