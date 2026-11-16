# Whole-epoch source removal from the 99% SGD trajectory

Start independent copies from the completed SmolLM2 SGD/greedy checkpoints **M2, M5 and M9**, then replay cycles **3, 6 and 10** using both sources, real only, or synthetic only. Each case uses its exact historical 100 real / 9,900 synthetic targets. No training targets are regenerated and no new ten-cycle experiment is created.

## Scientific contract

- Use the historical continued weights, SGD state and training RNG, FP32/TF32-off arithmetic, constant LR 0.01, microbatch 32 and effective batch 128. There is no momentum, decay, clipping, warm-up or scheduler.
- Keep BOS + 64 prompt + 64 continuation positions, prompt attention and continuation-only scoring. Remove source loss terms without changing the original whole-batch denominator. A real-free batch leaves real-only weights unchanged.
- Preserve the actual 78 full batches and final 16-row batch. Nine branches have 711 scheduled update positions, some with zero gradients.
- Branches evolve independently. These are accumulated epoch interventions, not connected copies of the original one-update shadows and not renormalized 100%-real training runs.
- Both-source replay must pass first for each case: terminal relative tensor error, maximum update-probe NLL discrepancy and complete validation/test NLL discrepancies must each be at most `1e-5`. Record bitwise equality separately.
- Measure the original 128-pair real-validation probe after every update; complete held-out endpoints use 7,447 validation and 7,303 test pairs. A probe metric is not the complete-test metric.

The pinned source is `20261114T001352Z-full-completion64-mixtures`, arm `mixed99`. The configuration records immutable repository commits/index hashes. CPU preparation verifies target reconstruction, generator/initializer/optimizer lineage and unchanged held-out inputs. Bulk weights remain cloud-side.

## Acceptance and automatic launch

```powershell
python -m model_collapse diagnose-epoch-branches --config comparisons/smollm2_sgd_epoch_ablation/diagnostic.yaml --preflight-only

# Bounded acceptance, then all nine full branches automatically:
python -m model_collapse diagnose-epoch-branches --config comparisons/smollm2_sgd_epoch_ablation/diagnostic.yaml

python -m model_collapse status DIAGNOSTIC_ID
python -m model_collapse verify-detached DIAGNOSTIC_ID
python -m model_collapse fetch DIAGNOSTIC_ID --metadata-only
python -m model_collapse report DIAGNOSTIC_ID --out NEW_OUTPUT_DIRECTORY
python -m model_collapse resume DIAGNOSTIC_ID
python -m model_collapse cleanup DIAGNOSTIC_ID
```

The miniature test starts from M5 and uses original cycle-6 slots 1, 5 and 79: 272 rows, six real and 266 synthetic. Three branches execute nine update positions, testing a mixed full batch, a source-empty full batch and the actual tail. Its 32-pair probe is validation evidence, not a full scientific trajectory. One representative private model is cold-restored and re-evaluated; the production 128-pair probe is benchmarked before full admission.

The full workflow uses one matching RTX PRO 4000 Blackwell and historical image/packages. It owns one 30 GB working volume, one 10 GB private backup volume, one detached supervisor and one independent stdlib-only watchdog. GPU acceptance is limited to $0.15 / 15 paid minutes; all attempts and execution share $4 / 90 paid minutes including a $2 reserve. The cumulative/setup ceilings are $130 / $6.10. See the [authorization](../../docs/validation/2026-11-16_epoch-ablation-authorization.json). Credits are not purchased, payment settings are not changed and historical resources are retained.

Each branch is sealed and privately backed up before advancing. Failed replay or budget exhaustion stops the sequence. Interrupted branches replay from their verified initializers; completed branches remain unchanged. Completed remote runs are rejected before resume provisioning. Recovery storage is removed only after complete checksum-verified retrieval.

## Outputs and interpretation

Square cycle-3/6/10 NLL and perplexity figures show coloured mixed replay, dashed grey real-only and dotted grey synthetic-only paths, with historical endpoints marked separately. An overview joins complete-test fork endpoints; its straight segments are not measured within-epoch test curves. CSV exports include all per-update source losses, original denominators, applied gradient norms, endpoints and parameter movement.

The engine's historical `train_loss` remains ordinary mixed-forward loss and is labelled accordingly; `applied_objective_loss` separately records the selected loss over the original token denominator. Only the selected source contributes gradients. Evaluation uses no held-out gradients and preserves training RNG and model mode.

Effects along diverging paths need not add. Cycle 6 is deliberately selected for its correction, and these three one-seed cases do not establish permanent stability. Reports preserve input/runtime/environment/output hashes and distinguish source recovery from GPU acceptance and full completion.
