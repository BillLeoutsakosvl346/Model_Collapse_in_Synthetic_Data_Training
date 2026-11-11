# Repeated real text with a constant learning rate

The user requested this control on 10 November 2026 to examine ordinary overfitting without the decreasing learning rate used in the earlier FineWeb experiments.

- Start independently from the identical pinned original pretrained OPT-125M.
- Continue the same weights, AdamW state, scheduler and training RNG for **20 epochs** on the exact **first 10,000 real prompt/continuation pairs**. Never reset between epochs. Shuffle training order using the existing seeded policy.
- Keep BOS + 64 prompt tokens + 64 real target tokens, with loss on the 64 targets only. This gives **12,800,000 scored target-token exposures and 1,580 optimizer updates**.
- Keep the existing fixed real validation/test collections and 256-prompt generated-text probe. Record loss/perplexity, training-anchor loss, diversity/repetition, learning rate and operational statistics at baseline and after every epoch.
- Set both `learning_rate` and `min_learning_rate` to **5e-6**. The unchanged shared engine uses CosineAnnealingLR with equal upper/lower rates, which is exactly constant at every update. Smoke acceptance and the full audit verify the saved rates.
- Retain the frozen ten-chunk bank (`full_cycles: 10`) byte-for-byte. The separate `training_cycles: 20` setting defines the longer optimization horizon. `repeat_fixed` always selects `cycle_01`.
- Run only the `real` endpoint. No synthetic training datasets are generated. Generated evaluation probes remain the same as in historical runs.

The previous matching RTX PRO 4000 was unavailable at live preflight. After five reconciled placement failures on RTX PRO 4500 / A6000, validation obtained an RTX 2000 Ada at the observed USD 0.24/hour, with the same image, PyTorch, precision and scientific settings. Full execution must use the hardware that passes fresh acceptance. Record this execution difference when comparing with the historical ten-epoch cosine-LR control.

This single endpoint uses a 60 GB persistent Pod disk and a separate 60 GB Standard S3 backup volume. Both survive GPU shutdown; terminating the Pod deletes its working disk. At the conservative recorded rates, retaining both costs approximately USD 0.53/day, so the USD 2 storage reserve covers about 3.75 days, not seven days. Historical resources remain untouched.

The smoke uses three epochs on the same 256 real pairs, 64 validation/test pairs and 64 probe prompts, including interruption recovery, full private checkpoint/optimizer restoration, repeated evaluation and actual watchdog shutdown. Full execution requires fresh acceptance for this runtime/protocol.

```powershell
$condition = "comparisons/fineweb_completion64_repeated_real_constant20"
python -m model_collapse smoke --config "$condition/smoke.yaml"
python -m model_collapse launch --config "$condition/full.yaml"
```

The named full configuration specifies twenty training cycles; no CLI generation override is needed. Do not duplicate an active or completed run. Consult current state and its completion receipt for the execution ID. Historical results and their recovery resources remain preserved.
