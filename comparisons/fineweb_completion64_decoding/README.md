# FineWeb completion targets: greedy versus pure sampling

Two new 100%-synthetic continued-training conditions compare with the preserved five-beam arm `20261106T193918Z-full-completion64-synthetic`. Each independently starts at original pretrained OPT-125M and processes the identical ten frozen chunks, 10,000 pairs per chunk. Each pair contains BOS, 64 real prompt tokens and 64 generated target tokens. Only the continuation contributes to loss.

| Condition | Training-target decoder | Evaluation-probe decoder |
| --- | --- | --- |
| Historical control | Five deterministic beams | Five deterministic beams |
| Greedy | One beam, no sampling | Five deterministic beams |
| Pure sampling | One beam, temperature 1, top-k 0, top-p 1 | Five deterministic beams |

Each arm generates a complete dataset using its own preceding checkpoint, freezes it, and trains those same weights for one terminal epoch. Model, AdamW, global cosine scheduler and training RNG continue across cycles. There are ten epochs, 790 updates and 6.4 million scored training targets per arm. The historical learning rate remains **5e-6 to 1e-6**. This experiment changes training-target decoding; it is separate from the constant-learning-rate repeated-real control.

Sampling uses an isolated, deterministic seed for each generation batch, derived from seed 42, cycle, batch offset, purpose, prompt hash and generator tensor hash. It preserves the training RNG. Aligned generation shards can be reconstructed after an interruption without changing outputs. A resumed run cannot change its recorded decoding settings or layout. The frozen bank builder and bank bytes are unchanged.

Primary metrics score fixed real held-out continuations: 7,322 validation pairs and 7,151 test pairs. Generated-output statistics use the same frozen 256-prompt, five-beam probe for every condition. Reports distinguish decoder as well as synthetic fraction, and only combine historical trajectories after matching the remaining scientific settings. A one-seed, ten-cycle result does not establish a universal decoding effect or permanent stabilization.

## Execution

```powershell
python -m model_collapse smoke --config comparisons/fineweb_completion64_decoding/smoke.yaml
python -m model_collapse launch --config comparisons/fineweb_completion64_decoding/full.yaml --generations 10
python -m model_collapse status COMPARISON_ID
python -m model_collapse fetch COMPARISON_ID --metadata-only
python -m model_collapse report COMPARISON_ID --reference-comparison 20261106T193918Z-full-completion64
```

The coordinator runs greedy then sampling sequentially on one GPU, with separate optimization states and one owner/watchdog. The user authorized RTX PRO 4500 Blackwell when matching RTX PRO 4000 stock was unavailable, subject to the existing budget. Both new conditions use the same GPU; the historical control's GPU difference is recorded explicitly in comparisons. Private recovery capacity, a current runtime/protocol GPU acceptance and sufficient budget remain admission requirements. The source definition alone is not an acceptance certificate or evidence of launch. See [current state](../../docs/current-state.md) for the latest dated record; live provider reads determine actual status.

## Recorded acceptance

[GPU validation](../../docs/validation/2026-11-10_decoding-setup.json) passed nineteen checks, including six terminal training epochs, interruption recovery, isolated sampling RNG, byte-identical sampling replay, historical five-beam parity, unchanged completed stages and 9.022 GB of private child restorations with exact held-out loss reproduction. The final offline suite passed 665 tests; the recovery-only host adapter has a separate offline regression. Temporary validation Pods and their working volume were removed after verification. Validation trajectories are pipeline tests, not full-data evidence of collapse.

## Recorded completion

Both arms completed as `20261110T234038Z-full-completion64-decoding`. [Results and interpretation](../../reports/20261110T234038Z-full-completion64-decoding/report.md), [reproduction](../../reports/20261110T234038Z-full-completion64-decoding/generated/decoding-20261111/README.md), and [delivery/recovery scope](../../docs/validation/2026-11-11_decoding-completion-analysis-and-doc.json) distinguish the common five-beam evaluation probe from training-target decoding. Do not resume this completed comparison.
