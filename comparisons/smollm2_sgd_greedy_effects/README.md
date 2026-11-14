# Plain SGD, greedy continuations and per-update source effects

This comparison prepares 0%, 99% and 100% synthetic-target conditions using the exact existing SmolLM2-135M checkpoint and native FineWeb completion bank. It is a new training recipe, rather than an optimizer-only ablation of the historical AdamW/five-beam experiment.

[GPU validation and handoff](../../reports/20261113T230319Z-smoke-completion64-mixtures/setup-report.md): all twelve checks passed; selected constant LR **0.01**; the full comparison is now active as `20261114T001352Z-full-completion64-mixtures`. See the [launch record](../../docs/validation/2026-11-14_sgd-greedy-full-launch.json); do not duplicate this active run.

## Scientific protocol

- Independently start every arm from original pretrained M0. Continue its own weights and training RNG over ten fresh 10,000-pair chunks; keep the terminal checkpoint after each one-epoch cycle.
- Every input contains BOS, 64 real prompt tokens and 64 continuation tokens. Only continuation targets are scored.
- Use FP32 parameters, forwards and evaluation, with TF32 disabled. Plain SGD has zero momentum, weight decay and dampening; no scheduler, warm-up or clipping; fused/foreach updates are disabled.
- Keep seed 42, microbatch 32 and effective batch 128. Each full cycle has 79 updates, including a final 16-example update; each arm has 790.
- The 99% condition retains 100 real targets and replaces 9,900 at their original rows using the existing deterministic selection. The synthetic condition replaces all 10,000. Each arm generates from its own preceding checkpoint using greedy decoding and exactly 64 new tokens under the existing length constraints.
- Generate each cycle's synthetic dataset completely before training. The same frozen real validation/test arrays remain outside training.

## Selecting the constant learning rate

Both definitions now pin **constant LR 0.01** and its immutable calibration receipt. Unresolved definitions are rejected before full training. The completed setup screened `1e-4`, `1e-3`, `1e-2` and `1e-1` using 32 real-only updates each, then extends the best two by 32 updates on fresh chunk-2 examples: at most 192 updates. Samples contain 4,096 distinct rows and use `SeedSequence([42, cycle, 7731])`.

Screen ranking uses a fixed 1,024-pair document-stratified validation probe. Final ranking uses the entire validation collection. Eligible candidates require finite losses/gradients, changed weights, training-probe NLL improvement over 1e-5, and held-out perplexity no more than 5% above the pretrained baseline. Ties within 1e-5 NLL favor the smaller rate. Test scores and synthetic outcomes never select the rate. The search never expands automatically.

The selected receipt locked the numeric rate before acceptance; subsequent smoke invocations reuse its verified numerical evidence without repeating calibration. Every smoke/full arm restarts M0; calibration checkpoints are not production initializers. Temporary candidate tensors live in hidden calibration workspaces, while their numerical records, selections, hashes and the receipt remain recoverable. The result is the best tested short-screen rate, not a guarantee of long-term stability or a global optimum.

## Update-level measurements

At each actual effective-batch update, directly measure weighted real and synthetic gradients at the same weights/forward graphs and verify their sum reconstructs the authoritative combined gradient. Record source counts/losses, global/per-layer norms, alignment and update size. A missing source contributes a zero vector and has no defined source-mean loss/alignment.

A fixed 128-pair real-validation probe evaluates unchanged, real-component, synthetic-component and actual mixed updates on scratch weights. Components retain their original batch weighting. `mixed NLL - synthetic-only NLL` measures the real component's immediate conditional effect. The analogous synthetic effect and a nonlinear interaction are also recorded. Scratch interventions preserve training mode and RNG and never supply held-out gradients to training.

Completed journals and per-example probe scores are sealed with checkpoints. An unfinished cycle replays from its verified starting state; abandoned attempts are excluded. Whole gradient tensors are not saved. Epoch evaluation retains full real validation/test perplexity, training NLL, greedy output diversity/repetition and secondary evaluations. Repeated probe measurements are not independent training-seed replicates.

## Setup and later launch

```powershell
$comparison = "comparisons/smollm2_sgd_greedy_effects"
python -m model_collapse smoke --config "$comparison/smoke.yaml"

# Recorded launch command; do not duplicate the active comparison:
python -m model_collapse launch --config "$comparison/full.yaml" --generations 10
```

Smoke uses all three arms, two fresh 400-pair chunks each, 128 held-out/update-probe pairs and 32 output prompts. The 99% mixture has 396 synthetic and four real rows per cycle. Four updates per stage include a 16-example tail; 24 logical training updates complete, with physical recovery replay counted separately.

Setup uses one matching RTX PRO 4000 Blackwell, the inspected historical image and one temporary 30 GB Standard volume. All new calibration/acceptance attempts are limited together to USD 0.40 and 30 paid minutes, including bootstrap/restoration. Existing USD 110 cumulative, USD 5.40 historical setup and USD 2 reserve limits remain binding. Failed admission or incomplete verification leaves readiness blocked; no limits are silently increased. No full working/backup storage is provisioned during setup.

The future full comparison runs real, mixed99 and synthetic sequentially with one owner/watchdog, 60 GB working storage and at least 30 GB private backup capacity. Setup stopped before launch; the subsequent user-authorized full run is recorded above. Existing operational status/fetch/report/resume/recovery-gated cleanup commands remain available.
