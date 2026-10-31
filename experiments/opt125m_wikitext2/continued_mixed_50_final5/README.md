# Continued weights with 50% real / 50% synthetic targets

**Full run launched on 31 October at 16:34 UTC/London:** `20261031T163428Z-full-continued`; [launch receipt](../../../docs/validation/2026-10-31_mixed-50-full-launch.json). Startup/first-epoch progress were verified, not full completion. Use [current state](../../../docs/current-state.md) before further cloud work; do not launch a duplicate. The earlier [acceptance](../../../reports/20261031T153536Z-smoke-continued/report.md) passed sixteen GPU/provider checks after 162 offline tests and independent review, with a later test-only admission regression. Historical acceptance-date budget/status statements are preserved in that report.

The experiment imports the exact historical real-trained S0 and its pinned D1. At cycle 1, S0 trains on a new mixture; historical synthetic-trained S1 checkpoints are never reused. Each following selected model generates a full synthetic pool and initializes the next training cycle. AdamW and the cosine scheduler restart each cycle. All five epochs execute; epoch 5 is retained regardless of validation loss.

Full mixtures contain 37,946 independent 64-token blocks: 18,973 real WikiText training blocks and 18,973 continuation-only synthetic blocks (1,214,272 tokens from each source). Generation uses the unchanged fixed training prompts and 64-token continuations, five deterministic beams, and repetition penalty 1.0. Validation/test blocks and the first 256 evaluation prompts remain fixed.

`synthetic_fraction` controls block sampling; omitted or 1.0 preserves historical all-synthetic behaviour. For other values, counts round half upwards and real blocks fill the remainder. Each cycle samples without replacement within each source using three PCG64 streams derived from seed, cycle and stream number (real selection, synthetic selection, shuffle). Real examples may recur across cycles. The mixture is fixed throughout all five epochs and recovery. Source indices, shuffled source labels, permutation, source file/content hashes and generator identity permit exact reconstruction. Real blocks come exclusively from the prepared training split. This fixed-total-block design adapts the paper's question rather than exactly reproducing its fixed-real-count theoretical setup.

Synthetic pools stay in `synthetic_for_gen_XX/`; actual training mixtures are separate in `mixed_for_gen_XX/`. A completion manifest is committed after tokens and selection data. Missing/corrupt completed mixtures, changed source identities, fractions or lineage are rejected. The final cycle generates only the 256-output metric probe, without a full unused D11.

## Commands

The small validation profile (256 mixed training blocks, 64 validation/test blocks, two cycles, ten epochs) is documented below. Repeating it is paid and requires separate authorization; it is not needed for documentation changes:

```powershell
python -m model_collapse smoke --config experiments/opt125m_wikitext2/continued_mixed_50_final5/smoke.yaml
python -m model_collapse verify-detached RUN_ID
python -m model_collapse status RUN_ID
python -m model_collapse fetch RUN_ID --metadata-only
python -m model_collapse validate RUN_ID
python -m model_collapse report RUN_ID
python -m model_collapse cleanup RUN_ID
```

Run `verify-detached` while the validation worker is active. Validation intentionally exits after a generation shard and after completed cycle 1; the supervisor restarts it. Private backup restoration and checkpoint re-evaluation occur on the GPU before the independent watchdog exercises actual shutdown. Cleanup requires verified recovery. Metadata-only fetching keeps bulk checkpoints and token arrays in the cloud.

Full configuration interface, already used by the authorized launch; **do not repeat for the active run**:

```powershell
python -m model_collapse launch --config experiments/opt125m_wikitext2/continued_mixed_50_final5/full.yaml --generations 10
```

This performs ten new training stages / fifty epochs, imports D1, generates nine new full pools and a final metric probe. Only after detached supervision and progress are verified can the laptop close. `resume RUN_ID` uses the frozen configuration and is only for unfinished, stopped runs; do not resume a running or completed experiment. Launch requires a matching runtime/protocol acceptance certificate and matching RTX PRO 4000 Blackwell/software, plus the existing budget checks.

Report metrics preserve previous definitions and add the actual real/synthetic block/token counts. Frozen-S0 scores are continuation-only predictability measurements, not output quality or model-distance measures. Smoke results validate the pipeline; they do not establish collapse or mitigation effectiveness.

Acceptance attempts shared a $0.75 campaign ceiling within the original $3 setup allowance, then-current $25 task ceiling and $2 reserve; the smoke session capped at $0.65. The later explicitly authorized full launch applied a $30 cumulative ceiling, retaining the $2 reserve and existing credit only. The accepted local launcher's historical default and explicit operational override are documented in [operations](../../../docs/operations.md). No payment settings were changed. Technical acceptance alone does not establish affordability or authorize a rerun.
