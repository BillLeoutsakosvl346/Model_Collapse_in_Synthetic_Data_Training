# Continued 50/50 mixture: implementation and validation handoff

**Technically validated; full experiment unlaunched. Full-run admission is currently budget-blocked.**

Acceptance `20261031T153536Z-smoke-continued` completed two actual OPT-125M training cycles on 31 October 2026. Each cycle used 128 real + 128 synthetic 64-token blocks, trained five epochs and retained epoch 5. The full configuration has ten cycles, continued weights and 37,946 blocks per mixture (18,973 from each source). It imports exact historical S0/D1 and creates a new mixed-data S1; previous all-synthetic S1 is never reused. AdamW and the scheduler restart each cycle.

## Evidence and checks

- **162 offline tests passed**, including the original suite, exact mixture reconstruction, half-up count rounding, deterministic sampling, source corruption/configuration rejection, two real tiny-OPT mixed cycles and exact epoch-boundary recovery. A further targeted regression passed after GPU acceptance, proving an accepted but unaffordable full configuration is rejected before provisioning. That last addition changes tests only; the accepted runtime is unchanged.
- A fresh independent sub-agent found no material issues; a minor ambiguous continuation-only audit label was corrected before acceptance. [Review receipt](validation/2026-10-31_mixed-50-independent-review.json).
- **All sixteen GPU/provider checks passed:** exact source/mixed dataset lineage, real weight updates, terminal checkpoint reloads, both continued-weight handoffs, fixed evaluation/prompts, two injected worker exits, shard reuse, unchanged completed stages, closed-SSH progress, private backup restoration/re-evaluation, direct NLL arithmetic and actual independent watchdog shutdown.
- A second local reconstruction verified both mixtures from the saved real/synthetic source indices and permutation. Every example has 64 tokens; each mixture contains 8,192 real and 8,192 synthetic tokens. [Selected examples and token IDs](../reports/20261031T153536Z-smoke-continued/dataset_preview.json).
- The GPU restored **118 files / 1,514,647,725 bytes**, including all three checkpoints. Restored S2 evaluation matched exactly; direct shifted-token NLL error was zero. Locally, 93 metadata files / 11,165,147 bytes and 24 small arrays / 560,128 bytes were recovered. No model weights were downloaded to the laptop.
- Provider shutdown was verified at **15:44:29 UTC/London**. Pod `583jaukf1l3fsn` and temporary volume `1dpuewdeox` were deleted. No GPU compute remains active; all four historical recovery volumes remain preserved.

[Validation report and figures](../reports/20261031T153536Z-smoke-continued/report.md) · [Machine-readable receipt](validation/2026-10-31_mixed-50-setup.json) · [Scientific definition](../experiments/opt125m_wikitext2/continued_mixed_50_final5/README.md)

These are pipeline checks using tiny data, not evidence that real-data mixing prevents collapse.

## Runtime identity and outputs

Implementation commit: `ff309fe7911ed2a964d3c4a025d5bdc93047921a`. Its normalized runtime file hashes independently match the GPU-tested runtime.

Accepted runtime SHA-256: `266afa984a2a51b63f850390232a12b3ec98a93975ad27f9401983d1afc060c5`.

Scientific protocol fingerprint: `539b6d8bb4c1afddecf0ebeb1e8b0694a9417cb1c997b0d607ebe6fb065e0c8c`. Full/smoke configurations resolve to the same protocol. Runtime/protocol changes require new acceptance; changing the fraction to 20% is not covered by this 50% certificate.

The cloud source bundle preserves its original base Git commit and exact runtime file hashes. Implementation supplied from the working tree is identified by the runtime hash rather than being misattributed to that base commit. Documentation and test-only additions do not change runtime identity.

Each full cycle saves its selected checkpoint, original synthetic pool, actual mixed training array, source selections/labels/shuffle, content hashes, epoch history, evaluation and fixed-prompt examples. Metrics include the existing pooled measures plus Week 3 within-output repetition/entropy, newline diagnostics and frozen-S0 score distributions. Frozen-S0 scoring remains continuation-only predictability, not a quality or model-distance measure.

## Timing and budget

The smoke completed in **8.23 minutes**, with provider-confirmed shutdown after **8.85 minutes** from Pod creation. Observed account debit through cleanup was **$0.051330**, including concurrent retained storage and provider billing timing; this is not an itemized compute invoice. The conservative campaign estimate was **$0.0976**, within the combined $0.75 acceptance allowance, existing $3 setup allowance and cumulative task ceiling. No credits or payment settings were changed.

Warmed production-batch throughput measured **34,047 training target tokens/s** and **1,306 generated tokens/s** using five beams. Ten cycles require fifty training epochs, nine new full synthetic pools and the final probe; D1 is imported. Historical actual throughput suggests **6-8 hours**. The conservative admission forecast, with 1.5x compute margin and checkpoint/upload allowances, is **9.33 hours / $5.37** at the quoted matching GPU price of $0.57/hour plus running storage. These are forecasts, not a completed mixed-data full run.

The cumulative conservative ledger currently estimates **$20.7854**, leaving about **$2.21 after the $2 reserve** under the unchanged **$25 ceiling**. Account credit does not enlarge that authorization. The full launcher must reject the current unaffordable forecast. Do not change the ceiling, shorten the scientific protocol, choose different hardware or delete historical recovery resources to bypass admission. Recheck verified billing, storage accrual, hardware availability and budget on any later launch instruction.

## Future launch and operations

The full command below is prepared and its acceptance/profile/admission paths are tested. **It has not been executed and will currently be refused by the budget guard.** A later user instruction must authorize an affordable launch within the applicable ceiling.

```powershell
.\.venv\Scripts\python.exe -m model_collapse launch --config experiments/opt125m_wikitext2/continued_mixed_50_final5/full.yaml --generations 10
```

For the returned future full run ID:

```powershell
python -m model_collapse verify-detached RUN_ID
python -m model_collapse status RUN_ID
python -m model_collapse fetch RUN_ID --metadata-only
python -m model_collapse report RUN_ID
```

Use `resume RUN_ID` only for an unfinished, stopped run with compatible source/settings. Never resume an active/completed run. `cleanup RUN_ID` requires verified recovery and removes only the registered run's resources. The smoke resources are already removed; do not resume or relaunch them.

After a future launch confirms detached supervision and advancing progress, closing the laptop, browser or VS Code does not stop the remote experiment. The independent watchdog owns shutdown; private backups and the network volume retain results.
