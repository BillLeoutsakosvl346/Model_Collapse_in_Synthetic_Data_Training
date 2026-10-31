# Shared start, final epoch 5

**Full pair completed on 31 October at 06:05 UTC/London:** `20261030T180755Z-full-paired-final5`. Both arms reached cycle 10; all 19 unique synthetic stages executed and retained epoch 5 (95 epochs). Runtime was 11.97 hours. Runpod confirms automatic GPU shutdown at 06:06:47, reason `complete`. See the [completion receipt](../../docs/validation/2026-10-31_final-epoch5-full-completion.json), [launch receipt](../../docs/validation/2026-10-30_final-epoch5-full-launch.json), [acceptance report](../../reports/20261030T172428Z-smoke-paired-final5/report.md) and [handoff](../../docs/final-epoch5-handoff.md). Do not launch or resume completed compute.

This is a new matched comparison. Historical Week 2 trained five epochs but selected epoch 1. This condition **retains epoch 5**, even if real validation loss worsens. It does not retrain or change the existing real-trained S0.

## Week 3 findings and reproduction

The [Week 3 report](../../reports/20261030T180755Z-full-paired-final5/report.md) contains three figure families, full endpoint statistics, saved text and limitations. Final real-test PPL is **107.37 continued / 44.50 reset**; reset has lower loss on 4,475/4,494 fixed test blocks. The continued probe contains 38.62% newline tokens and ten entirely newline outputs, versus reset's 2.80% and zero. Newline exclusion reverses pooled entropy/repetition rankings; these post-hoc diagnostics use unequal retained samples and do not replace the raw metrics or establish output quality.

The [evidence audit](../../reports/20261030T180755Z-full-paired-final5/analysis/audit/audit.json) verifies 259 selectively recovered files / 13.09 MB, including twenty unique probes. All 19 new stages executed and retained epoch 5, although every stage's best real validation loss was epoch 1. Historical/new cycle-1 loss trajectories agree exactly. No bulk weights/full datasets were downloaded or re-evaluated for this analysis.

From the published records, reproduce without cloud access or paid compute:

```powershell
python comparisons/shared_start_final_epoch5/analyse_evidence.py --out .state/week3-audit-review
python comparisons/shared_start_final_epoch5/analyse.py --out .state/week3-analysis-review
```

Choose new empty output directories; published analysis is protected from overwrite. `--recover` on the audit script optionally reads pinned private backups using saved run provenance; it creates no compute. [Recovery scope](../../reports/20261030T180755Z-full-paired-final5/records/recovery_manifest.json) and [analysis input/output hashes](../../reports/20261030T180755Z-full-paired-final5/analysis/results/analysis_manifest.json) are explicit. The authored report can be rebuilt with `python scripts/analysis/build_week3_report.py`.

The exact historical continued-run D1 is imported from immutable private backups. S0 trains on D1 once to create the new shared S1, which generates one new D2. Continued weights then train S1 on D2; reset trains fixed S0 on that exact D2. Each arm subsequently uses its own previous output dataset. AdamW and the cosine scheduler restart each cycle.

The full pair completes continued cycles 1–10 and reset cycles 2–10: 19 unique training stages / 95 executed epochs. Continued S10 inherits 50 synthetic epochs; reset A10 inherits its own five synthetic epochs from S0. Data lineage is recursive in both arms. Existing fixed real splits, 64→64 generation, five deterministic beams, token counts, seed, batches and metrics remain unchanged.

## Commands

Setup validation only:

```powershell
python -m model_collapse smoke --config comparisons/shared_start_final_epoch5/smoke.yaml
python -m model_collapse status PAIR_ID
python -m model_collapse verify-detached PAIR_ID
python -m model_collapse fetch PAIR_ID --metadata-only
python -m model_collapse validate PAIR_ID
python -m model_collapse report PAIR_ID
python -m model_collapse cleanup PAIR_ID
```

The separately authorised full launch used this command; it is now complete:

```powershell
python -m model_collapse launch --config comparisons/shared_start_final_epoch5/full.yaml --generations 10
```

One detached remote coordinator runs continued, seals its completed private backup, then runs reset on the same GPU. The pair owns one Pod, volume, ledger interval and watchdog deadline. `status`, `resume`, `fetch`, `report` and `cleanup` operate on the **pair ID**, not child IDs. Closing the laptop after verified launch does not control remote execution.

Full admission requires acceptance for the exact runtime and scientific profile, matching RTX PRO 4000 Blackwell/software, and a conservative affordable estimate for both arms. Setup does not authorise a full launch. The cumulative $25 task cap, $2 reserve, existing $3 setup allowance and $0.75 combined acceptance limit remain enforced.

Smoke uses 256 training / 64 validation / 64 test blocks and five real epochs for continued stages 1/2 and reset stages 2/3. It validates infrastructure and lineage; it is not evidence of collapse. A single full pair is descriptive, not an independent replication study.
