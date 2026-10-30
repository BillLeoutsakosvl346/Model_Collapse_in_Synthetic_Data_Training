# Shared start, final epoch 5

This is a new matched comparison. Historical Week 2 trained five epochs but selected epoch 1. This condition **retains epoch 5**, even if real validation loss worsens. It does not retrain or change the existing real-trained S0.

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

The separately authorised future full launch is:

```powershell
python -m model_collapse launch --config comparisons/shared_start_final_epoch5/full.yaml --generations 10
```

One detached remote coordinator runs continued, seals its completed private backup, then runs reset on the same GPU. The pair owns one Pod, volume, ledger interval and watchdog deadline. `status`, `resume`, `fetch`, `report` and `cleanup` operate on the **pair ID**, not child IDs. Closing the laptop after verified launch does not control remote execution.

Full admission requires acceptance for the exact runtime and scientific profile, matching RTX PRO 4000 Blackwell/software, and a conservative affordable estimate for both arms. Setup does not authorise a full launch. The cumulative $25 task cap, $2 reserve, existing $3 setup allowance and $0.75 combined acceptance limit remain enforced.

Smoke uses 256 training / 64 validation / 64 test blocks and five real epochs for continued stages 1/2 and reset stages 2/3. It validates infrastructure and lineage; it is not evidence of collapse. A single full pair is descriptive, not an independent replication study.
