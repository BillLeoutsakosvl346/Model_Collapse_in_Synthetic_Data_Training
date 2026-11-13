# SmolLM2 gradient diagnostic

This diagnostic reuses the completed SmolLM2 99% and 100% synthetic conditions. It restores M0, M4 and M9 and their exact subsequent cycle-1/5/10 targets. It does not execute another recursive-training experiment.

## Measurements

- Mean continuation-token gradients over complete saved source collections; FP32, dropout disabled, microbatch eight.
- Actual 99% mixture decomposition: 100 real rows and 9,900 synthetic rows. Same-prompt original real targets provide a separate control.
- Gradient norms, weighted norms and alignment, globally and per parameter tensor.
- One isolated aggregate-gradient AdamW step using original optimizer history, at both the historical learning rate and a common 3e-6 rate.
- Unchanged and explicit-zero-gradient optimizer-history controls. These distinguish data effects from AdamW momentum and weight decay.
- Fixed 1,024-pair document-stratified validation probe, paired document-bootstrap intervals, and full-validation confirmation of historical-rate mixed/history effects.

These steps deliberately use complete-source mean gradients. They are not replays of the historical 128-example minibatch updates. Probe intervals do not measure training-seed uncertainty. Gradient strength alone does not establish beneficial influence or explain long-term stability.

## Commands

```powershell
python -m model_collapse diagnose-gradients --config comparisons/smollm2_gradient_diagnostic/diagnostic.yaml --preflight-only
python -m model_collapse diagnose-gradients --config comparisons/smollm2_gradient_diagnostic/diagnostic.yaml
python -m model_collapse status DIAGNOSTIC_ID
python -m model_collapse fetch DIAGNOSTIC_ID
python -m model_collapse report DIAGNOSTIC_ID
python -m model_collapse validate DIAGNOSTIC_ID
python -m model_collapse cleanup DIAGNOSTIC_ID
```

The preflight checks private source access, source object sizes, configured GPU stock, budget and a small backup round trip; it creates no compute/storage resources. The actual execution uses one new temporary 30 GB Standard network volume and the original PyTorch image. Original PRO 4000/4500 stock was initially unavailable; two 5090 placement requests were refused and their temporary volumes removed. Original RTX PRO 4000 stock then returned, and all actual diagnostic cases use that matching hardware. The watchdog deadline includes bootstrap and restoration. Full fetching is small because historical bulk sources are excluded from diagnostic backups and remain in their original storage.

## Limits and recovery

The original USD 2/0.15 plan was amended under the user's subsequent execution authorization: the bounded task cap is USD 3, with at most USD 0.35 for miniature 16-training/16-validation-pair GPU acceptance and at most 60 minutes combined paid compute. The cumulative USD 110 ceiling and existing USD 2 reserve remain binding; the historical setup ceiling is USD 5.40. No credits or payment settings change.

Pinned source indexes, source-recovery records, tensor/optimizer checks, raw measurement arrays, analysis input/output hashes and figures are saved separately from historical evidence. Delete only diagnostic resources after full evidence recovery is verified. Preserve every historical volume.

## Supplementary statistics

The first deployed pass omitted synthetic per-parameter norms while retaining all global measurements and real per-parameter norms. A separately accepted `phase: layer_norms` supplement recovers that breakdown without changing primary records. It also measures exploratory held-out-gradient alignment and reproduces isolated synthetic/mixed AdamW steps to compare their first-order and observed loss effects. Held-out gradients never supply an update. Source checkpoint files remain read-only; temporary shadow weights are restored exactly.

The user's subsequent broad execution instruction authorizes at most **90 minutes combined paid compute** for the primary and supplementary jobs, sequentially on one GPU, within the same USD 3/0.35 task/validation caps. The original primary watchdog remains limited to 60 minutes. The supplement is admitted only after verified primary completion/shutdown and uses its immutable pinned measurement snapshot.
