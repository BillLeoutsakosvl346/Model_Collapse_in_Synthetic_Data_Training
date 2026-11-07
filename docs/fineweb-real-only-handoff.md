# FineWeb-Edu real-only pilot

## Completion verified on 6 November

**`20261106T151257Z-full-fresh-real` completed all ten cycles / ten epochs.** Runpod verifies automatic GPU shutdown at **15:46:14 UTC**, reason `complete`. Total provisioning-to-shutdown time was 33.26 minutes. The [completion receipt](validation/2026-11-06_fineweb-full-completion-status.json) verifies pinned saved metrics, stage seals, terminal-epoch selection, checkpoint/optimizer handoffs, all source-token quotas and frozen evaluation counts.

Validation PPL fell **27.96224 → 26.68859** and test PPL **28.34572 → 27.03484**, meeting the predeclared validation-stability criterion for this seed. Preserve stopped Pod `xf5ml3no72cg0z` and volume `mq5uhckkxr`. Bulk restoration/re-evaluation and authored analysis remain pending; this status check recovered small metadata. The launch details below are historical.

## Full pilot launched on 6 November

The user subsequently authorized the full pilot. At startup, **`20261106T151257Z-full-fresh-real` was running** on Pod `xf5ml3no72cg0z`, with volume `mq5uhckkxr` mounted at `/workspace`. Detached progress, the independent watchdog and a completed finite full-data epoch were verified. Training/backups/shutdown ran remotely. See the preserved [launch receipt](validation/2026-11-06_fineweb-full-launch.json).

Ada capacity disappeared before provisioning. The RTX PRO 4000 Blackwell capacity fallback passed [actual GPU acceptance](validation/2026-11-06_fineweb-blackwell-acceptance.json) before full launch, using the same runtime and scientific settings. The new GPU protocol is `ea54588894a9d0ccab721781c389848e7660aa57ac2595e5c005f95e9ee293f4`. The GPU costs USD 0.57/hour. The measured full-data forecast is **42.4 minutes / USD 0.407 running cost**, plus retained storage. Combined FineWeb validation spending is USD 0.23791 within USD 0.50; all temporary validation resources were removed. Earlier Ada figures below record the initial setup handoff.

```powershell
python -m model_collapse status 20261106T151257Z-full-fresh-real
```

## Readiness on 6 November

At the initial setup handoff, validation had passed and full launch was pending. See the preserved [validation report](../reports/20261106T143315Z-smoke-fresh-real/report.md) and [setup receipt](validation/2026-11-06_fineweb-setup.json).

- All **402 offline tests** passed. The actual GPU smoke completed three one-epoch cycles, each with 16,384 source-text tokens and fixed 64-document validation/test collections.
- Verified actual weight updates, terminal checkpoint reloads, continuous AdamW/scheduler state, exact data accounting and two injected worker interruptions. Completed cycle-1 artifacts remained unchanged.
- Restored and checksum-verified **92 private backup objects / 5.03 GB remotely**, then loaded the model and optimizer and reproduced test evaluation. Small records only were recovered to the laptop.
- Detached progress and independent budget-watchdog shutdown passed. Both temporary validation Pods and volumes were removed. Historical recovery resources were preserved.
- Fixed a local recovery-check serialization mismatch, added a regression and repeated GPU acceptance for the changed runtime. Both attempts reproduced identical data bytes and selected tensor hashes.

| Measurement or forecast | Recorded value |
| --- | ---: |
| Validation cost, both attempts, conservative ledger | USD 0.11383 |
| Observed combined account debit, including other storage/timing | USD 0.09545 |
| Full pilot, conservative time forecast | 50.4 minutes |
| Full pilot, compute and running storage forecast | USD 0.240 |
| Concurrent historical storage during that forecast | USD 0.024 |
| Seven days retaining the new 30 GB volume | USD 0.483 |
| Available cumulative allowance after the USD 2 reserve | USD 4.65 |

Technical acceptance and affordability passed at handoff. The forecast includes a 1.5× training/evaluation margin, setup and checkpoint/backup allowances. Matching hardware and funding are rechecked at launch. The accepted GPU is RTX 4000 Ada Generation, 20 GB, quoted at USD 0.28/hour, with official image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.

Accepted runtime is `13ddfea86bc9a87377bb08e81f1a43377c3e8767cf3965a3da34561dcf3cc0c7`, protocol `ee1cad91c589d649fd1edf4addf8ca8b0acc5757ea1f157341d49a12ed7a55ef`. The initial smoke validation/test PPL of **29.3578 / 32.6774** became **29.1006 / 32.4406** after its three small cycles. The full-data stability assessment remains pending.

## Protocol

Start from the pinned original pretrained OPT-125M. Evaluate it, then train ten consecutive one-epoch cycles. Each cycle receives 640,000 fresh source-text tokens from the pinned FineWeb-Edu `sample-10BT` collection. Continue the model, AdamW moments and global cosine schedule across cycles. Keep terminal weights after each epoch.

- Reserve 1,000 validation and 1,000 test documents before allocating training data. Evaluate these same collections initially and after every epoch.
- Keep accepted document identities and exact sliding 64-token passages disjoint across collections. Preserve original text, split into sequences up to 512 tokens and exclude padding from attention and loss.
- Meet each source-token quota exactly. Inserted BOS/EOS and padding have separate counts. The final training document can be trimmed, while its whole identity stays reserved.
- Use seed 42, BF16 autocast with FP32 parameters, effective batch 16 sequences, AdamW LR 5e-6 decaying globally to 1e-6.
- Save model, optimizer, scheduler, RNG, global step, data selections, hashes and measurements privately after every completed cycle.

The [condition definition](../experiments/opt125m_fineweb_edu/fresh_real_one_epoch/README.md) records exact revisions, sampling, sequence handling and limitations.

## Launch interface

The following command is the entrypoint used for the completed run recorded above. Use its existing run ID for inspection and recovery.

```powershell
$experiment = "experiments/opt125m_fineweb_edu/fresh_real_one_epoch"
python -m model_collapse launch --config "$experiment/full.yaml" --generations 10
```

The launcher rechecks hardware, private backups, existing resources, credit and the authorized budget. It returns a run ID after verifying the detached supervisor, independent watchdog and actual new training progress across an SSH disconnect. At that point the laptop can close. Training, backups and shutdown execute on Runpod.

```powershell
python -m model_collapse status RUN_ID
python -m model_collapse fetch RUN_ID --metadata-only
python -m model_collapse report RUN_ID
```

Metadata-only recovery retrieves small records and leaves model weights, optimizer tensors, token arrays and source documents in private cloud storage. Reports contain CSV/JSON and square figures for loss, perplexity, generalization gap, learning rate and training time.

```powershell
python -m model_collapse resume RUN_ID
python -m model_collapse fetch RUN_ID
python -m model_collapse cleanup RUN_ID
```

Resume applies to interrupted work. It verifies completed cycles, replays an unfinished cycle from its saved starting model/optimization state, and rejects completed runs or incompatible settings. Cleanup requires verified recovery. Full fetch downloads bulk artifacts to the laptop, so use it deliberately.

## Scientific assessment

The predeclared exploratory criterion is validation perplexity remaining at or below 105% of the untouched baseline at every cycle. The report retains the entire trajectory if this threshold is exceeded. The small GPU validation establishes execution/recovery behaviour. Scientific stability requires the later ten-cycle pilot.

The sample is generated by seeded shard order and a 10,000-document shuffle buffer. Its recorded coverage describes a shuffled stream. FineWeb-Edu is web text whose human authorship and overlap with OPT pretraining remain unaudited. Exact-passage checks cover 64 tokens and longer, with shorter or semantic overlap outside their scope. One seed gives descriptive results. The new data/context/optimizer protocol needs its own baseline before later synthetic-mixture comparisons.
