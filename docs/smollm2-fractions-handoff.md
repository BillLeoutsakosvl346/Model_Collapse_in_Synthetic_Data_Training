# SmolLM2 fraction sweep: execution handoff

**11 November 2026: All seven full conditions completed and were reported; compute is stopped.** The [acceptance/startup receipt](validation/2026-11-11_smollm2-acceptance-and-launch.json) records actual coverage, resource ownership and costs. The [earlier capacity-blocked receipt](validation/2026-11-11_smollm2-setup.json) remains historical evidence.

## Experiment

Full comparison **`20261111T210854Z-full-completion64-mixtures`** runs **0/50/80/95/99/99.9/100% synthetic** sequentially on **one RTX PRO 4000 Blackwell at USD 0.57/hour**, EU-RO-1. Each condition independently starts from pretrained SmolLM2-135M and carries its own weights, AdamW state, global cosine scheduler and training RNG through ten fresh chunks. Synthetic targets come from that condition's preceding checkpoint; only selected target rows are replaced. Historical OPT data/results remain unchanged.

One terminal epoch per chunk; BOS + 64 prompt + 64 target tokens, continuation-only loss. BF16 autocast with FP32 parameters, microbatch 32/effective batch 128, seed 42, five deterministic beams, exactly 64 new tokens. The historical global cosine LR remains **5e-6 to 1e-6**. Each condition has 790 optimizer updates and 6.4 million scored training targets. The complete sweep has 70 training stages and 523,900 generated training continuations.

## Checks and recovery scope

- **104 focused offline tests passed in 17.58 seconds**. Supplementary runs overlap; do not add their counts. The complete historical suite was not run. Independent review checked model compatibility, masks, mixtures, continued optimization and cloud safeguards; material findings were resolved.
- Actual mixed50 GPU acceptance passed **all ten checks** on the selected RTX PRO 4000 Blackwell, and separately on A40. Each test trained two fresh 128-pair chunks, one update per stage; 32 held-out/probe pairs. Other fractions were checked offline, not trained in GPU acceptance.
- Both GPU tests verified changed weights, stage-2 generator/initializer/optimizer parent, exact target scattering, masks, terminal reload and direct conditional NLL.
- Each accepted GPU cold-restored one complete terminal stage: **26 files / 1.615 GB**, including optimizer state, and reproduced held-out loss exactly. Parent/child metadata was checksum-verified locally. This does not establish every-checkpoint restoration.
- Detached progress and ordinary independent watchdog/provider shutdown were verified. A [time-only recovery extension](validation/2026-11-11_smollm2-recovery-only-extension.json) changed the combined limit from 900 to 1,200 seconds; recovery performed no extra updates. Actual combined compute: **1,134.46 seconds**, conservative compute estimate **USD 0.2114**, distinct from account debits. The USD 0.50 cap stayed unchanged.
- Accepted/deployed runtime: `44bae015987c72fb96d9750c59e2065c0fcaca5e3f2121379e2a6354df593a1f`. Exact bundles and host transport/retry adapters are retained privately. Recovery-only host controls are documented separately from the unchanged scientific runtime.

## Frozen data and storage

Model revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. The native bank preserves the pinned FineWeb source, held-out assignments and historical training-document allocations. Complete token reconstruction, document/passage exclusions and stratified-probe checks passed. Historical OPT bank bytes/builders remain unchanged.

| Collection | Pairs | Scored continuation tokens |
| --- | ---: | ---: |
| Each of ten training chunks | 10,000 | 640,000 |
| Fixed validation | 7,447 | 476,608 |
| Fixed test | 7,303 | 467,392 |

Local bank: `data/smollm2_completion64_v2/`, approximately 159 MB; readable passages are in `examples.json`. The pinned private bank identity is embedded in both configurations and the receipt.

Keep these full-run resources:

- Pod **`oan7wv9pkt61fi`**; working volume **`cw2u22hhef`**, 200 GB.
- Private backup **`tsp5kiuw8v`**, 200 GB, with pinned Hub indexes referencing checksummed bulk objects.
- Native bank **`1b2vuplxer`**, 10 GB.

All validation compute is stopped. Preserve validation workspaces `wnz3vyv6zu` and `f7x2kv68h4` until remaining parent evidence is fully recovered. Historical resources were not altered. Metadata-only fetching avoids bulk checkpoint downloads to the laptop.

## Runtime and spending

Measured warmed-up rates: **15,860 training targets/s**, **862 generated tokens/s**, **43,010 primary evaluation targets/s**. Generation dominates this workload.

Padded admission forecast: **22.2 hours / USD 13.17 compute and running work storage**. Other retained storage adds a conservative USD 3.59 during that period; seven days of the new 400 GB adds approximately USD 6.44. Thus the forecast is about **USD 23.20 including those retention assumptions**, not an upfront charge. Actual compute may be lower because admission includes a 50% throughput margin and per-stage backup overhead. USD 10-15 is a cost target, not a guaranteed all-inclusive bill.

The user manually added USD 30 and authorized proceeding. Limits: **USD 110 cumulative / USD 30 task / USD 2 reserve / USD 5.05 historical setup / USD 0.50 new validation**. No agent credit purchase or payment change. Shutdown is independent of SSH/laptop/agent; retained storage continues billing afterward.

## Operations

Do not resume or duplicate this completed run. See the [completion report](../reports/20261111T210854Z-full-completion64-mixtures/report.md) and [delivery receipt](validation/2026-11-12_smollm2-completion-analysis-and-doc.json). The earlier forecast and startup checks below remain historical.

```powershell
python -m model_collapse status 20261111T210854Z-full-completion64-mixtures
python -m model_collapse fetch 20261111T210854Z-full-completion64-mixtures --metadata-only
python -m model_collapse report 20261111T210854Z-full-completion64-mixtures
```

Resume only unfinished stopped work. Cleanup requires verified recovery. Within-model comparisons share tokenizer/evaluation; cross-model plots must label tokenizer/pretraining differences. SmolLM2 pretraining includes FineWeb-Edu, so held-out separation does not prove pretraining novelty. One seed and decreasing LR remain limitations.
