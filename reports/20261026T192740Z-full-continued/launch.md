# Continued-weight full experiment: launch record

**Running at handoff on 26 October 2026. Results are pending.**

The user explicitly authorized the full ten-cycle continued-training run after smoke validation. Experiment **`20261026T192740Z-full-continued`** launched at **19:27 UTC/London** on Runpod Pod `itwdnizndn5hws`, with 30 GB Standard network volume `z6ziujxxus` in EU-RO-1. The cheapest available compatible GPU at launch was an RTX PRO 4000 Blackwell (24 GB) at **$0.57/hour**. The earlier $0.24/hour GPU was not available when selected.

The detached supervisor, worker and independent budget watchdog were verified after the launcher disconnected SSH. CUDA/BF16 checks and actual warmed training/five-beam generation succeeded. A later observation confirmed live processes, fresh heartbeats and continuing generation progress: **1,280 of 37,946 continuations** saved at that observation. The first atomic shard passed its SHA-256 check and has shape **256 × 64**, with the expected saved real-G0 model tensor hash. All three real split arrays match the previous full experiment byte-for-byte. See [launch evidence](launch.json).

The configured chain starts from the pinned real-data-trained G0, generates its own synthetic dataset and continues training its weights on those data, repeating through S10. Selected weights carry forward; AdamW/scheduler restart per cycle. Each cycle uses five epochs, the fixed real validation checkpoint rule and the unchanged real test split. This is full data, not the smoke subset. No model-collapse result is claimed at launch.

The fresh benchmark measured **34,258.09 training target tokens/second** and **1,304.68 generated new tokens/second**. The conservative forecast is **10.11 hours / $5.82** for ten cycles, including the 1.5x compute margin and setup/checkpoint/upload allowance. The persistent budget ledger conservatively charged approximately $3.19 of earlier task costs at launch, with $2 reserved; the independent new-session allowance is $19.8133 within the **$25 cumulative task ceiling**. Actual final cost and runtime are not yet known. No credits were purchased and automatic payments remain disabled.

The laptop, browser, editor and Codex session may close. The remote supervisor runs the chain, saves every selected checkpoint and dataset on the network volume, backs up completed stages to private Hugging Face repositories, creates metrics/plots, and requests automatic GPU shutdown on completion, failure or the budget cutoff. Retained network storage continues billing after GPU shutdown until verified retrieval/cleanup.

```powershell
python -m model_collapse status 20261026T192740Z-full-continued
python -m model_collapse fetch 20261026T192740Z-full-continued --metadata-only
python -m model_collapse report 20261026T192740Z-full-continued
```

The metadata-only fetch retrieves small reports without multi-GB weights/token arrays. Complete artifact retrieval uses `fetch` without that flag. Resume only if the existing run has actually stopped and needs recovery, using its original code/config and retained volume. Do not launch a duplicate. Clean up only after verified recovery.

Private model backup: `Bill11235813/iib-model-collapse-week1-models`; private data/report backup: `Bill11235813/iib-model-collapse-week1-data`, both under this experiment ID. The recorded Pod/volume IDs and original source version are in the evidence file. Update this record with completion only after checking real artifacts and stopped compute.
