# Real-only pipeline validation — 2 November 2026

This is a small execution/recovery test, not a full-data scientific result. The [machine-readable acceptance receipt](../../docs/validation/2026-11-02_real-only-gpu-acceptance.json) records runtime, protocol, hashes and spend.

## Protocol and results

- Imported the exact historical real-trained S0. Trained continued weights for two cycles, five actual epochs each, retaining epoch 5.
- Each cycle used 256 real training blocks and zero synthetic training blocks, with 64 validation and 64 test blocks. Saved source selections/shuffles reconstructed exactly.
- Both checkpoint tensors changed; cycle 2 initialized from cycle 1's selected checkpoint. Reloaded terminal checkpoints passed evaluation.
- Two intentional worker exits recovered; partial generation shards and completed cycle-1 evidence were reused unchanged.
- Independent SSH sessions demonstrated detached progress. The separate budget watchdog stopped the actual Pod; provider shutdown was verified at 15:31:43 UTC.
- Private backups were fully restored on the GPU: 118 indexed files / 1,514,640,718 bytes. Restored test NLL/PPL exactly matched the recorded values, with zero direct shifted-NLL accounting error.
- Local recovery verified 93 metadata files / 11,158,271 bytes. Bulk weights/arrays remained cloud-side and were covered by the remote full-restore audit. Temporary Pod and disk were then deleted.

## Environment, cost and forecast

The test used RTX PRO 4500 Blackwell Server Edition in US-KS-2, the pinned official PyTorch image, BF16, microbatch 32 / effective batch 128 and five-beam probe decoding. It used a 30 GB persistent Pod disk at `/workspace`, with independent private backups. This differs from the historical RTX PRO 4000/network-volume environment and must remain explicit in later comparisons.

New control validation attempts conservatively cost USD 0.16588, including a failed host boot, capacity-only probe, preallocated-volume failures and the successful test. The successful test's whole-account debit was USD 0.08017; it includes concurrent historical storage and differs from conservative ledger estimates.

| Warm benchmark / full-control forecast | Value |
| --- | ---: |
| Training target tokens/second | 55,447.59 |
| Generated continuation tokens/second | 2,835.93 |
| New full-data training stages | 10 |
| New full synthetic training pools | 0 |
| New fixed 256-output probes | 10 |
| Estimated running time, including margins | 1.71 hours |
| Estimated compute and running storage | USD 1.25 |

Unused full synthetic pools are omitted at zero synthetic fraction; the real training selections and fixed measurement probes are preserved. Older retained storage and subsequent retention are separate charges. Full launch requires fresh capacity/budget admission; consult [current state](../../docs/current-state.md) before launching anything.
