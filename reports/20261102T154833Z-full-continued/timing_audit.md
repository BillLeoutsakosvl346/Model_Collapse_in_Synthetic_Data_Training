# Real-only execution and timing audit

The completed control executed the intended ten continued-training cycles and all fifty epochs. The 48.54-minute provider interval is consistent with the saved histories. This audit found no shortened training, smoke limits, skipped cycles or reused synthetic-trained checkpoints. It does not replace a later scientific analysis or independent model re-evaluation.

## Checks

- Recovered and checksum-verified 229 metadata files / 51,706,836 bytes from pinned private backup commits. Separately verified the 9,714,304-byte real training array; no checkpoint weights were downloaded.
- All 50 unique `epoch_complete` events match the saved histories. Every epoch's measured throughput × training time recovers exactly 37,946 processed blocks, including the final partial batch. The full configuration has no training, validation or test subset limits.
- Reconstructed all ten mixtures from the frozen real array and saved indices/permutations. All reconstructed content hashes and exact NumPy-file hashes match the training records, mixture manifests and immutable backup index. Each contains the full real pool once, with zero synthetic targets.
- All ten initializers match S0 or the preceding cycle's selected checkpoint. Every stage changes weights and records a reloaded terminal checkpoint equal to its selected epoch-5 tensor hash. All ten new checkpoints are distinct.
- All validation/test evaluations use the frozen full arrays: 3,922 / 4,494 blocks, scoring 247,086 / 283,122 tokens. Corpus NLL/perplexity arithmetic and all saved per-test-block NLL arrays agree.
- The exact archived runtime matches its manifest and recorded identity. Its training engine visits all effective batches, runs backward/AdamW updates, and retains epoch 5. The available structured logs contain no training-reuse, interruption, failure or budget-pause events.
- Provider lifecycle logs show the container starting at 15:48:39 and stopping at 16:37:07 UTC. The completion manifest is timestamped 16:36:32; the independent watchdog records `complete`, and provider shutdown is verified at 16:37:06.

All **161 consistency checks passed**. See [machine-readable audit](analysis/timing_audit.json) and [per-cycle timings](analysis/timing_by_cycle.csv).

## Actual work and timing

Each epoch processed 2,428,544 input tokens / 2,390,598 next-token targets. Fifty epochs therefore scored approximately **119.53 million training targets**. The batch loops imply **297 optimizer updates per epoch / 14,850 overall**; these step counts are derived from the audited code and full block counts, rather than a separate per-step log.

| Measured component | Minutes |
| --- | ---: |
| Fifty training loops | 37.47 |
| Validation and checkpoint work inside training stages | 2.99 |
| Ten new fixed measurement probes | 0.95 |
| Remaining setup, evaluation, backup, reporting and shutdown | 7.13 |
| **Provisioning to verified shutdown** | **48.54** |

Each training epoch took 44.50–45.43 seconds. OPT-125M and 64-token contexts are small workloads for this GPU. During supervised training, all next-token positions are evaluated together in a forward/backward pass. Synthetic generation instead produces successive tokens autoregressively and uses five beams; generating whole datasets can dominate these experiments.

## Why the original forecast was eight hours

The original forecast was reproduced exactly from the preserved RTX PRO 4000 benchmark. It retained the mixed-data workflow's generation of **nine full synthetic pools plus the final probe: 21,873,280 new tokens**, although this control selects none of those synthetic targets.

The accepted real-only path generates only **ten fixed 256-output probes: 163,840 new tokens**. The original D1 is imported unchanged and contributes no newly generated tokens or training targets. This removes unused generation while preserving training data, epoch count and measurement probes.

| Forecast reconstruction | Hours |
| --- | ---: |
| Original benchmark, including unused full synthetic pools | 8.51 |
| Same old benchmark, with measurement probes only | 2.22 |
| Actual RTX PRO 4500 benchmark, probes only | 1.71 |
| **Measured elapsed run** | **0.81** |

The 1.71-hour forecast further included a 1.5× compute margin, five minutes of setup, and three minutes per checkpoint stage (33 minutes). Actual I/O and overhead were much lower. These conservative allowances explain the remaining difference; training was not reduced. The RTX PRO 4500 is also faster in the recorded benchmark than the original RTX PRO 4000 and remains an explicit hardware difference in scientific comparisons.

## Coverage limits

The private `pipeline.log` redirects worker stdout on the stopped Pod and is excluded from the backup allowlist. This audit inspected the available provider lifecycle logs, structured events, all epoch histories, lineage, stage manifests and pinned artifact hashes without restarting compute. Checkpoint bytes were not downloaded or independently re-evaluated here; preserve the stopped Pod/disk until adequate bulk recovery is established. The generic provider log says “network volume” while the actual resource mount is a persistent Pod disk; the saved ownership/mount records determine its lifecycle.

Earlier forecasts and launch receipts remain unchanged as historical evidence. No GPU was restarted, provisioned or deleted for this audit.
