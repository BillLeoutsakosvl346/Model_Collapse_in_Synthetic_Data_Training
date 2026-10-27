# Shared S0/S1: reset versus continue

**Question:** after one identical synthetic-training cycle, does returning to a fixed real-data-trained checkpoint behave differently from repeatedly updating the preceding selected checkpoint?

**State:** the continued arm is complete; the reset full run `20261027T211852Z-full-reset-s0` is running following separate user authorization. Existing Week 1 and Week 2 evidence remains unchanged. [Machine-readable definition](comparison.yaml) · [Launch record](../../docs/validation/2026-10-27_reset-full-launch.json).

The reset setup passed [current-source GPU acceptance](../../reports/20261027T203952Z-smoke-reset-s0/report.md). This confirms the branch implementation and recovery, not full-data behaviour. [Handoff](../../docs/implementation-handoff.md).

## Common prefix and branch point

Use the completed continued run `20261026T192740Z-full-continued`:

```text
S0 evaluated -> generate D1 -> train S0 on D1 -> S1 evaluated -> generate D2
                                                             |
                              +------------------------------+----------------------+
                              |                                                     |
Reset arm:             train S0 on D2 -> A2                    Continue arm: train S1 on D2 -> B2
                       A2 generates DA3                                      B2 generates DB3
                       train S0 on DA3 -> A3                                 train B2 on DB3 -> B3
                       ... through A10                                       ... through B10
```

Both plotted arms reference the identical saved S0 and S1 evaluation records. Their shared first jump is approximately 33.89 to 48.38 test perplexity. At cycle 2, the synthetic data are still identical; only training initialization differs. From cycle 3 the feedback loop also makes the datasets branch-specific.

| Shared artifact | Content/tensor SHA-256 |
| --- | --- |
| S0 weights | `dd12f1e10e953a352299108fc013fa8c5b6d2afcd99addfc41e8df5ddc586b10` |
| S1 weights | `5cbccf4740806deeb6d972426309982ae9713651fa1b41602d81cb0ff421f91f` |
| D2 continuation tokens | `4ed3a1b44b031cad1e2a6ba72cb368c0e2efb7a848237e262ce744a3de255e42` |

These are content identities, not serialized-file checksums. Import resolves pinned model/dataset Hub commits and verifies index hashes and every required file checksum separately. The existing private backup locations and stage prefixes are preserved.

## Controlled factors

Hold fixed model/tokenizer and data revisions, packing, prompt order, continuation-only targets, equal token counts, frozen validation/test arrays, five-beam decoding, repetition penalty 1.0, five epochs, fresh AdamW and scheduler, effective batch size 128, seed 42, precision and validation selection. Match the continued arm's RTX PRO 4000 Blackwell and recorded software.

Use the same shared training/generation engine for both arms. Optimizer state is not carried in either arm; the deliberate policy difference is which model weights initialise training. The new reset arm performs nine new stages, cycles 2-10; imported stages retain their original provenance, timing and cost rather than being counted as newly executed work.

## Interpretation and outputs

Compare corpus validation/test NLL/PPL, passage distributions, fixed-probe diversity/repetition, duplicates, empirical token entropy, frozen-S0 scores and timing. Preserve metric definitions and the one-cycle offset between a model's evaluation and its generated next training dataset.

New comparison analyses belong under `analyses/<analysis_id>/`, with input hashes, analysis revision/environment and output hashes. The shared points must be referenced, not fabricated or averaged as independent samples. Original reports and CSV schemas are protected.

This is still one seeded pair with a historical continued arm. Matching hardware/software reduces confounding but does not create independent replicates, guarantee bitwise deterministic recomputation or establish a general causal law. If the intended environment cannot be matched, do not label the comparison fully matched; leave the launch pending or plan a separately authorised paired rerun.

The old pretrained-reset arm remains scientifically useful, but is not this reset arm: it starts from original pretrained OPT rather than real-trained S0. Its first-cycle difference should not be presented as the branch point in this comparison.
