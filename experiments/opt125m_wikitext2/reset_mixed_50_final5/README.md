# Fixed-S0 reset with 50% real / 50% synthetic targets

**Setup in progress; full training is unlaunched.** This condition extends the completed [50/50 continued run](../continued_mixed_50_final5/README.md) using its exact S0, final-epoch-5 S1 and D2. Its first new training stage is cycle 2; stages 2–10 initialize from the same real-trained S0, with fresh AdamW and scheduler state. Every stage executes five epochs and retains epoch 5.

Each full mixture has 18,973 real WikiText training blocks and 18,973 synthetic continuation blocks: 37,946 blocks / 2,428,544 tokens. Cycle 2 must reproduce the source run's exact M2 token array, source selections, labels and shuffle. Later synthetic pools come from this reset arm's preceding selected model. The fixed real split, prompts, evaluation arrays, decoding and metric definitions remain unchanged.

The smoke profile uses 256 training / 64 validation / 64 test blocks and two new stages, 2–3. Its 128/128 mixture uses a reduced source population and must not be described as the full historical M2. Acceptance separately reconstructs and hashes the full historical M2 on the GPU; imported S0/S1 measurements retain their original full-split scope.

Use the [campaign](../../../comparisons/mixed_fraction_final5/README.md) for the tested launch and operations interface. The scientific definitions are [full.yaml](full.yaml) and [smoke.yaml](smoke.yaml). Full training remains a separate launch decision after acceptance and affordability checks. See [shared methodology](../../../docs/methodology.md) and [metrics](../../../docs/metrics.md).
