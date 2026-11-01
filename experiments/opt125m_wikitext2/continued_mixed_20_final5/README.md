# Continued weights with 80% real / 20% synthetic targets

**Setup in progress; full training is unlaunched.** Start from the exact historical real-trained S0 and its pinned original synthetic D1 pool. Each new stage trains the previous selected model for five epochs on a new real/synthetic mixture and always retains epoch 5. AdamW and the cosine scheduler restart each cycle. No synthetic-trained S1 from another mixture condition is reused.

Each full mixture contains 30,357 real WikiText training blocks and 7,589 synthetic continuation blocks: 37,946 blocks / 2,428,544 tokens. The synthetic count is 20% rounded to the nearest block, half upwards; the real count fills the remainder. Sampling is without replacement within each source/cycle, using separate deterministic selection and shuffle streams. Mixtures remain fixed across their five epochs and interruptions. Real blocks may recur across cycles; they are resampled from the existing training split rather than newly collected text.

Ten cycles execute fifty epochs. D1 is imported; stages 1–9 generate full next-cycle pools and stage 10 generates only the fixed 256-output probe. The smoke profile uses 205 real + 51 synthetic blocks, 64 validation/test blocks and stages 1–2. This job creates the exact new S1/D2/M2 used by the matched 20/80 reset job.

Use the [campaign](../../../comparisons/mixed_fraction_final5/README.md) for launch and operations. Scientific definitions: [full.yaml](full.yaml), [smoke.yaml](smoke.yaml). [Shared methodology](../../../docs/methodology.md) and [metrics](../../../docs/metrics.md) retain the historical evaluation and reporting conventions.
