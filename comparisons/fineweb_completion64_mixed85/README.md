# FineWeb 85% synthetic / 15% real

Ten fresh frozen chunks, 10,000 prompt/target pairs per cycle. Each cycle generates 8,500 64-token continuations using this condition's own preceding checkpoint, preserving 1,500 real targets. Nested SeedSequence selections, original row positions, continuation-only loss, one terminal epoch, continued AdamW/global cosine/RNG, and fixed evaluations match the completed conditions.

This condition has its own GPU, 60 GB working volume, supervisor, watchdog, allocation and checkpoint chain. Bulk artifacts use a shared private content-addressed Runpod S3 backup volume; small immutable indexes and records stay in pinned private Hugging Face commits. Historical backup behavior is unchanged.

Launch only through the scoped parallel launcher after this runtime/protocol passes GPU acceptance.

Hardware note: these three conditions use matching RTX PRO 4500 Blackwell GPUs after user-approved capacity fallback. The earlier 0/20/50/80/100% conditions used RTX PRO 4000 Blackwell. Scientific settings and frozen bank stay the same; cross-hardware numerical identity is not assumed.
