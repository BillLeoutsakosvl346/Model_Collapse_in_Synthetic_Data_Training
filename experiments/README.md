# Scientific conditions

The [catalogue](catalog.yaml) maps named conditions to configurations, runs, acceptance and recovery evidence. Scientific names describe the protocol; Week 1/2/3 are presentation aliases.

| Definition | Initialization | Checkpoint selection | Synthetic fraction |
| --- | --- | --- | --- |
| [Pretrained reset](opt125m_wikitext2/pretrained_reset/README.md) | Original pretrained OPT each synthetic cycle | Best real validation; selected epoch 1 historically | 100% |
| [Continued from real S0](opt125m_wikitext2/continued_from_real_s0/README.md) | Previous selected weights | Best real validation; selected epoch 1 historically | 100% |
| [Reset to real S0](opt125m_wikitext2/reset_to_real_s0/README.md) | Fixed real-trained S0 after imported shared first cycle | Best real validation; selected epoch 1 historically | 100% |
| [Final-epoch-5 paired conditions](../comparisons/shared_start_final_epoch5/README.md) | Continued and fixed-S0 reset arms | Always epoch 5 | 100% |
| [Continued mixed 50/50](opt125m_wikitext2/continued_mixed_50_final5/README.md) | Previous epoch-5 weights | Always epoch 5 | 50%, fixed total training blocks |

Shared scientific definitions are in [methodology](../docs/methodology.md). Arm matching and exact shared artifacts belong in [comparisons](../comparisons/README.md). Measurements belong in dated [reports](../reports/README.md); launch authorization/current observations belong in [current state](../docs/current-state.md).

For new conditions, add a named definition and full/smoke configurations, register it in the catalogue, and record deviations explicitly. Reuse the shared engine. Configuration support alone is not GPU acceptance, spending permission or a completed experiment. Preserve [legacy configurations](../configs/README.md) for existing invocations.
