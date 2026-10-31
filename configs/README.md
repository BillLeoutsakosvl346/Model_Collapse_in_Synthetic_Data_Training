# Legacy configuration paths

These files remain for existing CLI defaults, scripts and tests. Their historical scientific meaning is unchanged. Prefer named configurations under [experiments](../experiments/README.md) for new explicit invocations.

| Compatibility path | Named condition/profile |
| --- | --- |
| `full.yaml` | [Pretrained reset full](../experiments/opt125m_wikitext2/pretrained_reset/full.yaml) |
| `smoke.yaml` | [Pretrained reset smoke](../experiments/opt125m_wikitext2/pretrained_reset/smoke.yaml) |
| `full-continued.yaml` | [Continued-from-real-S0 full](../experiments/opt125m_wikitext2/continued_from_real_s0/full.yaml) |
| `smoke-continued.yaml` | [Continued-from-real-S0 smoke](../experiments/opt125m_wikitext2/continued_from_real_s0/smoke.yaml) |

Named copies add catalogue metadata; their scientific settings must continue to match. `--mode fresh_base` means original pretrained reset, not fixed real-trained S0. `--mode continued` selects the historical continued condition, not the final-epoch-5 or mixed profiles. `--mode` and `--config` are mutually exclusive. Do not delete these paths or silently retarget defaults during housekeeping.
