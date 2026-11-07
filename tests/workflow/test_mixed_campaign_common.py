"""Mixed reset configuration keeps the explicit initializer and source contract."""
import copy

import pytest

from model_collapse.common import PROJECT, config, validate_config


@pytest.mark.parametrize("fraction", [.5, .2])
def test_mixed_reset_requires_the_same_pinned_initializer_and_branch(fraction):
    c = config(PROJECT / "experiments/opt125m_wikitext2/reset_to_real_s0/full.yaml")
    c["synthetic_fraction"] = fraction
    validate_config(c)
    for key in ("start_checkpoint", "branch_source"):
        changed = copy.deepcopy(c)
        changed.pop(key)
        with pytest.raises(ValueError):
            validate_config(changed)


def test_mixing_does_not_redefine_pretrained_reset():
    c = config(PROJECT / "configs/full.yaml")
    c["synthetic_fraction"] = .2
    with pytest.raises(ValueError, match="continued or reset_s0"):
        validate_config(c)
