import copy
from pathlib import Path

import numpy as np
import pytest

from model_collapse.common import config
from model_collapse.completion_mixing import select_rows
from model_collapse.workflow import completion
from model_collapse.workflow.profiles import protocol_fingerprint, require_acceptance


CONFIG = Path("comparisons/fineweb_completion64_mixed80")


def test_single_condition_scope_forecast_and_acceptance_are_separate():
    full, smoke = (config(CONFIG / (kind + ".yaml")) for kind in ("full", "smoke"))
    completion.validate_config(full)
    completion.validate_config(smoke)
    assert protocol_fingerprint(full) == protocol_fingerprint(smoke)
    assert completion.allocation_id(full) == "fineweb_completion64_mixtures_mixed80"
    assert completion.benchmark_filename(full) == "benchmark_completion_mixed80.json"
    assert completion.acceptance_campaign(full) == "fineweb_completion64_mixtures_mixed80_acceptance"
    assert list(completion.child_roots(Path("run"), full)) == ["mixed80"]
    benchmark = {"full_primary_targets": {"test": 457664, "validation": 468608},
                 "full_secondary_targets": 1000, "training_target_tokens_per_second": 100,
                 "generation_new_tokens_per_second": 100, "primary_target_tokens_per_second": 100,
                 "secondary_target_tokens_per_second": 100}
    forecast = completion.forecast(benchmark, full, .57)
    assert forecast["unique_training_stages"] == forecast["epochs"] == forecast["generation_stages"] == 10
    assert forecast["generated_training_continuations"] == 80000
    assert forecast["generated_training_tokens"] == 5120000
    assert forecast["training_target_exposures"] == 6400000
    assert forecast["probe_sets"] == 11
    assert completion.required_backup_bytes(full) == 15300000000
    assert completion.required_backup_bytes(smoke) == 19890000000
    certificate = {"workflow": "completion_comparison", "source_code_sha256": "runtime",
                   "protocol_fingerprint": protocol_fingerprint(full),
                   "checks": {key: True for key in completion.acceptance_checks(full)}}
    require_acceptance(full, certificate, "runtime")
    assert "three_terminal_epochs" in certificate["checks"]
    assert "private_restore" in certificate["checks"]
    assert "six_terminal_epochs" not in certificate["checks"]
    wrong = copy.deepcopy(full)
    wrong["completion_arms"][0]["synthetic_fraction"] = .5
    with pytest.raises(RuntimeError, match="scientific protocol"):
        require_acceptance(wrong, certificate, "runtime")
    with pytest.raises(RuntimeError, match="source version"):
        require_acceptance(full, certificate, "different-runtime")


@pytest.mark.parametrize("cycle", range(1, 11))
def test_eighty_percent_selection_matches_prior_nested_selections(cycle):
    selection, real_count = select_rows(10000, .8, 42, cycle)
    assert real_count == 2000
    assert len(np.unique(selection["synthetic_indices"])) == 8000
    for fraction, count in ((.2, 2000), (.5, 5000)):
        historical, _ = select_rows(10000, fraction, 42, cycle)
        assert np.array_equal(selection["permutation"], historical["permutation"])
        assert np.array_equal(selection["synthetic_indices"][:count], historical["synthetic_indices"])
    small, small_real = select_rows(320, .8, 42, cycle)
    assert len(small["synthetic_indices"]) == 256 and small_real == 64
