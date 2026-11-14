"""Matched 50/80 SGD suite routing, accounting and source interpretation."""
import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from model_collapse.workflow import sgd_profile as profile
from model_collapse.workflow import sgd_runtime
from model_collapse.workflow.completion import allocation_id, arm_config, acceptance_checks, benchmark_filename
from model_collapse.workflow.profiles import protocol_fingerprint, require_acceptance
from model_collapse.workflow.sgd_calibration import calibration_contract


def config(kind="smoke", historical=False):
    name = "smollm2_sgd_greedy_effects" if historical else "smollm2_sgd_greedy_mixtures"
    return yaml.safe_load(Path(f"comparisons/{name}/{kind}.yaml").read_text())


@pytest.mark.parametrize("kind", ["smoke", "full"])
def test_scientific_profile_and_independent_child_routing(kind):
    c = config(kind); profile.validate(c)
    assert allocation_id(c) == "smollm2_sgd_greedy_mixtures" + ("_setup" if kind == "smoke" else "")
    assert [s["id"] for s in profile.specs(c)] == ["mixed50", "mixed80"]
    for name, fraction in (("mixed50", .5), ("mixed80", .8)):
        child = arm_config(c, name); profile.validate(child)
        assert child["synthetic_fraction"] == fraction and child["data_kind"] == "mixed"
        assert child["learning_rate"] == child["min_learning_rate"] == .01
    assert profile.interruption_arm(c) == "mixed80"
    assert "sixteen_sgd_updates" in acceptance_checks(c)
    assert "twenty_four_sgd_updates" not in acceptance_checks(c)
    assert benchmark_filename(c) != benchmark_filename(config(historical=True))


def test_calibration_implementation_and_scientific_identity_reused():
    c, old = config(), config(historical=True)
    assert calibration_contract(c) == calibration_contract(old)
    assert c["calibration_reference"] == old["calibration_reference"]
    assert protocol_fingerprint(c) == protocol_fingerprint(config("full"))
    assert protocol_fingerprint(c) != protocol_fingerprint(old)
    certificate = {"workflow": "completion_comparison", "source_code_sha256": "x",
        "protocol_fingerprint": protocol_fingerprint(old), "checks": dict.fromkeys(profile.CHECKS, True)}
    with pytest.raises(RuntimeError, match="protocol"):
        require_acceptance(c, certificate, "x")


@pytest.mark.parametrize("fault", ["order", "fraction", "missing_arm", "calibration", "lr", "time", "cap"])
def test_unapproved_suite_or_settings_rejected(fault):
    c = config()
    if fault == "order": c["completion_arms"].reverse()
    if fault == "fraction": c["completion_arms"][0]["synthetic_fraction"] = .51
    if fault == "missing_arm": c["completion_arms"].pop()
    if fault == "calibration": c["calibration_required"] = True
    if fault == "lr": c["learning_rate"] = c["min_learning_rate"] = .001
    if fault == "time": c["validation_max_seconds"] = 1800
    if fault == "cap": c["max_session_usd"] = .4
    with pytest.raises((ValueError, RuntimeError)):
        profile.validate(c)


@pytest.mark.parametrize("cycle", [1, 2, 10])
@pytest.mark.parametrize("pairs", [400, 10000])
def test_nested_rows_and_target_scattering(cycle, pairs, tmp_path):
    from model_collapse.completion_mixing import select_rows, freeze_selection, freeze_targets
    from model_collapse.completion_data import assemble
    first, n_real = select_rows(pairs, .5, 42, cycle)
    second, real80 = select_rows(pairs, .8, 42, cycle)
    assert n_real == pairs // 2 and real80 == pairs // 5
    assert set(first["synthetic_indices"]) < set(second["synthetic_indices"])
    assert np.array_equal(first["synthetic_indices"], select_rows(pairs, .5, 42, cycle)[0]["synthetic_indices"])
    real = np.arange(pairs * 64, dtype=np.int32).reshape(pairs, 64)
    generated = np.full((pairs // 2, 64), 7, dtype=np.int32)
    first, description = freeze_selection(tmp_path, real + 13, real, .5, 42, cycle, "w" * 64)
    targets, _ = freeze_targets(tmp_path, real, generated, first, description, "g" * 64)
    real_indices = np.flatnonzero(first["source_labels"] == 0)
    assert np.array_equal(targets[first["synthetic_indices"]], generated)
    assert np.array_equal(targets[real_indices], real[real_indices])
    prompt = real + 13
    tokens, attention, scoring = assemble(prompt, targets, 0)
    assert np.array_equal(tokens[:, 1:65], prompt)
    assert np.all(attention == 1) and np.all(scoring[:, :65] == 0)
    assert np.all(scoring.sum(1) == 64)


def test_forecast_counts_both_mixtures_and_diagnostic_overhead():
    c = config("full")
    b = {"training_includes_update_observer": True, "step_seconds_by_arm": {"mixed50": 1., "mixed80": 2.},
         "generation_new_tokens_per_second": 2000., "primary_target_tokens_per_second": 50000.,
         "secondary_target_tokens_per_second": 50000., "full_primary_targets": {"validation": 476608, "test": 467392},
         "full_secondary_targets": 1000}
    result = sgd_runtime.forecast(b, c, .58)
    assert result["optimizer_updates"] == 1580
    assert result["unique_training_stages"] == result["epochs"] == 20
    assert result["generated_training_continuations"] == 130000
    assert result["training_target_exposures"] == 12800000 and result["probe_sets"] == 21


def test_loss_share_and_cancellation_keep_tail_denominators_distinct():
    from model_collapse.analysis.sgd_effects import source_summary
    row = {"arm": "mixed50", "cycle": 1, "real_pairs": 64, "synthetic_pairs": 64,
           "real_weighted_training_nll": 2., "synthetic_weighted_training_nll": 1.,
           "total_scored_targets": 8192, "real_gradient_norm": 3., "synthetic_gradient_norm": 4.,
           "total_gradient_norm": 4., "source_cosine": -.375,
           "real_conditional_delta_nll": -.1, "synthetic_conditional_delta_nll": .1,
           "mixed_delta_nll": 0., "interaction_nll": .01}
    tail = {**row, "real_pairs": 8, "synthetic_pairs": 8, "total_scored_targets": 1024,
            "real_weighted_training_nll": 4.}
    values = {r["population"]: r for r in source_summary([row, tail])}
    assert values["all_updates"]["real_share_of_mean_batch_loss"] == .75
    assert values["full_batches"]["real_share_of_mean_batch_loss"] == pytest.approx(2 / 3)
    assert values["tail_batches"]["real_share_of_mean_batch_loss"] == .8
    assert values["all_updates"]["real_share_of_corpus_loss"] == pytest.approx(20 / 29)
    assert values["all_updates"]["cancellation_vs_orthogonal_reference"] == pytest.approx(.2)


def test_historical_report_contract_ignores_location_but_keeps_receipt(tmp_path):
    from model_collapse.analysis.completion import comparison_contract
    from model_collapse.common import atomic_json
    c = config("full")
    baseline = {"weights_sha256": "w" * 64,
                "validation": {"scored_tokens": 476608}, "test": {"scored_tokens": 467392}}
    left, right = tmp_path / "left", tmp_path / "right"
    atomic_json(left / "config.json", c); atomic_json(left / "seed_42/gen_00/metrics.json", baseline)
    c["calibration_reference"]["snapshot"] = {"different": "verified recovery location"}
    atomic_json(right / "config.json", c); atomic_json(right / "seed_42/gen_00/metrics.json", baseline)
    assert comparison_contract(left) == comparison_contract(right)
    c["calibration_reference"]["receipt_sha256"] = "b" * 64
    atomic_json(right / "config.json", c)
    assert comparison_contract(left) != comparison_contract(right)
