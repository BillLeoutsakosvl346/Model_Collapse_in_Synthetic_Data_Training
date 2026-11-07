"""Sparse real targets must remain represented in the 99.9% acceptance scope."""
import argparse
import copy

import numpy as np
import pytest

from model_collapse.common import config, read_json
from model_collapse.completion_mixing import freeze_selection, freeze_targets, select_rows
from model_collapse.workflow.completion import smoke_pair_count, validate_config
from model_collapse.workflow.profiles import protocol_fingerprint, require_acceptance


def settings(kind):
    return config(f"comparisons/fineweb_completion64_mixed99p9/{kind}.yaml")


def test_sparse_profiles_match_historical_science_and_bind_acceptance():
    full, smoke = settings("full"), settings("smoke")
    previous = config("comparisons/fineweb_completion64_mixed99/full.yaml")
    assert full["completion_arms"] == [{"id": "mixed99p9", "synthetic_fraction": .999}]
    assert smoke_pair_count(smoke) == smoke["train_pairs_per_cycle"] == 1000
    assert protocol_fingerprint(full) == protocol_fingerprint(smoke)
    excluded = {"experiment_id", "completion_arms", "smoke_train_pairs", "backup_store"}
    assert {k:v for k,v in full.items() if k not in excluded} == {
        k:v for k,v in previous.items() if k not in excluded}
    assert "parallel_campaign" not in full and "parallel_campaign" not in smoke
    certificate = {"source_code_sha256": "runtime", "protocol_fingerprint": protocol_fingerprint(previous)}
    with pytest.raises(RuntimeError, match="scientific protocol"):
        require_acceptance(full, certificate, "runtime")
    changed = copy.deepcopy(full)
    changed["smoke_train_pairs"] = 2000
    assert protocol_fingerprint(changed) != protocol_fingerprint(full)


@pytest.mark.parametrize("cycle", range(1, 11))
def test_full_and_smoke_counts_and_nested_selections(cycle):
    previous = None
    for fraction, expected in ((.95, 9500), (.99, 9900), (.999, 9990)):
        selection, real = select_rows(10000, fraction, 42, cycle)
        assert len(selection["synthetic_indices"]) == expected and real == 10000 - expected
        if previous is not None:
            assert np.array_equal(selection["synthetic_indices"][:len(previous)], previous)
        previous = selection["synthetic_indices"]
    selection, real = select_rows(1000, .999, 42, cycle)
    assert real == 1 and len(selection["synthetic_indices"]) == 999
    assert len(np.unique(selection["synthetic_indices"])) == 999


@pytest.mark.parametrize("count", [True, 0, 255, 10001, 1000.0, "1000"])
def test_invalid_smoke_scopes_rejected(count):
    c = settings("full")
    c["smoke_train_pairs"] = count
    with pytest.raises(ValueError, match="smoke_train_pairs"):
        validate_config(c)


def test_historical_smoke_scopes_and_wrong_scope_rejection():
    assert smoke_pair_count({}) == 256
    assert smoke_pair_count({"completion_arms": []}) == 320
    c = settings("smoke")
    c["train_pairs_per_cycle"] = 320
    with pytest.raises(ValueError, match="scope differs"):
        validate_config(c)


def test_sparse_targets_preserve_the_one_real_row_and_resume(tmp_path):
    prompt = np.arange(64000, dtype=np.int32).reshape(1000, 64)
    real = prompt + 1000000
    real_before = real.copy()
    selection, description = freeze_selection(tmp_path, prompt, real, .999, 42, 1, "checkpoint")
    generated = prompt[selection["synthetic_indices"]] + 2000000
    result, seal = freeze_targets(tmp_path, real, generated, selection, description, "generation")
    real_rows = np.flatnonzero(selection["source_labels"] == 0)
    assert len(real_rows) == 1
    assert np.array_equal(result[real_rows], real[real_rows])
    assert np.array_equal(result[selection["synthetic_indices"]], generated)
    assert np.array_equal(real, real_before)
    restored_selection, restored_description = freeze_selection(
        tmp_path, prompt, real, .999, 42, 1, "checkpoint", create=False)
    restored, restored_seal = freeze_targets(tmp_path, real, generated, restored_selection,
                                           restored_description, "generation", create=False)
    assert restored.tobytes() == result.tobytes() and restored_seal == seal
    assert read_json(tmp_path / "mixture/manifest.json")["scored_targets"] == 64000


@pytest.mark.parametrize("value, expected", [("99.9", 99.9), ("99", 99), ("85", 85)])
def test_decimal_cli_names_and_exact_fractions(value, expected):
    from scripts.operations.launch_high_fraction_completions import parse_percent, profile
    assert parse_percent(value) == expected
    name, fraction = profile(parse_percent(value))
    if value == "99.9":
        assert name == "mixed99p9" and fraction == .999
    else:
        assert name == "mixed" + value


@pytest.mark.parametrize("value", ["99.99", "999", "nan", "100"])
def test_unsupported_percentages_not_silently_rounded(value):
    from scripts.operations.launch_high_fraction_completions import parse_percent
    with pytest.raises(argparse.ArgumentTypeError):
        parse_percent(value)


def test_decimal_launcher_keeps_own_identity_and_restores_host_hooks(monkeypatch):
    from model_collapse import remote
    from scripts.operations.launch_high_fraction_completions import launch_one
    monkeypatch.setattr(remote, "ledger", lambda: {"launches": [], "setup_budget_usd": 3.5})
    original = remote.verify_detached
    def launch(**kwargs):
        assert kwargs["config_path"].as_posix().endswith("mixed99p9/full.yaml")
        return {"configured": True}
    monkeypatch.setattr(remote, "launch", launch)
    assert launch_one(99.9) == {"configured": True}
    assert remote.verify_detached is original


def test_report_labels_keep_99_point_9_distinct_from_fully_synthetic():
    from model_collapse.analysis.completion import fraction_label
    labels = [fraction_label(fraction) for fraction in (.95, .99, .999, 1)]
    assert labels == ["95% synthetic", "99% synthetic", "99.9% synthetic", "100% synthetic"]
