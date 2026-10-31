"""Pure reconstruction, immutable mixtures, lineage and compatibility regressions."""
import copy

import numpy as np
import pytest

from model_collapse.common import PROJECT, atomic_json, config, read_json, sha256, verify_files
from model_collapse.data import save_array
from model_collapse.mixing import (certificate_name, mixture_counts, mixture_enabled, mixture_input,
                                   reconstruct, sample_mixture, synthetic_fraction)
from model_collapse.workflow.lineage import token_hash, training_lineage
from model_collapse.workflow.profiles import MIXED_CHECKS, protocol_fingerprint, require_acceptance


@pytest.fixture
def sources(tmp_path):
    real = np.arange(8 * 64, dtype=np.int32).reshape(8, 64)
    synthetic = real + 10000
    save_array(tmp_path / "data/train.npy", real)
    save_array(tmp_path / "data/validation.npy", real + 20000)
    save_array(tmp_path / "data/test.npy", real + 30000)
    atomic_json(tmp_path / "data/manifest.json", {"files": {f"{s}.npy": sha256(tmp_path / f"data/{s}.npy")
                                                            for s in ("train", "validation", "test")}})
    source = tmp_path / "synthetic_for_gen_01"
    save_array(source / "tokens.npy", synthetic)
    manifest = {"files": {"tokens.npy": sha256(source / "tokens.npy")},
                "continuation_tokens_sha256": token_hash(synthetic),
                "provenance": {"parent_weights_sha256": "parent"}}
    atomic_json(source / "manifest.json", manifest)
    parent = tmp_path / "seed_42/gen_00"
    (parent / "checkpoint").mkdir(parents=True)
    (parent / "checkpoint/model.safetensors").write_bytes(b"original weights")
    atomic_json(parent / "training.json", {"selected_weights_sha256": "parent",
                "checkpoint_files": {"model.safetensors": sha256(parent / "checkpoint/model.safetensors")}})
    c = {"seed": 42, "initialization": "continued", "synthetic_fraction": .5}
    return tmp_path, c, real, synthetic, manifest


@pytest.mark.parametrize("value", [True, False, -.1, 1.1, float("nan"), float("inf"), "0.5", None])
def test_fraction_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="finite number"):
        synthetic_fraction({"synthetic_fraction": value})


def test_counts_full_smoke_rounding_and_legacy():
    assert mixture_counts(37946, .5) == (18973, 18973)
    assert mixture_counts(256, .5) == (128, 128)
    assert mixture_counts(5, .5) == (2, 3)
    assert mixture_counts(37946, .2) == (30357, 7589)
    assert synthetic_fraction({}) == 1 and not mixture_enabled({})
    assert not mixture_enabled({"synthetic_fraction": 1})


@pytest.mark.parametrize("fraction", [0, .2, .5, 1])
def test_sampling_reconstructs_exactly_without_replacement(fraction):
    real = np.arange(37946 * 64, dtype=np.int32).reshape(-1, 64)
    synthetic = real + 10000000
    rows, selection = sample_mixture(real, synthetic, fraction, 42, 1)
    rebuilt, labels = reconstruct(real, synthetic, selection)
    assert np.array_equal(rows, rebuilt) and labels == selection["source_labels"]
    n_real, n_synthetic = mixture_counts(len(real), fraction)
    assert labels.count(0) == n_real and labels.count(1) == n_synthetic
    assert len(set(selection["real_indices"])) == n_real
    assert len(set(selection["synthetic_indices"])) == n_synthetic
    assert sorted(selection["permutation"]) == list(range(len(real)))
    again, same = sample_mixture(real, synthetic, fraction, 42, 1)
    assert np.array_equal(rows, again) and same == selection
    if 0 < fraction < 1:
        assert labels != sorted(labels)


def test_cycle_streams_change_but_do_not_depend_on_other_calls():
    real = np.arange(100 * 64, dtype=np.int32).reshape(-1, 64)
    _, first = sample_mixture(real, real + 10000, .5, 42, 1)
    _, second = sample_mixture(real, real + 10000, .5, 42, 2)
    _, again = sample_mixture(real, real + 10000, .5, 42, 1)
    assert first == again and first["real_indices"] != second["real_indices"]
    assert first["synthetic_indices"] != second["synthetic_indices"]


def test_insufficient_source_or_wrong_shape_is_rejected():
    rows = np.zeros((8, 64), dtype=np.int32)
    with pytest.raises(ValueError, match="Insufficient"):
        sample_mixture(rows, rows[:3], .5, 42, 1)
    with pytest.raises(ValueError, match="64-token"):
        sample_mixture(rows[:, :63], rows, .5, 42, 1)
    with pytest.raises(ValueError, match="int32"):
        sample_mixture(rows.astype(np.int64), rows, .5, 42, 1)


def test_persisted_mixture_and_resume_preserve_bytes(sources):
    root, c, real, synthetic, manifest = sources
    a, first = mixture_input(root, 1, c, synthetic, manifest, "parent")
    dest = root / "mixed_for_gen_01"
    before = {p.name: p.read_bytes() for p in dest.iterdir()}
    b, second = mixture_input(root, 1, c, synthetic, manifest, "parent")
    assert first == second and np.array_equal(a, b)
    assert before == {p.name: p.read_bytes() for p in dest.iterdir()}
    assert first["real_blocks"] == first["synthetic_blocks"] == 4
    assert first["sources"]["real_split"] == "train"
    assert not np.any(a >= 20000), "Validation/test source rows must never enter training"


def test_partial_mixture_reconstructs_after_interruption(sources, monkeypatch):
    from model_collapse import mixing
    root, c, real, synthetic, manifest = sources
    normal = mixing.atomic_json
    def interrupt(path, value):
        if str(path).endswith("mixed_for_gen_01/manifest.json") or str(path).endswith("mixed_for_gen_01\\manifest.json"):
            raise RuntimeError("interrupted before commit")
        return normal(path, value)
    monkeypatch.setattr(mixing, "atomic_json", interrupt)
    with pytest.raises(RuntimeError, match="before commit"):
        mixture_input(root, 1, c, synthetic, manifest, "parent")
    assert not (root / "mixed_for_gen_01/manifest.json").exists()
    monkeypatch.setattr(mixing, "atomic_json", normal)
    rows, complete = mixture_input(root, 1, c, synthetic, manifest, "parent")
    expected, _ = sample_mixture(real, synthetic, .5, 42, 1)
    assert np.array_equal(rows, expected)
    verify_files(root / "mixed_for_gen_01", complete["files"])


@pytest.mark.parametrize("artifact", ["tokens.npy", "selection.json", "manifest.json"])
def test_corrupted_mixture_is_rejected(sources, artifact):
    root, c, _, synthetic, manifest = sources
    mixture_input(root, 1, c, synthetic, manifest, "parent")
    (root / "mixed_for_gen_01" / artifact).write_bytes(b"corrupt")
    with pytest.raises((RuntimeError, ValueError)):
        mixture_input(root, 1, c, synthetic, manifest, "parent")


def test_coherently_rehashed_wrong_selection_is_rejected(sources):
    root, c, _, synthetic, manifest = sources
    mixture_input(root, 1, c, synthetic, manifest, "parent")
    dest = root / "mixed_for_gen_01"
    selected = read_json(dest / "selection.json")
    selected["source_labels"][0] = 1 - selected["source_labels"][0]
    atomic_json(dest / "selection.json", selected)
    old = read_json(dest / "manifest.json")
    old["files"]["selection.json"] = sha256(dest / "selection.json")
    atomic_json(dest / "manifest.json", old)
    with pytest.raises(RuntimeError, match="reconstruction"):
        mixture_input(root, 1, c, synthetic, manifest, "parent")


@pytest.mark.parametrize("field,value", [("synthetic_fraction", .2), ("seed", 43)])
def test_changed_sampling_contract_is_rejected(sources, field, value):
    root, c, _, synthetic, manifest = sources
    mixture_input(root, 1, c, synthetic, manifest, "parent")
    with pytest.raises(RuntimeError, match="changed"):
        mixture_input(root, 1, {**c, field: value}, synthetic, manifest, "parent")


def test_wrong_generator_and_missing_completed_mixture_rejected(sources):
    root, c, _, synthetic, manifest = sources
    with pytest.raises(RuntimeError, match="generator"):
        mixture_input(root, 1, c, synthetic, manifest, "wrong")
    with pytest.raises(RuntimeError, match="completion manifest"):
        mixture_input(root, 1, c, synthetic, manifest, "parent", create=False)
    atomic_json(root / "seed_42/gen_01/training.json", {"finished": True})
    with pytest.raises(RuntimeError, match="completion manifest"):
        training_lineage(root, 1, c, None)


def test_lineage_distinguishes_pool_mixture_and_initializer(sources):
    root, c, _, synthetic, manifest = sources
    rows, initializer, checkpoint, lineage = training_lineage(root, 1, c, None)
    assert initializer == "parent" and checkpoint == root / "seed_42/gen_00/checkpoint"
    assert lineage["generator_weights_sha256"] == initializer
    assert lineage["training_tokens_sha256"] == token_hash(rows) != token_hash(synthetic)
    assert lineage["synthetic_source_tokens_sha256"] == token_hash(synthetic)
    assert lineage["training_dataset"] == "mixed_for_gen_01"
    original = {**c, "synthetic_fraction": 1.0}
    rows, _, _, historical = training_lineage(root, 1, original, None)
    assert np.array_equal(rows, synthetic) and "mixture" not in historical


def test_profiles_and_acceptance_are_mixture_specific():
    directory = PROJECT / "experiments/opt125m_wikitext2/continued_mixed_50_final5"
    full, smoke = (config(directory / f"{kind}.yaml") for kind in ("full", "smoke"))
    assert full["epochs"] == smoke["epochs"] == 5
    assert full["checkpoint_selection"] == smoke["checkpoint_selection"] == "final_epoch"
    assert full["start_checkpoint"] == smoke["start_checkpoint"]
    assert full["initial_dataset"] == smoke["initial_dataset"]
    assert protocol_fingerprint(full) == protocol_fingerprint(smoke)
    historical = {**full}
    historical.pop("synthetic_fraction")
    assert protocol_fingerprint(full) != protocol_fingerprint(historical)
    assert certificate_name(full) != certificate_name(historical)
    cert = {"source_code_sha256": "runtime", "protocol_fingerprint": protocol_fingerprint(full), "checks": {}}
    with pytest.raises(RuntimeError, match="mixed-data"):
        require_acceptance(full, cert, "runtime")
    cert["checks"] = dict.fromkeys(MIXED_CHECKS, True)
    require_acceptance(full, cert, "runtime")
    with pytest.raises(RuntimeError, match="protocol"):
        require_acceptance({**full, "synthetic_fraction": .2}, cert, "runtime")


def test_full_forecast_excludes_imported_d1():
    from model_collapse.analysis.forecasts import forecast
    bench = {"full_split_blocks": {"train": 37946, "validation": 3922, "test": 4494},
             "training_target_tokens_per_second": 33000, "generation_new_tokens_per_second": 1300}
    estimate = forecast(bench, 10, .58, initialization="continued", epochs=5, imported_d1=True)
    assert estimate["new_training_stages"] == 10
    assert estimate["new_full_generation_stages"] == 9


def test_mixed_campaign_budget_blocks_before_provisioning(tmp_path, monkeypatch):
    from model_collapse import remote
    from model_collapse.workflow import initial_data
    from model_collapse.mixing import CAMPAIGN
    class API:
        def request(self, method, path, body=None):
            assert method == "GET", "An exhausted campaign must not provision"
            return {"pods": []}
        def candidates(self):
            return [{"gpu": "NVIDIA RTX PRO 4000 Blackwell", "data_center": "EU-RO-1", "hourly_usd": .57}]
    monkeypatch.setattr(remote, "Runpod", API)
    monkeypatch.setattr(remote, "STATE", tmp_path)
    monkeypatch.setattr(remote, "preflight", lambda **kw: {"account": {"clientBalance": 30}})
    monkeypatch.setattr(remote, "ledger", lambda: {"launches": []})
    monkeypatch.setattr(remote, "source_archive", lambda: ("unused", "runtime"))
    monkeypatch.setattr(initial_data, "resolve_initial_dataset", lambda c: c)
    calls = []
    monkeypatch.setattr(remote, "campaign_accrued", lambda d, campaign: calls.append(campaign) or .75)
    with pytest.raises(RuntimeError, match="Insufficient budget"):
        remote.launch(smoke=True, config_path=PROJECT / "experiments/opt125m_wikitext2/continued_mixed_50_final5/smoke.yaml")
    assert calls == [CAMPAIGN]


def test_changed_real_source_and_generator_are_rejected(sources):
    root, c, real, synthetic, manifest = sources
    mixture_input(root, 1, c, synthetic, manifest, "parent")
    save_array(root / "data/train.npy", real + 1)
    with pytest.raises(RuntimeError, match="corrupt"):
        mixture_input(root, 1, c, synthetic, manifest, "parent")
    data_manifest = read_json(root / "data/manifest.json")
    data_manifest["files"]["train.npy"] = sha256(root / "data/train.npy")
    atomic_json(root / "data/manifest.json", data_manifest)
    with pytest.raises(RuntimeError, match="source identity changed"):
        mixture_input(root, 1, c, synthetic, manifest, "parent")


def test_probe_diagnostics_include_within_output_repetition_and_blank_lines():
    from model_collapse.science.metrics import probe_diagnostics
    values = np.stack((np.full(64, 50118, dtype=np.int32), np.arange(64, dtype=np.int32)))
    measured = probe_diagnostics(values)
    assert measured["newline_token_50118_count"] == 64
    assert measured["newline_token_50118_share"] == .5
    assert measured["all_newline_outputs"] == measured["newline_heavy_outputs_ge32"] == 1
    assert measured["mean_within_repeated_4gram_rate"] == pytest.approx((60/61)/2)
    assert measured["mean_within_entropy_nats"] == pytest.approx(np.log(64)/2)


def test_accepted_mixed_full_run_refuses_unaffordable_launch_before_provisioning(tmp_path, monkeypatch):
    from model_collapse import remote, pipeline
    from model_collapse.workflow import initial_data
    full_path = PROJECT / "experiments/opt125m_wikitext2/continued_mixed_50_final5/full.yaml"
    c = config(full_path)
    class API:
        def request(self, method, path, body=None):
            assert method == "GET" and path == "/pods", "Unaffordable full compute must never provision"
            return {"pods": []}
        def candidates(self):
            return [{"gpu": "NVIDIA RTX PRO 4000 Blackwell", "data_center": "EU-RO-1", "hourly_usd": .57}]
    monkeypatch.setattr(remote, "Runpod", API)
    monkeypatch.setattr(remote, "STATE", tmp_path)
    monkeypatch.setattr(remote, "preflight", lambda **kw: {"account": {"clientBalance": 30}})
    monkeypatch.setattr(remote, "ledger", lambda: {"launches": []})
    monkeypatch.setattr(remote, "accrued", lambda d, kind=None: 20.8)
    monkeypatch.setattr(remote, "source_archive", lambda: ("unused", "runtime"))
    monkeypatch.setattr(initial_data, "resolve_initial_dataset", lambda value: value)
    cert = {"source_code_sha256": "runtime", "protocol_fingerprint": protocol_fingerprint(c),
            "checks": dict.fromkeys(MIXED_CHECKS, True)}
    atomic_json(tmp_path / certificate_name(c), cert)
    atomic_json(tmp_path / "benchmark.json", {
        "full_split_blocks": {"train": 37946, "validation": 3922, "test": 4494},
        "training_target_tokens_per_second": 34047, "generation_new_tokens_per_second": 1306})
    real_forecast = pipeline.forecast
    def checked_forecast(*args, **kwargs):
        assert kwargs["epochs"] == 5 and kwargs["imported_d1"] is True
        result = real_forecast(*args, **kwargs)
        assert result["new_training_stages"] == 10 and result["new_full_generation_stages"] == 9
        return result
    monkeypatch.setattr(pipeline, "forecast", checked_forecast)
    with pytest.raises(RuntimeError, match="Conservative full-run estimate exceeds"):
        remote.launch(generations=10, config_path=full_path)
