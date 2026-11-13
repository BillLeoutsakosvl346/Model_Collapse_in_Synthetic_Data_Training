from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from model_collapse.common import atomic_json, read_json, sha256
from model_collapse.workflow import sgd_calibration as calibration
from model_collapse.workflow import sgd_profile as profile


def definition(kind="smoke"):
    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load((root / "comparisons/smollm2_sgd_greedy_effects" / (kind + ".yaml")).read_text())


@pytest.mark.parametrize("kind", ["smoke", "full"])
def test_named_definitions_are_pending_but_valid(kind):
    c = definition(kind)
    profile.validate(c)
    assert [x["id"] for x in profile.specs(c)] == ["real", "mixed99", "synthetic"]
    with pytest.raises(RuntimeError, match="Freeze"):
        profile.require_calibration(c)
    assert c["precision"] == "fp32" and c["learning_rate_policy"] == "constant"


@pytest.mark.parametrize("key,value", [("momentum", .9), ("weight_decay", .01), ("num_beams", 5),
    ("precision", "bf16"), ("scheduler_policy", "cosine"), ("foreach", True), ("fused", True),
    ("microbatch", 16), ("gradient_clipping", 1.), ("generation_decoder", "beam")])
def test_new_profile_rejects_changed_science(key, value):
    c = definition(); c[key] = value
    with pytest.raises(ValueError): profile.validate(c)


def test_legacy_seven_arm_profile_not_redefined():
    from model_collapse.workflow import smollm
    assert len(smollm.ARMS) == 7
    assert not smollm.enabled(definition())
    assert not profile.enabled({"completion_profile": smollm.PROFILE})


def test_rate_lock_binds_input_and_implementation():
    c = definition()
    receipt = {"complete": True, "selection_policy": calibration.POLICY, "selected_learning_rate": .001,
               "contract_sha256": calibration.calibration_contract(c)}
    resolved = profile.apply_calibration(c, receipt, "a" * 64)
    profile.validate(resolved); profile.require_calibration(resolved)
    assert resolved["learning_rate"] == resolved["min_learning_rate"] == .001
    assert c["calibration_required"] is True
    changed = copy.deepcopy(c); changed["prepared_bank"]["manifest_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="different calibration"):
        profile.apply_calibration(changed, receipt, "a" * 64)


def test_sampler_is_deterministic_distinct_and_cycle_specific():
    first = calibration.sample_indices(42, 1)
    assert len(first) == len(set(first.tolist())) == 4096
    assert np.array_equal(first, calibration.sample_indices(42, 1))
    assert not np.array_equal(first, calibration.sample_indices(42, 2))
    expected = np.random.default_rng(np.random.SeedSequence([42, 1, 7731])).permutation(10000)[:4096]
    assert np.array_equal(first, expected)
    with pytest.raises(ValueError): calibration.sample_indices(42, 1, 100)


def test_source_contract_ignores_only_checkout_line_endings(tmp_path):
    left, right = tmp_path / "left.py", tmp_path / "right.py"
    left.write_bytes(b"x = 1\n"); right.write_bytes(b"x = 1\r\n")
    assert calibration._source_hash(left) == calibration._source_hash(right)
    right.write_bytes(b"x = 2\r\n")
    assert calibration._source_hash(left) != calibration._source_hash(right)


def test_rank_filters_unstable_candidates_and_prefers_smaller_near_tie():
    rows = [{"learning_rate": .01, "validation_nll": 3., "eligible": True},
            {"learning_rate": .001, "validation_nll": 3.000005, "eligible": True},
            {"learning_rate": .1, "validation_nll": 2., "eligible": False}]
    assert [r["learning_rate"] for r in calibration.rank_candidates(rows)] == [.001, .01]
    base = {"training_nll": 2.9, "baseline_training_nll": 3., "validation_nll": 3.,
            "baseline_validation_nll": 3., "weights_changed": True}
    assert calibration.eligible(base)
    assert not calibration.eligible({**base, "validation_nll": 3.1})
    assert not calibration.eligible({**base, "training_nll": 3.})
    assert not calibration.eligible({**base, "weights_changed": False})


def fake_environment(monkeypatch, tmp_path, *, all_bad=False, interrupt=False):
    import torch
    from model_collapse.science import engine
    c = definition()
    bank = tmp_path / "bank"; bank.mkdir()
    atomic_json(bank / "manifest.json", {"bos_token_id": 0, "collections": {
        f"cycle_{n:02d}": {"files": {x: "a" * 64 for x in ("prompts.npy", "targets.npy", "windows.npy")}}
        for n in (1, 2)}})
    c["prepared_bank"]["manifest_sha256"] = sha256(bank / "manifest.json")
    for n in (1, 2):
        folder = bank / f"cycle_{n:02d}";folder.mkdir();atomic_json(folder / "manifest.json", {"cycle": n})
    (bank / "validation").mkdir()
    np.save(bank / "validation/windows.npy", np.column_stack([np.arange(1100), np.zeros(1100, dtype=int)]))
    reads, calls = [], []

    def arrays(_bank, split, rows=None):
        reads.append(split)
        assert split in ("validation", "cycle_01", "cycle_02"), "test data used in calibration"
        n = 1100 if split == "validation" else 10000
        return np.ones((n, 64), dtype=np.int32), np.full((n, 64), 2, dtype=np.int32)
    monkeypatch.setattr(calibration, "_arrays", arrays)
    monkeypatch.setattr(calibration, "verify_files", lambda *a, **kw: None)

    class Model:
        def __init__(self, identity=0, rate=0., cycle=0): self.identity, self.rate, self.cycle = identity, rate, cycle
        def save_pretrained(self, path, **kwargs):
            path = Path(path);path.mkdir(parents=True, exist_ok=True)
            atomic_json(path / "weights.json", {"identity": self.identity, "rate": self.rate, "cycle": self.cycle})
    def load_model(_c, checkpoint=None):
        return Model(**read_json(Path(checkpoint) / "weights.json")) if checkpoint else Model()
    def weights_hash(model): return f"{model.identity:064x}"
    def evaluate(model, tokens, _c, **kwargs):
        loss = 3. if not model.cycle else (3.4 if all_bad or model.rate == .1 else 3. - model.cycle * {1e-4: .001, .001: .03, .01: .1}[model.rate])
        return {"loss": loss, "perplexity": float(np.exp(loss)), "scored_tokens": len(tokens) * 64}
    def train(root, generation, blocks, val, cfg, expected_base_hash=None, *, progress_callback=None, **kwargs):
        calls.append((cfg["learning_rate"], generation, kwargs.get("optimizer_checkpoint")))
        assert cfg["purpose"] == "sgd_calibration" and cfg["update_diagnostics"] == {"enabled": False}
        assert len(blocks) == 4096 and len(val) == 1024 and kwargs["total_steps"] == 64
        assert weights_hash(load_model(cfg, kwargs["initial_checkpoint"])) == expected_base_hash
        if interrupt: raise RuntimeError("simulated infrastructure interruption")
        if generation == 2:
            parent = torch.load(kwargs["optimizer_checkpoint"], weights_only=False)
            assert parent["global_step"] == 32
        for step in range((generation - 1) * 32 + 1, generation * 32 + 1):
            if progress_callback: progress_callback(step, generation=generation)
        stage = Path(root) / f"seed_42/gen_{generation:02d}"
        model = Model(int(cfg["learning_rate"] * 1e8) + generation, cfg["learning_rate"], generation)
        model.save_pretrained(stage / "checkpoint")
        (stage / "optimizer_state").mkdir()
        torch.save({"global_step": generation * 32, "scheduler": None,
                    "optimizer": {"state": {}, "param_groups": [{"lr": cfg["learning_rate"]}]}}, stage / "optimizer_state/cycle.pt")
        atomic_json(stage / "training_history.json", [{"learning_rate": cfg["learning_rate"], "learning_rate_end": cfg["learning_rate"]}])
        result = {"selected_weights_sha256": weights_hash(model), "global_step": generation * 32,
                  "initial_weights_sha256": expected_base_hash,
                  "checkpoint_files": {"weights.json": sha256(stage / "checkpoint/weights.json")},
                  "state_files": {"optimizer_state/cycle.pt": sha256(stage / "optimizer_state/cycle.pt")}}
        atomic_json(stage / "training.json", result)
        return result
    for name, value in (("load_model", load_model), ("weights_hash", weights_hash), ("evaluate", evaluate), ("train", train)):
        monkeypatch.setattr(engine, name, value)
    return c, bank, calls, reads


def test_full_bounded_calibration_uses_only_real_train_and_validation(monkeypatch, tmp_path):
    c, bank, calls, reads = fake_environment(monkeypatch, tmp_path)
    root = tmp_path / "run"
    receipt = calibration.calibrate(root, c, bank)
    assert receipt["selected_learning_rate"] == .01
    assert receipt["recorded_physical_updates"] == receipt["reserved_update_upper_bound"] == 192
    assert len(calls) == 6 and [v[1] for v in calls] == [1, 1, 1, 1, 2, 2]
    assert all(v[2] is None for v in calls[:4]) and all(v[2] is not None for v in calls[4:])
    assert set(reads) == {"validation", "cycle_01", "cycle_02"}
    assert receipt["test_used"] is False and "not_a_calibration" in receipt["production_initializer"]
    assert calibration.calibrate(root, c, bank) == receipt
    assert len(calls) == 6


def test_no_viable_candidate_does_not_expand_search(monkeypatch, tmp_path):
    c, bank, calls, _ = fake_environment(monkeypatch, tmp_path, all_bad=True)
    with pytest.raises(calibration.CalibrationBlocked, match="No candidate"):
        calibration.calibrate(tmp_path / "run", c, bank)
    assert len(calls) == 4
    assert not (tmp_path / "run/calibration/selection.json").exists()


def test_corrupted_completed_candidate_is_not_used_for_rate_ranking(monkeypatch, tmp_path):
    c, bank, calls, _ = fake_environment(monkeypatch, tmp_path)
    root = tmp_path / "run"
    calibration.calibrate(root, c, bank)
    (root / "calibration/selection.json").unlink()
    path = root / "calibration/lr_1em04_screen.json"
    altered = read_json(path); altered["validation_nll"] = .5; atomic_json(path, altered)
    with pytest.raises(RuntimeError, match="corrupt or unsealed"):
        calibration.calibrate(root, c, bank)
    assert len(calls) == 6


def test_interrupted_calibration_does_not_repeat_reserved_updates(monkeypatch, tmp_path):
    c, bank, calls, _ = fake_environment(monkeypatch, tmp_path, interrupt=True)
    root = tmp_path / "run"
    with pytest.raises(RuntimeError, match="infrastructure"):
        calibration.calibrate(root, c, bank)
    with pytest.raises(calibration.CalibrationBlocked, match="cannot be replayed"):
        calibration.calibrate(root, c, bank)
    assert len(calls) == 1


def test_smoke_backup_uses_new_namespace_without_mutating_source(monkeypatch):
    original = {"backend": "runpod_s3", "bucket": "abcde12345", "region": "EU-RO-1", "prefix": "historical"}
    monkeypatch.setattr(profile, "read_json", lambda path: original)
    from model_collapse.cloud import s3_artifacts
    class API:
        def request(self, *args): return {"size": 200, "dataCenter": "EU-RO-1"}
    c = definition();resolved = profile.resolve_backup(c, API())
    assert resolved["backup_store"]["prefix"] == "historical-sgd-greedy-effects-setup"
    s3_artifacts.validate_store(resolved["backup_store"])
    assert original["prefix"] == "historical" and "backup_store" not in c


def test_setup_prefix_keeps_s3_limit_and_distinguishes_long_sources(monkeypatch):
    from model_collapse.cloud.s3_artifacts import validate_store
    class API:
        def request(self, *args): return {"size": 200, "dataCenter": "EU-RO-1"}
    results = []
    for ending in ("a", "b"):
        source = {"backend": "runpod_s3", "bucket": "abcde12345", "region": "EU-RO-1", "prefix": "x" * 99 + ending}
        monkeypatch.setattr(profile, "read_json", lambda path, value=source: value)
        store = profile.resolve_backup(definition(), API())["backup_store"]
        validate_store(store); assert len(store["prefix"]) <= 100
        results.append(store["prefix"])
    assert results[0] != results[1]
