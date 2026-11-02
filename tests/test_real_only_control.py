"""A real-only control uses the historical mixer at its zero-synthetic endpoint."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from model_collapse.common import PROJECT, atomic_json, config, sha256
from model_collapse.data import load_blocks, save_array
from model_collapse.mixing import mixture_counts, mixture_input, reconstruct, sample_mixture
from model_collapse.workflow.lineage import token_hash
from model_collapse.workflow.profiles import protocol_fingerprint, require_acceptance


DEFINITIONS = PROJECT / "experiments/opt125m_wikitext2"


@pytest.mark.parametrize("kind", ["full", "smoke"])
def test_only_ratio_and_name_change_from_historical_mixed50(kind):
    historical = config(DEFINITIONS / "continued_mixed_50_final5" / f"{kind}.yaml")
    control = config(DEFINITIONS / "continued_real_only_final5" / f"{kind}.yaml")
    assert {key for key in historical.keys() | control.keys()
            if historical.get(key) != control.get(key)} == {"experiment_id", "synthetic_fraction"}
    assert control["synthetic_fraction"] == 0.0
    assert control["initialization"] == "continued"
    assert control["epochs"] == 5 and control["checkpoint_selection"] == "final_epoch"
    assert control["start_checkpoint"] == historical["start_checkpoint"]


def test_scope_and_acceptance_are_specific_to_real_only():
    full, smoke = (config(DEFINITIONS / "continued_real_only_final5" / f"{kind}.yaml")
                   for kind in ("full", "smoke"))
    assert full["generations"] == 10 and smoke["generations"] == 2
    assert all(full[f"{split}_blocks"] is None for split in ("train", "validation", "test"))
    assert [smoke[f"{split}_blocks"] for split in ("train", "validation", "test")] == [256, 64, 64]
    assert protocol_fingerprint(full) == protocol_fingerprint(smoke)
    for fraction in (.2, .5, 1):
        historical = {**full, "synthetic_fraction": fraction}
        assert protocol_fingerprint(full) != protocol_fingerprint(historical)
        with pytest.raises(RuntimeError, match="scientific protocol"):
            require_acceptance(full, {"source_code_sha256": "runtime",
                                     "protocol_fingerprint": protocol_fingerprint(historical)}, "runtime")


def test_all_full_pool_blocks_once_each_cycle_with_historical_rng_streams():
    real = np.arange(37946 * 64, dtype=np.int32).reshape(-1, 64)
    synthetic = real + 10000000
    assert mixture_counts(len(real), 0) == (37946, 0)
    previous = None
    for cycle in range(1, 11):
        blocks, selected = sample_mixture(real, synthetic, 0, 42, cycle)
        expected_ids = np.random.default_rng(np.random.SeedSequence([42, cycle, 0])).choice(
            len(real), len(real), replace=False)
        expected_order = np.random.default_rng(np.random.SeedSequence([42, cycle, 2])).permutation(len(real))
        assert selected["real_indices"] == expected_ids.tolist()
        assert selected["synthetic_indices"] == []
        assert selected["source_labels"] == [0] * len(real)
        assert selected["permutation"] == expected_order.tolist()
        assert sorted(selected["real_indices"]) == list(range(len(real)))
        assert np.array_equal(blocks, real[expected_ids][expected_order])
        assert np.array_equal(np.sort(blocks[:, 0]), real[:, 0])
        restored, labels = reconstruct(real, synthetic, selected)
        assert np.array_equal(blocks, restored) and labels == selected["source_labels"]
        if previous is not None:
            assert not np.array_equal(previous, blocks)
        previous = blocks


def test_zero_synthetic_tokens_are_independent_of_synthetic_contents():
    real = np.arange(256 * 64, dtype=np.int32).reshape(-1, 64)
    a, selection_a = sample_mixture(real, real + 100000, 0, 42, 2)
    b, selection_b = sample_mixture(real, np.full_like(real, 5), 0, 42, 2)
    assert np.array_equal(a, b) and selection_a == selection_b


def test_persisted_real_only_mixture_excludes_heldout_rows_and_freezes_on_resume(tmp_path):
    real = np.arange(256 * 64, dtype=np.int32).reshape(-1, 64)
    sources = {"train": real, "validation": real[:64] + 100000, "test": real[:64] + 200000}
    for split, values in sources.items():
        save_array(tmp_path / f"data/{split}.npy", values)
    files = {f"{split}.npy": sha256(tmp_path / f"data/{split}.npy") for split in sources}
    atomic_json(tmp_path / "data/manifest.json", {"files": files})
    synthetic = real + 300000
    source = tmp_path / "synthetic_for_gen_01"
    save_array(source / "tokens.npy", synthetic)
    manifest = {"files": {"tokens.npy": sha256(source / "tokens.npy")},
                "continuation_tokens_sha256": token_hash(synthetic),
                "provenance": {"parent_weights_sha256": "parent"}}
    atomic_json(source / "manifest.json", manifest)
    settings = {"seed": 42, "synthetic_fraction": 0.0, "initialization": "continued"}
    blocks, evidence = mixture_input(tmp_path, 1, settings, synthetic, manifest, "parent")
    assert evidence["real_blocks"] == 256 and evidence["synthetic_blocks"] == 0
    assert evidence["sources"]["real_split"] == "train"
    assert np.array_equal(np.sort(blocks[:, 0]), real[:, 0])
    assert not np.any(blocks >= 100000)
    dest = tmp_path / "mixed_for_gen_01"
    frozen = {p.name: p.read_bytes() for p in dest.iterdir()}
    repeated, verified = mixture_input(tmp_path, 1, settings, synthetic, manifest, "parent", create=False)
    assert np.array_equal(blocks, repeated) and evidence == verified
    assert frozen == {p.name: p.read_bytes() for p in dest.iterdir()}
    assert all(sha256(tmp_path / f"data/{split}.npy") == expected
               for split, expected in ((name.removesuffix(".npy"), value) for name, value in files.items()))
    assert all(np.array_equal(load_blocks(tmp_path, split), values) for split, values in sources.items())
    changed = copy.deepcopy(settings)
    changed["synthetic_fraction"] = .2
    with pytest.raises(RuntimeError, match="changed"):
        mixture_input(tmp_path, 1, changed, synthetic, manifest, "parent")
