import numpy as np
import pytest

from model_collapse.common import read_json, sha256
from model_collapse.completion_mixing import freeze_selection, freeze_targets, select_rows


@pytest.mark.parametrize("total,fraction,synthetic", [(10000, .5, 5000), (10000, .2, 2000),
                                                   (320, .5, 160), (320, .2, 64),
                                                   (5, .5, 3), (320, 0, 0), (320, 1, 320)])
def test_nested_selection_counts_and_unique_prompt_rows(total, fraction, synthetic):
    chosen, real = select_rows(total, fraction, 42, 1)
    assert len(chosen["synthetic_indices"]) == synthetic and real == total - synthetic
    assert int(chosen["source_labels"].sum()) == synthetic
    assert len(np.unique(chosen["permutation"])) == total
    assert np.array_equal(chosen["row_indices"], np.arange(total))


def test_selection_is_nested_repeatable_and_cycle_specific():
    a, _ = select_rows(10000, .2, 42, 1)
    b, _ = select_rows(10000, .5, 42, 1)
    again, _ = select_rows(10000, .2, 42, 1)
    later, _ = select_rows(10000, .2, 42, 2)
    assert np.array_equal(a["synthetic_indices"], b["synthetic_indices"][:2000])
    assert np.array_equal(a["permutation"], again["permutation"])
    assert not np.array_equal(a["permutation"], later["permutation"])


@pytest.mark.parametrize("fraction", [-.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_fractions_are_rejected(fraction):
    with pytest.raises(ValueError):
        select_rows(10, fraction, 42, 1)


@pytest.mark.parametrize("fraction", [0, .2, .5, 1])
def test_frozen_targets_reconstruct_from_original_row_mapping(tmp_path, fraction):
    prompt = np.arange(640, dtype=np.int32).reshape(10, 64)
    real = prompt + 1000
    selection, meta = freeze_selection(tmp_path, prompt, real, fraction, 42, 1, "generator")
    indices = selection["synthetic_indices"]
    generated = (prompt[indices] + 2000) if len(indices) else None
    target, manifest = freeze_targets(tmp_path, real, generated, selection, meta, "generation")
    expected = real.copy()
    if len(indices):
        expected[indices] = generated
    assert np.array_equal(target, expected)
    assert np.array_equal(prompt, np.arange(640, dtype=np.int32).reshape(10, 64))
    assert np.array_equal(real, prompt + 1000)
    assert manifest == sha256(tmp_path / "mixture/manifest.json")
    selection2, meta2 = freeze_selection(tmp_path, prompt, real, fraction, 42, 1, "generator", create=False)
    restored, _ = freeze_targets(tmp_path, real, generated, selection2, meta2, "generation", create=False)
    assert restored.tobytes() == target.tobytes()


def test_partial_selection_recovery_and_corruption_rejection(tmp_path):
    prompt = np.zeros((10, 64), dtype=np.int32)
    real = np.ones_like(prompt)
    folder = tmp_path / "mixture"
    folder.mkdir()
    np.save(folder / "source_labels.npy", np.full(10, 99))
    selection, meta = freeze_selection(tmp_path, prompt, real, .5, 42, 1, "generator")
    generated = np.full((5, 64), 7, dtype=np.int32)
    freeze_targets(tmp_path, real, generated, selection, meta, "generation")
    for fraction, parent in ((.2, "generator"), (.5, "other")):
        with pytest.raises(RuntimeError, match="selection changed"):
            freeze_selection(tmp_path, prompt, real, fraction, 42, 1, parent)
    with pytest.raises(RuntimeError, match="provenance changed"):
        freeze_targets(tmp_path, real, generated, selection, meta, "other")
    (folder / "synthetic_indices.npy").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="corrupt"):
        freeze_selection(tmp_path, prompt, real, .5, 42, 1, "generator")


def test_completed_mixture_cannot_be_created_during_verification(tmp_path):
    prompt = np.zeros((10, 64), dtype=np.int32)
    with pytest.raises(RuntimeError, match="selection is missing"):
        freeze_selection(tmp_path, prompt, prompt, .5, 42, 1, "generator", create=False)
    selection, meta = freeze_selection(tmp_path, prompt, prompt, .5, 42, 1, "generator")
    with pytest.raises(RuntimeError, match="targets are missing"):
        freeze_targets(tmp_path, prompt, np.ones((5, 64)), selection, meta, "generated", create=False)
