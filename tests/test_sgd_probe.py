"""Exercise the real completion-bank API, whose third return is a manifest."""
import numpy as np
import pytest

from model_collapse.common import atomic_json, sha256
from model_collapse.workflow.completion import sgd_update_probe


def test_update_probe_reads_document_rows_from_verified_windows(tmp_path):
    folder = tmp_path / "validation"
    folder.mkdir()
    prompts = np.arange(128 * 64, dtype=np.int32).reshape(128, 64) + 2
    windows = np.column_stack((np.arange(128) + 900, np.zeros(128, dtype=int)))
    for name, values in (("prompts.npy", prompts), ("targets.npy", prompts + 1), ("windows.npy", windows)):
        np.save(folder / name, values)
    atomic_json(folder / "manifest.json", {"pairs": 128,
        "files": {name: sha256(folder / name) for name in ("prompts.npy", "targets.npy", "windows.npy")}})
    np.save(tmp_path / "probe_indices.npy", np.arange(127, -1, -1))
    c = {"bos_token_id": 0, "prepared_bank": {"manifest_sha256": "b" * 64}}
    probe = sgd_update_probe(tmp_path, c)
    assert np.array_equal(probe["documents"], windows[::-1, 0])
    assert np.array_equal(probe["tokens"][:, 1:65], prompts[::-1])
    assert np.all(probe["attention"] == 1)
    assert np.all(probe["scoring"][:, :65] == 0)
    assert np.all(probe["scoring"][:, 65:] == 1)
    assert np.all(probe["scoring"].sum(1) == 64)
    assert sgd_update_probe(tmp_path, c)["identity"] == probe["identity"]
    np.save(tmp_path / "probe_indices.npy", np.zeros(128, dtype=int))
    with pytest.raises(RuntimeError, match="distinct"):
        sgd_update_probe(tmp_path, c)
