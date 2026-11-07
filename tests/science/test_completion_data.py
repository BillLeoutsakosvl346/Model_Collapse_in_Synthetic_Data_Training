"""Completion-bank selection and mask regressions, with no network access."""
import json

import numpy as np
import pytest

from model_collapse import completion_data as data
from model_collapse.common import read_json, sha256


class Tokenizer:
    bos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return list(map(int, text.split()))

    def decode(self, values):
        return " ".join(map(str, values))


@pytest.fixture
def bank_setup(tmp_path):
    source = tmp_path / "source"
    for role, offset in (("validation", 10000), ("test", 20000)):
        folder = source / "fineweb_edu" / role
        folder.mkdir(parents=True)
        rows = [{"id": role + str(i), "text": " ".join(map(str, range(offset + i * 400, offset + i * 400 + 300))),
                 "used_source_tokens": 300} for i in range(3)]
        (folder / "documents.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        if role == "test":
            np.save(folder / "tokens.npy", np.ones((3, 512), dtype=np.int32))
            np.save(folder / "masks.npy", np.ones((3, 512), dtype=np.int32))
    c = {key: "pinned" for key in ("model", "model_revision", "dataset", "dataset_revision", "dataset_config")}
    c.update(dataset_files=["sample"], seed=42, shuffle_buffer=3, full_cycles=3, full_train_pairs=4,
             corpus_bank={"manifest_sha256": "x"}, secondary_corpora=["fineweb_edu"])
    rows = [{"id": "train" + str(i), "text": " ".join(map(str, range(30000 + i * 400, 30000 + i * 400 + 300)))} for i in range(15)]
    return source, c, rows


def test_exact_pairs_reconstruction_separation_and_repeatability(tmp_path, bank_setup):
    source, c, rows = bank_setup
    root = tmp_path / "bank"
    result = data.prepare_bank(root, c, Tokenizer(), source=source, iterator=rows)
    assert all(result["collections"][f"cycle_{i:02d}"]["pairs"] == 4 for i in range(1, 4))
    assert data.audit_bank(root, Tokenizer())["all_pairs_reconstructed"]
    before = sha256(root / "manifest.json")
    data.prepare_bank(root, c, Tokenizer(), source=source, iterator=[])
    assert sha256(root / "manifest.json") == before
    other = tmp_path / "other"
    second = data.prepare_bank(other, c, Tokenizer(), source=source, iterator=rows)
    assert {k: v["files"] for k, v in result["collections"].items()} == {k: v["files"] for k, v in second["collections"].items()}


def test_attention_and_scoring_masks_have_distinct_roles():
    prompt = np.arange(128).reshape(2, 64)
    target = prompt + 1000
    ids, attention, scoring = data.assemble(prompt, target, 2)
    assert ids.shape == (2, 129)
    assert np.all(attention == 1)
    assert np.all(scoring[:, :65] == 0) and np.all(scoring[:, 65:] == 1)
    assert scoring[:, 1:].sum() == 128
    assert np.array_equal(ids[:, 1:65], prompt) and np.array_equal(ids[:, 65:], target)


def test_recovery_changed_settings_and_corruption(tmp_path, bank_setup, monkeypatch):
    source, c, rows = bank_setup
    root = tmp_path / "bank"
    original = data._write_collection
    def fail(destination, *args):
        if destination.name == "cycle_02":
            raise OSError("injected")
        return original(destination, *args)
    monkeypatch.setattr(data, "_write_collection", fail)
    with pytest.raises(OSError, match="injected"):
        data.prepare_bank(root, c, Tokenizer(), source=source, iterator=rows)
    before = sha256(root / "cycle_01/manifest.json")
    monkeypatch.setattr(data, "_write_collection", original)
    data.prepare_bank(root, c, Tokenizer(), source=source, iterator=rows)
    assert sha256(root / "cycle_01/manifest.json") == before
    assert data.audit_bank(root, Tokenizer())["disjoint_documents_and_passages"]
    with pytest.raises(RuntimeError, match="configuration"):
        data.prepare_bank(root, {**c, "seed": 43}, Tokenizer(), source=source, iterator=rows)
    (root / "cycle_02/targets.npy").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="corrupt"):
        data.prepare_bank(root, c, Tokenizer(), source=source, iterator=rows)


def test_reserved_documents_and_embedded_passages_are_excluded(tmp_path, bank_setup):
    source, c, rows = bank_setup
    protected = next(data.records(source / "fineweb_edu/test/documents.jsonl"))
    candidate = {"id": "different", "text": "999 " + protected["text"]}
    root = tmp_path / "bank"
    m = data.prepare_bank(root, c, Tokenizer(), source=source, iterator=[protected, candidate, *rows])
    assert m["rejected"]["identity"] == m["rejected"]["passage"] == 1
