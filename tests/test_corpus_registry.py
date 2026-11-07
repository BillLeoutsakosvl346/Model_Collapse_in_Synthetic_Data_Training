"""Corpus-bank split, leakage, quota and recovery regressions. No cloud access."""
import json
from pathlib import Path

import numpy as np
import pytest

from model_collapse.common import read_json, sha256
from scripts.data import readers, registry


class Tokenizer:
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0

    def encode(self, text, add_special_tokens=False):
        return list(map(int, text.split()))

    def decode(self, tokens):
        return " ".join(map(str, tokens))


@pytest.fixture
def settings():
    return {"seed": 42, "max_sequence_length": 72, "maximum_document_tokens": 128,
            "train_source_tokens": 128, "validation_source_tokens": 96, "test_source_tokens": 96,
            "shuffle_buffer": 3, "sources": {"alpha": {}, "beta": {}}}


def rows(name, role, c, coverage, project):
    offset = (1 if name == "alpha" else 2) * 100000 + {"train": 0, "validation": 10000, "test": 20000}[role]
    for i in range(8):
        yield {"id": f"{name}:{role}:{i}", "url": None,
               "text": " ".join(map(str, range(offset + i * 200, offset + i * 200 + 100)))}


def test_bank_quotas_masks_and_deterministic_reuse(tmp_path, settings):
    result = registry.prepare(tmp_path, settings, Tokenizer(), tmp_path, reader=rows, protected=(set(), set()))
    assert len(result["collections"]) == 6
    for key, meta in result["collections"].items():
        tokens = np.load(tmp_path / key / "tokens.npy")
        masks = np.load(tmp_path / key / "masks.npy")
        assert meta["source_text_tokens"] == settings[f"{meta['role']}_source_tokens"]
        assert tokens.shape == masks.shape and meta["scored_tokens"] == masks[:, 1:].sum()
        assert np.all(tokens[masks == 0] == 0)
        assert np.all(masks[:, 1:] <= masks[:, :-1])
    before = sha256(tmp_path / "manifest.json")
    registry.prepare(tmp_path, settings, Tokenizer(), tmp_path, reader=lambda *args: (_ for _ in ()).throw(AssertionError()), protected=(set(), set()))
    assert sha256(tmp_path / "manifest.json") == before
    other = tmp_path / "reproduced"
    second = registry.prepare(other, settings, Tokenizer(), tmp_path, reader=rows, protected=(set(), set()))
    for key in result["collections"]:
        assert result["collections"][key]["files"] == second["collections"][key]["files"]


def test_actual_overlap_rejected_before_training_pool(tmp_path, settings):
    common = next(rows("alpha", "validation", settings, {}, tmp_path))
    protected_row = {"id": "historical", "url": None, "text": " ".join(map(str, range(900000, 900100)))}
    def candidates(name, role, c, coverage, project):
        if name == "beta" and role == "test":
            yield protected_row
        if name == "alpha" and role == "train":
            yield common
            yield {**common, "id": "different-id", "text": "999 " + common["text"]}
        yield from rows(name, role, c, coverage, project)
    result = registry.prepare(tmp_path, settings, Tokenizer(), tmp_path, reader=candidates,
                              protected=({"id:historical"}, set()))
    assert result["collections"]["beta/test"]["rejected_documents"]["historical_training_overlap"] == 1
    rejected = result["collections"]["alpha/train"]["rejected_documents"]
    assert rejected["duplicate_identity"] == rejected["cross_collection_passage"] == 1


def test_interrupted_preparation_preserves_completed_collections(tmp_path, settings, monkeypatch):
    original = registry.save_array
    calls = 0
    def interrupted(path, array):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected interruption")
        return original(path, array)
    monkeypatch.setattr(registry, "save_array", interrupted)
    with pytest.raises(OSError, match="injected"):
        registry.prepare(tmp_path, settings, Tokenizer(), tmp_path, reader=rows, protected=(set(), set()))
    selected = tmp_path / "alpha/validation/manifest.json"
    before = sha256(selected)
    monkeypatch.setattr(registry, "save_array", original)
    registry.prepare(tmp_path, settings, Tokenizer(), tmp_path, reader=rows, protected=(set(), set()))
    assert sha256(selected) == before
    (tmp_path / "alpha/train/masks.npy").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="corrupt"):
        registry.prepare(tmp_path, settings, Tokenizer(), tmp_path, reader=rows, protected=(set(), set()))


def test_changed_configuration_rejected(tmp_path, settings):
    registry.prepare(tmp_path, settings, Tokenizer(), tmp_path, reader=rows, protected=(set(), set()))
    with pytest.raises(RuntimeError, match="configuration"):
        registry.prepare(tmp_path, {**settings, "seed": 43}, Tokenizer(), tmp_path, reader=rows, protected=(set(), set()))


def test_wikitext_reassembles_whole_articles_across_files():
    values = ["\n", " = First = \n", "body\n", " == Section == \n", "tail\n", " = Second = \n", "last\n"]
    stream = ({"text": text, "source_file": str(i // 4), "source_row": i % 4} for i, text in enumerate(values))
    documents = list(readers.wiki_documents(stream))
    assert len(documents) == 2
    assert documents[0]["text"] == "".join(values[1:5])
    assert documents[0]["source_end"]["source_file"] == "1"


def test_long_document_window_and_native_partition_are_reproducible(settings):
    row = {"id": "long", "text": " ".join(map(str, range(1000)))}
    result = registry.token_window(row, Tokenizer(), settings, "books", "train")
    assert len(result[0]) == 128 and result[1] > 0
    assert result == registry.token_window(row, Tokenizer(), settings, "books", "train")
    assert readers.partition("example", 16) == readers.partition("example", 16)


def test_ag_news_native_test_and_derived_validation_stay_separate(monkeypatch, settings):
    values = [{"text": str(i), "label": 2, "source_row": i, "source_file": "native"} for i in range(300)]
    monkeypatch.setattr(readers, "parquet_rows", lambda *args: iter(values))
    c = {**settings, "sources": {"news": {"adapter": "ag_news"}}}
    train = list(readers.documents("news", "train", c, {}, Path('.')))
    validation = list(readers.documents("news", "validation", c, {}, Path('.')))
    test = list(readers.documents("news", "test", c, {}, Path('.')))
    assert train and validation and len(test) == len(values)
    assert {r["text"] for r in train}.isdisjoint(r["text"] for r in validation)
    assert all(r["native_split"] == "train" for r in train + validation)
    assert all(r["native_split"] == "test" for r in test)
    assert all("label" not in r and r["topic_label"] == 2 for r in train)
