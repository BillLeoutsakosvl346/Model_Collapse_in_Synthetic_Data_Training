"""Exercise real tiny OPT updates, beam generation, parent enforcement and recovery."""
import hashlib

import numpy as np
import pytest
import torch
from transformers import OPTConfig, OPTForCausalLM

from model_collapse import engine, remote
from model_collapse.common import atomic_json, config, PROJECT, read_json


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "device", lambda: "cpu")
    torch.set_num_threads(2)
    def loader(c, checkpoint=None):
        if checkpoint:
            return OPTForCausalLM.from_pretrained(checkpoint)
        raise AssertionError("Continued training attempted to reload original pretrained weights")
    monkeypatch.setattr(engine, "load_model", loader)
    engine.seed_all(5)
    model = OPTForCausalLM(OPTConfig(vocab_size=32, hidden_size=16, word_embed_proj_dim=16,
                                    ffn_dim=32, num_hidden_layers=1, num_attention_heads=2,
                                    max_position_embeddings=128, dropout=.1))
    parent = tmp_path / "parent"
    model.save_pretrained(parent)
    c = {"seed": 42, "epochs": 2, "learning_rate": 2e-5, "min_learning_rate": 1e-6,
         "weight_decay": .01, "microbatch": 2, "effective_batch": 4, "eval_batch": 2,
         "precision": "fp32", "initialization": "continued", "generation_batch": 2, "generation_shard": 2}
    blocks = np.random.default_rng(1).integers(4, 32, size=(4, 64), dtype=np.int32)
    return c, blocks, parent, model


def test_two_cycles_generate_then_train_the_same_model(tmp_path, monkeypatch, setup):
    c, prompts, parent, model = setup
    expected = engine.weights_hash(model)
    original = expected
    optimizer_starts = []
    normal_adam = torch.optim.AdamW
    def adam(*args, **kwargs):
        opt = normal_adam(*args, **kwargs)
        optimizer_starts.append(len(opt.state))
        return opt
    monkeypatch.setattr(torch.optim, "AdamW", adam)
    for cycle in (1, 2):
        values, manifest = engine.generate(tmp_path, cycle - 1, model, prompts, c, expected)
        trained = engine.train(tmp_path, cycle, values, prompts[:2], c, expected, initial_checkpoint=parent)
        assert trained["initial_weights_sha256"] == manifest["provenance"]["parent_weights_sha256"] == expected
        assert trained["input_tokens_sha256"] == hashlib.sha256(values.astype(np.int32).tobytes()).hexdigest()
        assert trained["input_tokens_sha256"] == manifest["continuation_tokens_sha256"]
        assert trained["selected_weights_sha256"] != expected
        parent = tmp_path / "seed_42" / f"gen_{cycle:02d}" / "checkpoint"
        model = engine.load_model(c, parent)
        assert engine.weights_hash(model) == trained["selected_weights_sha256"]
        expected = trained["selected_weights_sha256"]
    assert optimizer_starts == [0, 0]
    assert read_json(tmp_path / "seed_42/gen_02/training.json")["initial_weights_sha256"] != original


def test_continued_weights_epoch_resume_matches_control(tmp_path, monkeypatch, setup):
    c, blocks, parent, model = setup
    c["epochs"] = 5
    expected = engine.weights_hash(model)
    control = engine.train(tmp_path / "control", 1, blocks, blocks[:2], c, expected, initial_checkpoint=parent)
    normal = engine.evaluate
    calls = []
    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("interrupted epoch")
        return normal(*args, **kwargs)
    monkeypatch.setattr(engine, "evaluate", interrupted)
    with pytest.raises(RuntimeError, match="interrupted epoch"):
        engine.train(tmp_path / "resume", 1, blocks, blocks[:2], c, expected, initial_checkpoint=parent)
    monkeypatch.setattr(engine, "evaluate", normal)
    result = engine.train(tmp_path / "resume", 1, blocks, blocks[:2], c, expected, initial_checkpoint=parent)
    assert result["selected_weights_sha256"] == control["selected_weights_sha256"]


def test_missing_or_wrong_parent_is_rejected(tmp_path, setup):
    c, blocks, parent, model = setup
    with pytest.raises(RuntimeError, match="preceding model"):
        engine.train(tmp_path / "missing", 1, blocks, blocks[:2], c, engine.weights_hash(model))
    with pytest.raises(RuntimeError, match="expected initial weights"):
        engine.train(tmp_path / "wrong", 1, blocks, blocks[:2], c, "incorrect", initial_checkpoint=parent)


def test_continued_configs_pin_same_initial_checkpoint_and_protocol():
    smoke, full = (config(PROJECT / f"configs/{kind}-continued.yaml") for kind in ("smoke", "full"))
    assert smoke["initialization"] == full["initialization"] == "continued"
    assert smoke["start_checkpoint"] == full["start_checkpoint"]
    assert smoke["generations"] == 2 and full["generations"] == 10
    assert full["train_blocks"] is None and smoke["train_blocks"] == 256


def test_fresh_base_validation_cannot_authorize_continued_full_launch(tmp_path, monkeypatch):
    class API:
        def request(self, method, path, body=None):
            assert method == "GET", "An unvalidated experiment must never provision resources"
            return {"pods": []}
        def candidates(self):
            return [{"gpu": "test", "data_center": "test", "hourly_usd": .2}]
    monkeypatch.setattr(remote, "Runpod", API)
    monkeypatch.setattr(remote, "STATE", tmp_path)
    monkeypatch.setattr(remote, "ledger", lambda: {"launches": []})
    monkeypatch.setattr(remote, "preflight", lambda **kw: {"account": {"clientBalance": 30}})
    monkeypatch.setattr(remote, "source_archive", lambda: ("unused", "code"))
    atomic_json(tmp_path / "validation.json", {"source_code_sha256": "code"})
    with pytest.raises(RuntimeError, match="Validate the complete smoke test"):
        remote.launch(mode="continued")
