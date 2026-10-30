"""Five-epoch selection and resume are tested with actual tiny OPT weight updates."""
import copy

import numpy as np
import pytest
import torch
from transformers import OPTConfig, OPTForCausalLM

from model_collapse import engine
from model_collapse.common import read_json


@pytest.fixture
def setup(monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(engine, "device", lambda: "cpu")
    def loader(c, checkpoint=None):
        if checkpoint:
            return OPTForCausalLM.from_pretrained(checkpoint)
        engine.seed_all(c["seed"])
        return OPTForCausalLM(OPTConfig(vocab_size=32, hidden_size=16, word_embed_proj_dim=16,
                             ffn_dim=32, num_hidden_layers=1, num_attention_heads=2,
                             max_position_embeddings=128, dropout=.1))
    monkeypatch.setattr(engine, "load_model", loader)
    c = {"seed": 42, "epochs": 5, "learning_rate": 2e-5, "min_learning_rate": 1e-6, "weight_decay": .01,
         "microbatch": 2, "effective_batch": 4, "eval_batch": 2, "precision": "fp32", "checkpoint_selection": "final_epoch"}
    blocks = np.random.default_rng(2).integers(4, 32, size=(7, 64), dtype=np.int32)
    return c, blocks, engine.weights_hash(loader(c))


def test_final_epoch_selected_even_when_validation_worsens(setup, tmp_path, monkeypatch):
    c, blocks, initial = setup
    losses = iter([1., 2., 3., 4., 5.])
    monkeypatch.setattr(engine, "evaluate", lambda *a: {"loss": next(losses)})
    t = engine.train(tmp_path / "final", 0, blocks, blocks[:2], c, initial)
    assert t["selected_epoch"] == t["executed_epochs"] == 5
    assert t["best_validation_loss"] == 1. and t["best_validation_epoch"] == 1
    assert t["selected_validation_loss"] == 5.
    assert t["terminal_weights_sha256"] == t["selected_weights_sha256"] != initial
    legacy = {k: v for k, v in c.items() if k != "checkpoint_selection"}
    losses = iter([1., 2., 3., 4., 5.])
    old = engine.train(tmp_path / "legacy", 0, blocks, blocks[:2], legacy, initial)
    assert old["selected_epoch"] == 1 and old["executed_epochs"] == 5
    assert old["selected_weights_sha256"] != t["selected_weights_sha256"]


@pytest.mark.parametrize("fail_epoch", [2, 5])
def test_final_epoch_resume_matches_uninterrupted(setup, tmp_path, monkeypatch, fail_epoch):
    c, blocks, initial = setup
    control = engine.train(tmp_path / "control", 0, blocks, blocks[:2], c, initial)
    normal, calls = engine.evaluate, []
    def failure(*args, **kwargs):
        calls.append(1)
        if len(calls) == fail_epoch:
            raise RuntimeError("injected epoch failure")
        return normal(*args, **kwargs)
    monkeypatch.setattr(engine, "evaluate", failure)
    with pytest.raises(RuntimeError, match="injected"):
        engine.train(tmp_path / "resume", 0, blocks, blocks[:2], c, initial)
    monkeypatch.setattr(engine, "evaluate", normal)
    resumed = engine.train(tmp_path / "resume", 0, blocks, blocks[:2], c, initial)
    assert resumed["selected_weights_sha256"] == control["selected_weights_sha256"]
    assert len(read_json(tmp_path / "resume/seed_42/gen_00/training_history.json")) == 5


def test_terminal_persistence_recovery_and_changed_contract(setup, tmp_path):
    c, blocks, initial = setup
    t = engine.train(tmp_path, 0, blocks, blocks[:2], c, initial)
    gen = tmp_path / "seed_42/gen_00"
    for changed in ({**c, "epochs": 4}, {**c, "checkpoint_selection": "lowest_fixed_real_validation_loss"},
                    {**c, "learning_rate": 1e-4}):
        with pytest.raises(RuntimeError, match="changed"):
            engine.train(tmp_path, 0, blocks, blocks[:2], changed, initial)
    (gen / "training.json").unlink()
    (gen / "checkpoint/model.safetensors").unlink()
    (gen / "training_history.json").unlink()
    recovered = engine.train(tmp_path, 0, blocks, blocks[:2], c, initial)
    assert recovered["selected_weights_sha256"] == t["selected_weights_sha256"]
    (gen / "training.json").unlink()
    with pytest.raises(RuntimeError, match="changed"):
        engine.train(tmp_path, 0, blocks, blocks[:2], {**c, "epochs": 4}, initial)


def test_corrupt_terminal_checkpoint_rejected(setup, tmp_path):
    c, blocks, initial = setup
    engine.train(tmp_path, 0, blocks, blocks[:2], c, initial)
    (tmp_path / "seed_42/gen_00/checkpoint/model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="corrupt"):
        engine.train(tmp_path, 0, blocks, blocks[:2], c, initial)
