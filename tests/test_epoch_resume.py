"""A real tiny OPT checks five-epoch selection and optimizer/scheduler recovery offline."""
import numpy as np
import pytest
import torch
from transformers import OPTConfig, OPTForCausalLM

from model_collapse import engine
from model_collapse.common import read_json


def test_five_epoch_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "device", lambda: "cpu")
    torch.set_num_threads(2)
    def loader(c, checkpoint=None):
        engine.seed_all(c["seed"])
        if checkpoint:
            return OPTForCausalLM.from_pretrained(checkpoint)
        return OPTForCausalLM(OPTConfig(vocab_size=32, hidden_size=16, word_embed_proj_dim=16,
                             ffn_dim=32, num_hidden_layers=1, num_attention_heads=2,
                             max_position_embeddings=128, dropout=.1))
    monkeypatch.setattr(engine, "load_model", loader)
    c = {"seed": 42, "epochs": 5, "learning_rate": 2e-5, "min_learning_rate": 1e-6, "weight_decay": .01,
         "microbatch": 2, "effective_batch": 4, "eval_batch": 2, "precision": "fp32"}
    # Seven blocks exercise both a partial accumulation window and a partial microbatch.
    blocks = np.random.default_rng(1).integers(4, 32, size=(7, 64), dtype=np.int32)
    base_hash = engine.weights_hash(loader(c))
    control = engine.train(tmp_path / "control", 0, blocks, blocks[:2], c, base_hash)
    normal_eval = engine.evaluate
    calls = []
    def fail_second_epoch(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("injected epoch failure")
        return normal_eval(*args, **kwargs)
    monkeypatch.setattr(engine, "evaluate", fail_second_epoch)
    with pytest.raises(RuntimeError, match="injected epoch"):
        engine.train(tmp_path / "interrupted", 0, blocks, blocks[:2], c, base_hash)
    monkeypatch.setattr(engine, "evaluate", normal_eval)
    resumed = engine.train(tmp_path / "interrupted", 0, blocks, blocks[:2], c, base_hash)
    assert resumed["selected_weights_sha256"] == control["selected_weights_sha256"]
    history = read_json(tmp_path / "interrupted/seed_42/gen_00/training_history.json")
    assert len(history) == 5
    assert resumed["selected_epoch"] == min(history, key=lambda row: row["validation_loss"])["epoch"]
    assert history[0]["learning_rate"] == 2e-5
    assert history[-1]["learning_rate"] < history[0]["learning_rate"]
