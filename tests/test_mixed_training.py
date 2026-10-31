"""Two real tiny-OPT updates with mixtures and exact epoch-boundary recovery."""
import numpy as np
import pytest
import torch
from transformers import OPTConfig, OPTForCausalLM

from model_collapse import engine
from model_collapse.common import atomic_json, checkpoint_manifest, read_json, sha256
from model_collapse.data import save_array
from model_collapse.workflow.lineage import token_hash, training_lineage


def test_two_mixed_cycles_keep_terminal_weights_and_resume_exactly(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "device", lambda: "cpu")
    torch.set_num_threads(2)
    engine.seed_all(5)
    parent_model = OPTForCausalLM(OPTConfig(vocab_size=32, hidden_size=16, word_embed_proj_dim=16,
        ffn_dim=32, num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=128, dropout=.1))
    parent = tmp_path / "seed_42/gen_00"
    parent_model.save_pretrained(parent / "checkpoint")
    parent_hash = engine.weights_hash(parent_model)
    atomic_json(parent / "training.json", {"selected_weights_sha256": parent_hash,
                                           "checkpoint_files": checkpoint_manifest(parent / "checkpoint")})
    monkeypatch.setattr(engine, "load_model", lambda c, checkpoint=None: OPTForCausalLM.from_pretrained(checkpoint))
    c = {"seed": 42, "epochs": 5, "checkpoint_selection": "final_epoch", "synthetic_fraction": .5,
         "learning_rate": 2e-5, "min_learning_rate": 1e-6, "weight_decay": .01,
         "microbatch": 2, "effective_batch": 4, "eval_batch": 2, "precision": "fp32",
         "initialization": "continued", "generation_batch": 2, "generation_shard": 4}
    real = np.random.default_rng(5).integers(4, 32, (8, 64), dtype=np.int32)
    val = np.random.default_rng(6).integers(4, 32, (4, 64), dtype=np.int32)
    save_array(tmp_path / "data/train.npy", real)
    save_array(tmp_path / "data/validation.npy", val)
    atomic_json(tmp_path / "data/manifest.json", {"files": {f"{name}.npy": sha256(tmp_path / f"data/{name}.npy")
                                                            for name in ("train", "validation")}})
    baseline_val = val.copy()
    first_snapshot = None
    for cycle in (1, 2):
        generated, generation = engine.generate(tmp_path, cycle - 1, parent_model, real, c, parent_hash)
        blocks, expected, checkpoint, lineage = training_lineage(tmp_path, cycle, c, None)
        assert expected == parent_hash == generation["provenance"]["parent_weights_sha256"]
        assert token_hash(blocks) != token_hash(generated)
        assert lineage["mixture"]["real_blocks"] == lineage["mixture"]["synthetic_blocks"] == 4
        mixed_files = {p: p.read_bytes() for p in (tmp_path / f"mixed_for_gen_{cycle:02d}").iterdir()}
        if cycle == 2:
            control = engine.train(tmp_path / "control", cycle, blocks, val, c, expected,
                                   initial_checkpoint=checkpoint, lineage=lineage)
            normal = engine.evaluate
            calls = []
            def interrupted(*args, **kwargs):
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError("interrupted epoch")
                return normal(*args, **kwargs)
            monkeypatch.setattr(engine, "evaluate", interrupted)
            with pytest.raises(RuntimeError, match="interrupted epoch"):
                engine.train(tmp_path, cycle, blocks, val, c, expected, initial_checkpoint=checkpoint, lineage=lineage)
            monkeypatch.setattr(engine, "evaluate", normal)
        trained = engine.train(tmp_path, cycle, blocks, val, c, expected, initial_checkpoint=checkpoint, lineage=lineage)
        assert trained["selected_epoch"] == trained["executed_epochs"] == 5
        assert trained["terminal_weights_sha256"] == trained["selected_weights_sha256"] != expected
        assert trained["input_tokens_sha256"] == token_hash(blocks)
        if cycle == 2:
            assert trained["selected_weights_sha256"] == control["selected_weights_sha256"]
        assert all(p.read_bytes() == before for p, before in mixed_files.items())
        assert np.array_equal(val, baseline_val)
        if first_snapshot:
            assert all(p.read_bytes() == before for p, before in first_snapshot.items())
        else:
            first_snapshot = {p: p.read_bytes() for p in (tmp_path / "seed_42/gen_01").rglob("*") if p.is_file()}
        parent = tmp_path / f"seed_42/gen_{cycle:02d}"
        parent_model = engine.load_model(c, parent / "checkpoint")
        parent_hash = engine.weights_hash(parent_model)
        assert engine.train(tmp_path, cycle, blocks, val, c, expected,
                            initial_checkpoint=checkpoint, lineage=lineage) == trained
        with pytest.raises(RuntimeError, match="settings changed"):
            engine.train(tmp_path, cycle, blocks, val, {**c, "synthetic_fraction": .2}, expected,
                         initial_checkpoint=checkpoint, lineage=lineage)
    assert read_json(tmp_path / "seed_42/gen_02/training.json")["initial_weights_sha256"] == read_json(
        tmp_path / "seed_42/gen_01/training.json")["selected_weights_sha256"]
