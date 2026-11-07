"""Conditional loss, target-normalized gradients and independent-arm matching."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import OPTConfig, OPTForCausalLM

from model_collapse import engine
from model_collapse.common import read_json
from model_collapse.completion_data import assemble


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    engine.seed_all(42)
    model = OPTForCausalLM(OPTConfig(vocab_size=40, hidden_size=16, ffn_dim=32, num_hidden_layers=1,
               num_attention_heads=2, max_position_embeddings=256, word_embed_proj_dim=16,
               dropout=0., attention_dropout=0., activation_dropout=0., pad_token_id=1, bos_token_id=2, eos_token_id=2))
    base = tmp_path / "base"
    model.save_pretrained(base, safe_serialization=True)
    def load(c, checkpoint=None):
        engine.seed_all(c["seed"])
        return OPTForCausalLM.from_pretrained(checkpoint or base).cpu()
    monkeypatch.setattr(engine, "load_model", load)
    monkeypatch.setattr(engine, "device", lambda: "cpu")
    c = {"workflow": "completion_arm", "data_kind": "real", "initialization": "continued", "epochs": 1,
         "checkpoint_selection": "final_epoch", "seed": 42, "learning_rate": 5e-6, "min_learning_rate": 1e-6,
         "weight_decay": .01, "microbatch": 1, "effective_batch": 2, "eval_batch": 2, "precision": "fp32",
         "prepared_bank": {"manifest_sha256": "0" * 64}}
    rng = np.random.default_rng(42)
    prompt, target = (rng.integers(3, 40, size=(3, 64), dtype=np.int32) for _ in range(2))
    tokens, attention, scoring = assemble(prompt, target, 2)
    return model, base, c, tokens, attention, scoring


def train(root, base, c, tokens, attention, scoring, *, cycle=1, parent_state=None, steps=4):
    model = engine.load_model(c, base)
    expected = engine.weights_hash(model)
    return engine.train(root, cycle, tokens, tokens, c, expected, initial_checkpoint=base,
                        masks=attention, val_masks=attention, loss_masks=scoring, val_loss_masks=scoring,
                        optimizer_checkpoint=parent_state, total_steps=steps)


def test_exact_teacher_forced_loss_and_first_target_alignment(tiny):
    model, base, c, tokens, attention, scoring = tiny
    ids = torch.tensor(tokens, dtype=torch.long)
    logits = model(input_ids=ids, attention_mask=torch.tensor(attention)).logits
    losses = engine.shifted_nll(logits, ids, torch.tensor(scoring))
    direct = -torch.log_softmax(logits[:, 64:128].float(), -1).gather(-1, ids[:, 65:, None]).squeeze(-1)
    assert torch.allclose(losses[:, 64:], direct, atol=1e-6)
    assert torch.count_nonzero(losses[:, :64]) == 0
    result = engine.evaluate(model, tokens, c, masks=attention, loss_masks=scoring)
    assert result["scored_tokens"] == 192
    assert result["loss"] == pytest.approx(float(direct.mean().detach()), abs=1e-6)
    gradient = torch.autograd.grad(losses.sum(), logits)[0]
    assert torch.count_nonzero(gradient[:, :64]) == 0
    assert torch.count_nonzero(gradient[:, 64:128]) > 0


def test_token_weighted_accumulation_matches_larger_microbatch(tmp_path, tiny):
    model, base, c, tokens, attention, scoring = tiny
    scoring[1, 97:] = 0
    scoring[2, 72:] = 0
    a, b = tmp_path / "a", tmp_path / "b"
    train(a, base, c, tokens, attention, scoring)
    train(b, base, {**c, "microbatch": 2}, tokens, attention, scoring)
    left, right = (engine.load_model(c, p / "seed_42/gen_01/checkpoint") for p in (a, b))
    for name, value in left.state_dict().items():
        assert torch.allclose(value, right.state_dict()[name], atol=1e-7, rtol=1e-6)


def test_same_targets_produce_identical_arms_and_carried_states(tmp_path, tiny):
    model, base, c, tokens, attention, scoring = tiny
    results = []
    for name in ("real", "synthetic"):
        root = tmp_path / name
        arm = {**c, "data_kind": name}
        first = train(root, base, arm, tokens, attention, scoring)
        state = root / "seed_42/gen_01/optimizer_state/cycle.pt"
        second = train(root, root / "seed_42/gen_01/checkpoint", arm, tokens, attention, scoring,
                       cycle=2, parent_state=state)
        assert first["global_step"] == 2 and second["global_step"] == 4
        optimizer = torch.load(root / "seed_42/gen_02/optimizer_state/cycle.pt", weights_only=False)
        assert optimizer["scheduler"]["last_epoch"] == 4
        assert all(int(s["step"]) == 4 for s in optimizer["optimizer"]["state"].values())
        results.append(second["selected_weights_sha256"])
    assert results[0] == results[1]


def test_changed_loss_mask_rejects_completed_stage(tmp_path, tiny):
    model, base, c, tokens, attention, scoring = tiny
    train(tmp_path, base, c, tokens, attention, scoring)
    changed = scoring.copy()
    changed[:, 65] = 0
    with pytest.raises(RuntimeError, match="provenance"):
        train(tmp_path, base, c, tokens, attention, changed)
    assert read_json(tmp_path / "seed_42/gen_01/training.json")["executed_epochs"] == 1


def test_identical_targets_through_both_mixture_builders_match_weights(tmp_path, tiny):
    from model_collapse.completion_mixing import freeze_selection, freeze_targets
    model, base, c, tokens, attention, scoring = tiny
    prompt, real = tokens[:, 1:65], tokens[:, 65:]
    hashes = []
    for name, fraction in (("mixed50", .5), ("mixed20", .2)):
        root = tmp_path / name
        selected, meta = freeze_selection(root, prompt, real, fraction, 42, 1, engine.weights_hash(model))
        targets, _ = freeze_targets(root, real, real[selected["synthetic_indices"]], selected, meta, "same-targets")
        assembled, masks, loss_masks = assemble(prompt, targets, 2)
        assert assembled.tobytes() == tokens.tobytes()
        result = train(root, base, {**c, "synthetic_fraction": fraction}, assembled, masks, loss_masks)
        hashes.append(result["selected_weights_sha256"])
    assert hashes[0] == hashes[1]
