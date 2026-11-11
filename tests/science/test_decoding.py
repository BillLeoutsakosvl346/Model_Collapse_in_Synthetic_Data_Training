"""Decoder-only interventions retain legacy kwargs, masks and RNG independence."""
import copy
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from model_collapse import engine
from model_collapse.common import read_json
from model_collapse.science.decoding import decoder_config, decoding_settings, sampling_seed
from model_collapse.workflow import completion
from model_collapse.workflow.profiles import protocol_fingerprint, require_acceptance
from model_collapse.workflow.selection import training_contract


def profiles():
    base = Path("comparisons/fineweb_completion64_decoding")
    return [yaml.safe_load((base / (kind + ".yaml")).read_text()) for kind in ("full", "smoke")]


def test_historical_kwargs_unchanged_and_pure_sampling_explicit():
    old = {"num_beams": 5, "do_sample": False, "min_new_tokens": 64, "max_new_tokens": 64,
           "repetition_penalty": 1., "no_repeat_ngram_size": 0, "forced_eos_token_id": None, "use_cache": True}
    assert engine.decoding_settings({}) == old
    assert decoding_settings(decoder_config({}, "beam")) == old
    assert decoding_settings(decoder_config({}, "greedy")) == {**old, "num_beams": 1}
    assert decoding_settings(decoder_config({}, "sampling")) == {
        **old, "num_beams": 1, "do_sample": True, "temperature": 1., "top_k": 0, "top_p": 1.}


@pytest.mark.parametrize("changes", [
    {"num_beams": 5}, {"do_sample": False}, {"temperature": .9}, {"top_k": 50},
    {"top_p": .95}, {"temperature": float("nan")}, {"generation_decoder": "typo"},
])
def test_unsupported_decoder_settings_rejected(changes):
    with pytest.raises(ValueError):
        decoding_settings({**decoder_config({}, "sampling"), **changes})


def test_named_profiles_preserve_bank_and_separate_certificates():
    from model_collapse.completion_data import bank_contract
    full, smoke = profiles()
    historical = read_json(Path(__file__).resolve().parents[2] /
                           "reports/20261110T234038Z-full-completion64-decoding/records/20261106T193918Z-full-completion64/config.json")
    for c in (full, smoke):
        completion.validate_config(c)
        assert completion.allocation_id(c) == completion.DECODING_ALLOCATION
        assert completion.source_collection(c, 10) == "cycle_10"
        assert bank_contract(c) == historical["prepared_bank"]["contract"]
        for mode in ("greedy", "sampling"):
            arm = completion.arm_config(c, mode)
            completion.validate_config(arm)
            assert arm["data_kind"] == "synthetic" and arm["num_beams"] == 1
            assert arm["generation_decoder"] == mode
    assert protocol_fingerprint(full) == protocol_fingerprint(smoke)
    assert protocol_fingerprint(full) != protocol_fingerprint(historical)
    assert training_contract(completion.arm_config(full, "greedy")) != training_contract(completion.arm_config(full, "sampling"))
    for key in ("prepared_bank", "model_revision", "dataset_revision", "learning_rate", "min_learning_rate",
                "microbatch", "effective_batch", "precision", "generation_batch", "required_image"):
        assert full[key] == historical[key]
    assert full["reference_execution_difference"] == {"run_id": "20261106T193918Z-full-completion64", "field": "required_gpu",
        "old": historical["required_gpu"], "new": full["required_gpu"]}
    cert = {"workflow": "completion_comparison", "source_code_sha256": "runtime",
            "protocol_fingerprint": protocol_fingerprint(historical), "checks": {}}
    with pytest.raises(RuntimeError, match="protocol"):
        require_acceptance(full, cert, "runtime")


@pytest.mark.parametrize("changes", [
    {"completion_arms": [{"id": "x", "synthetic_fraction": 1.}]},
    {"endpoint_arms": ["synthetic"]}, {"decoding_arms": [{"id": "greedy", "decoder": "sampling"}]},
    {"training_chunk_policy": "repeat_fixed", "fixed_chunk": 1}, {"generation_shard": 33},
    {"generation_decoder": "greedy", "num_beams": 1},
])
def test_parent_protocol_does_not_accept_ambiguous_arm_definitions(changes):
    with pytest.raises(ValueError):
        completion.validate_config({**profiles()[1], **changes})


class RandomModel:
    def eval(self):
        return self

    def generate(self, input_ids, attention_mask, **settings):
        assert input_ids.shape[1] == 65 and attention_mask.all()
        assert settings["top_k"] == 0 and settings["do_sample"]
        values = torch.multinomial(torch.ones(len(input_ids), 31), 64, replacement=True) + 3
        return torch.cat([input_ids, values], dim=1)


def test_sampling_resume_matches_uninterrupted_and_preserves_training_rng(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "device", lambda: "cpu")
    c = {**decoder_config({}, "sampling"), "workflow": "completion_arm", "seed": 42,
         "generation_batch": 32, "generation_shard": 32, "precision": "fp32"}
    prompt = np.full((80, 65), 3, dtype=np.int32)  # exercises a partial last batch
    prompt[:, 0] = 2
    engine.seed_all(71)
    before = engine.capture_rng()
    uninterrupted, _ = engine.generate(tmp_path / "a", 1, RandomModel(), prompt, c, "checkpoint")
    assert torch.equal(before["torch"], torch.get_rng_state())
    expected_random, expected_numpy = random.random(), np.random.random()
    engine.restore_rng(before)
    with pytest.raises(engine.ValidationInterruption):
        engine.generate(tmp_path / "b", 1, RandomModel(), prompt, c, "checkpoint", inject=True)
    torch.manual_seed(400)  # unrelated operations between process exits must not matter
    resumed, meta = engine.generate(tmp_path / "b", 1, RandomModel(), prompt, c, "checkpoint", inject=True)
    assert np.array_equal(uninterrupted, resumed) and meta["reused_continuations"] == 32
    assert random.random() == expected_random and np.random.random() == expected_numpy
    repartitioned, _ = engine.generate(tmp_path / "c", 1, RandomModel(), prompt, {**c, "generation_shard": 64}, "checkpoint")
    assert np.array_equal(uninterrupted, repartitioned), "Shard sizes do not change aligned sampling batches"
    with pytest.raises(RuntimeError, match="provenance"):
        engine.generate(tmp_path / "b", 1, RandomModel(), prompt, {**c, "seed": 43}, "checkpoint")
    with pytest.raises(RuntimeError, match="provenance"):
        engine.generate(tmp_path / "b", 1, RandomModel(), prompt, {**c, "generation_shard": 64}, "checkpoint")


def test_sampling_seed_separates_cycle_batch_purpose_and_generator():
    c = {"seed": 42, "generation_batch": 32}
    source = {"parent_weights_sha256": "a", "prompt_tokens_sha256": "b"}
    values = [sampling_seed(c, 0, 0, source), sampling_seed(c, 1, 0, source), sampling_seed(c, 0, 32, source),
              sampling_seed({**c, "generation_purpose": "probe"}, 0, 0, source),
              sampling_seed(c, 0, 0, {**source, "parent_weights_sha256": "c"})]
    assert len(set(values)) == 5


def test_sampling_rng_restored_even_when_generation_raises(tmp_path):
    class Broken(RandomModel):
        def generate(self, *args, **kwargs):
            torch.rand(10)
            raise RuntimeError("simulated exit")
    c = {**decoder_config({}, "sampling"), "workflow": "completion_arm", "seed": 42,
         "generation_batch": 2, "generation_shard": 2, "precision": "fp32"}
    before = torch.get_rng_state()
    with pytest.raises(RuntimeError, match="simulated"):
        engine.generate(tmp_path, 0, Broken(), np.ones((2, 65), dtype=np.int32), c, "checkpoint")
    assert torch.equal(before, torch.get_rng_state())


def test_actual_opt_greedy_first_token_is_conditional_argmax():
    from transformers import OPTConfig, OPTForCausalLM
    torch.set_num_threads(1)
    engine.seed_all(42)
    m = OPTForCausalLM(OPTConfig(vocab_size=40, hidden_size=16, ffn_dim=32, num_hidden_layers=1,
        num_attention_heads=2, max_position_embeddings=256, word_embed_proj_dim=16, dropout=0.,
        attention_dropout=0., pad_token_id=1, bos_token_id=2, eos_token_id=2)).eval()
    prompt = torch.full((2, 65), 3)
    prompt[:, 0] = 2
    with torch.inference_mode():
        expected = m(input_ids=prompt, attention_mask=torch.ones_like(prompt)).logits[:, -1].argmax(-1)
        output = m.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                            **decoding_settings(decoder_config({}, "greedy")))
    assert output.shape == (2, 129) and torch.equal(output[:, 65], expected)


def test_decoder_forecast_uses_each_benchmark_and_common_probe():
    c = profiles()[0]
    b = {"training_target_tokens_per_second": 10000, "generation_new_tokens_per_second": 1000,
         "primary_target_tokens_per_second": 20000, "secondary_target_tokens_per_second": 20000,
         "full_primary_targets": {"validation": 468608, "test": 457664}, "full_secondary_targets": 100000,
         "decoder_rates": {"greedy": {"new_tokens_per_second": 4000}, "sampling": {"new_tokens_per_second": 3000}}}
    f = completion.forecast(b, c, .58)
    assert f["unique_training_stages"] == 20 and f["generated_training_continuations"] == 200000
    slower = copy.deepcopy(b)
    slower["decoder_rates"]["sampling"]["new_tokens_per_second"] = 1000
    assert completion.forecast(slower, c, .58)["estimated_hours"] > f["estimated_hours"]
