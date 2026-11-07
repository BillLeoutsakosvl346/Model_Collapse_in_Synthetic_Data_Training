"""Regression coverage for FineWeb sampling, masked losses and continuous state."""
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import OPTConfig, OPTForCausalLM

from model_collapse import engine
from model_collapse.common import config, read_json, sha256
from model_collapse.fresh_data import encode_sequences, load_collection, prepare_fresh
from model_collapse.workflow.fresh_real import training_input_hash, validate_config
from model_collapse.workflow.profiles import protocol_fingerprint, require_acceptance


class Tokenizer:
    bos_token_id = 2
    eos_token_id = 2
    pad_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [int(value) for value in text.split()]

    def save_pretrained(self, folder):
        Path(folder).mkdir(parents=True, exist_ok=True)
        (Path(folder) / "tokenizer_config.json").write_text('{}')


@pytest.fixture
def settings():
    c = config("experiments/opt125m_fineweb_edu/fresh_real_one_epoch/smoke.yaml")
    c.update(max_sequence_length=16, source_tokens_per_cycle=180, validation_documents=2,
             test_documents=2, microbatch=2, effective_batch=4, eval_batch=3, precision="fp32")
    return c


def documents():
    rng = np.random.default_rng(10)
    return [{"id": str(i), "url": f"https://example.org/{i}",
             "text": " ".join(map(str, rng.integers(4, 32, size=70 + i % 13)))} for i in range(100)]


@pytest.fixture
def tiny_engine(monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(engine, "device", lambda: "cpu")

    def loader(c, checkpoint=None):
        engine.seed_all(c["seed"])
        if checkpoint:
            return OPTForCausalLM.from_pretrained(checkpoint)
        return OPTForCausalLM(OPTConfig(vocab_size=32, hidden_size=16, word_embed_proj_dim=16,
                             ffn_dim=32, num_hidden_layers=1, num_attention_heads=2,
                             max_position_embeddings=64, dropout=0.0, attention_dropout=0.0))
    monkeypatch.setattr(engine, "load_model", loader)
    return loader


def test_named_profiles_bind_the_same_protocol():
    folder = Path("experiments/opt125m_fineweb_edu/fresh_real_one_epoch")
    full, smoke = (config(folder / f"{kind}.yaml") for kind in ("full", "smoke"))
    assert protocol_fingerprint(full) == protocol_fingerprint(smoke)
    changed = {**full, "max_sequence_length": 64}
    assert protocol_fingerprint(changed) != protocol_fingerprint(full)
    with pytest.raises(RuntimeError):
        require_acceptance(full, {"source_code_sha256": "a", "protocol_fingerprint": protocol_fingerprint(full), "checks": {}}, "a")


def test_data_exact_quotas_reconstruction_and_immutable_reuse(tmp_path, settings):
    from model_collapse.workflow.fresh_validation import audit_data
    tok = Tokenizer()
    first = prepare_fresh(tmp_path / "a", settings, tok, documents())
    other = prepare_fresh(tmp_path / "b", settings, tok, documents())
    assert first["collections"] == other["collections"]
    actual = audit_data(tmp_path / "a", settings, tok)
    assert actual["cycle_01"]["source_text_tokens"] == 180
    assert actual["cycle_03"]["source_text_tokens"] == 180
    assert prepare_fresh(tmp_path / "a", settings, tok, []) == first
    with pytest.raises(RuntimeError, match="configuration changed"):
        prepare_fresh(tmp_path / "a", {**settings, "seed": 43}, tok, documents())


def test_duplicate_and_shared_passage_rejections(tmp_path, settings):
    rows = documents()
    rows.insert(1, {**rows[0], "id": "copied", "url": "https://other.org", "text": rows[0]["text"] + " 9"})
    rows.insert(2, dict(rows[0]))
    c = {**settings, "validation_documents": 1, "test_documents": 1}
    result = prepare_fresh(tmp_path, c, Tokenizer(), rows)
    assert result["rejected_documents"]["shared_64_token_passage"] == 1
    assert result["rejected_documents"]["duplicate_identity"] == 1


def test_preparation_interruption_replays_identically_and_corruption_rejected(tmp_path, settings):
    def broken():
        for index, row in enumerate(documents()):
            if index == 7:
                raise OSError("stream interruption")
            yield row
    with pytest.raises(OSError, match="stream interruption"):
        prepare_fresh(tmp_path, settings, Tokenizer(), broken())
    assert not (tmp_path / "data/manifest.json").exists()
    prepare_fresh(tmp_path, settings, Tokenizer(), documents())
    path = tmp_path / "data/cycle_01/masks.npy"
    values = np.load(path)
    values[0, 0] = 0
    np.save(path, values)
    with pytest.raises(RuntimeError):
        load_collection(tmp_path, "cycle_01")


def test_short_and_trimmed_document_special_tokens():
    tok = Tokenizer()
    assert encode_sequences([7], tok, 8, genuine_end=True) == [[2, 7, 2]]
    assert encode_sequences([7], tok, 8, genuine_end=False) == [[2, 7]]
    chunks = encode_sequences(list(range(4, 17)), tok, 8, genuine_end=True)
    assert all(len(row) <= 8 for row in chunks)
    assert chunks[-1][-1] == 2


def test_masked_corpus_loss_matches_direct_calculation(settings, tiny_engine):
    c = settings
    model = tiny_engine(c)
    tokens = np.array([[2, 7, 8, 2, 1, 1], [2, 9, 2, 1, 1, 1]], dtype=np.int32)
    masks = np.array([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=np.int32)
    actual = engine.evaluate(model, tokens, c, masks=masks)
    with torch.no_grad():
        ids, attention = engine.tensor(tokens), engine.tensor(masks)
        logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
        direct = -logits[:, :-1].log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    assert actual["scored_tokens"] == 5
    assert actual["loss"] == pytest.approx(float((direct * attention[:, 1:]).sum()) / 5, abs=1e-6)
    assert actual["perplexity"] == pytest.approx(math.exp(actual["loss"]))
    changed = tokens.copy()
    changed[masks == 0] = 31
    assert engine.evaluate(model, changed, c, masks=masks)["loss"] == pytest.approx(actual["loss"], abs=1e-7)


def train_cycle(root, cycle, c, loader, previous=None, *, inject=False):
    tokens, masks, _ = load_collection(root, f"cycle_{cycle:02d}")
    val, val_masks, _ = load_collection(root, "validation")
    source = root / "initial"
    if previous is None:
        model = loader(c)
        model.save_pretrained(source)
        initial = engine.weights_hash(model)
        del model
        parent_state = None
    else:
        source = root / f"seed_{c['seed']}/gen_{cycle - 1:02d}/checkpoint"
        parent_state = source.parent / "optimizer_state/cycle.pt"
        initial = previous["selected_weights_sha256"]
    total_steps = sum(math.ceil(len(load_collection(root, f"cycle_{i:02d}")[0]) / c["effective_batch"])
                      for i in range(1, c["generations"] + 1))
    return engine.train(root, cycle, tokens, val, c, initial, initial_checkpoint=source,
                        masks=masks, val_masks=val_masks, optimizer_checkpoint=parent_state,
                        total_steps=total_steps, inject=inject)


def test_continuous_optimizer_midcycle_resume_matches_control(tmp_path, settings, tiny_engine):
    c = settings
    for name in ("control", "interrupted"):
        prepare_fresh(tmp_path / name, c, Tokenizer(), documents())
    a = tmp_path / "control"
    first = train_cycle(a, 1, c, tiny_engine)
    second = train_cycle(a, 2, c, tiny_engine, first)
    third = train_cycle(a, 3, c, tiny_engine, second)
    b = tmp_path / "interrupted"
    other_first = train_cycle(b, 1, c, tiny_engine)
    snapshot = sha256(b / "seed_42/gen_01/checkpoint/model.safetensors")
    with pytest.raises(engine.ValidationInterruption):
        train_cycle(b, 2, c, tiny_engine, other_first, inject=True)
    other_second = train_cycle(b, 2, c, tiny_engine, other_first, inject=True)
    other_third = train_cycle(b, 3, c, tiny_engine, other_second)
    assert other_third["selected_weights_sha256"] == third["selected_weights_sha256"]
    assert sha256(b / "seed_42/gen_01/checkpoint/model.safetensors") == snapshot
    assert train_cycle(b, 3, c, tiny_engine, other_second) == other_third
    state = torch.load(b / "seed_42/gen_03/optimizer_state/cycle.pt", weights_only=False)
    assert state["global_step"] == state["total_steps"]
    assert all(int(value["step"]) == state["global_step"] for value in state["optimizer"]["state"].values())
    assert state["optimizer"]["param_groups"][0]["lr"] == pytest.approx(c["min_learning_rate"])
    with pytest.raises(RuntimeError):
        train_cycle(b, 3, {**c, "learning_rate": 1e-5}, tiny_engine, other_second)


def test_token_weighted_gradient_accumulation(tmp_path, settings, tiny_engine):
    c = {**settings, "generations": 1, "effective_batch": 4}
    records = []
    for batch in (1, 4):
        current = {**c, "microbatch": batch}
        root = tmp_path / str(batch)
        prepare_fresh(root, current, Tokenizer(), documents())
        result = train_cycle(root, 1, current, tiny_engine)
        records.append(tiny_engine(current, root / "seed_42/gen_01/checkpoint").state_dict())
        assert result["global_step"] == result["total_optimizer_steps"]
    for name in records[0]:
        torch.testing.assert_close(records[0][name], records[1][name], atol=2e-7, rtol=2e-6)


def test_terminal_checkpoint_interruption_recovers_without_extra_updates(tmp_path, settings, tiny_engine, monkeypatch):
    c = {**settings, "generations": 1}
    prepare_fresh(tmp_path, c, Tokenizer(), documents())
    original = engine.checkpoint_manifest
    monkeypatch.setattr(engine, "checkpoint_manifest", lambda path: (_ for _ in ()).throw(OSError("terminal interruption")))
    with pytest.raises(OSError, match="terminal interruption"):
        train_cycle(tmp_path, 1, c, tiny_engine)
    weights = sha256(tmp_path / "seed_42/gen_01/checkpoint/model.safetensors")
    monkeypatch.setattr(engine, "checkpoint_manifest", original)
    result = train_cycle(tmp_path, 1, c, tiny_engine)
    assert result["executed_epochs"] == 1
    assert result["global_step"] == result["total_optimizer_steps"]
    assert sha256(tmp_path / "seed_42/gen_01/checkpoint/model.safetensors") == weights


def test_production_worker_restarts_and_report_without_provider(tmp_path, settings, tiny_engine, monkeypatch):
    from model_collapse import fresh_data, hub, pipeline
    from model_collapse.common import atomic_json
    from model_collapse.workflow import fresh_real, fresh_validation
    from transformers import AutoTokenizer
    c = settings
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(fresh_data, "shuffled_documents", lambda *args: iter(documents()))
    monkeypatch.setattr(pipeline, "environment", lambda: {"gpu": "cpu"})
    backups = []
    monkeypatch.setattr(hub, "backup", lambda root: backups.append(root))
    monkeypatch.setattr(fresh_validation, "validate_remote", lambda root: None)
    atomic_json(tmp_path / "config.json", c)
    atomic_json(tmp_path / "resources.json", {"code_sha256": "a" * 64})
    for _ in range(2):
        with pytest.raises(engine.ValidationInterruption):
            fresh_real.run(tmp_path, validation=True)
    fresh_real.run(tmp_path, validation=True)
    assert read_json(tmp_path / "completion.json")["cycles"] == 3
    snapshot = sha256(tmp_path / "seed_42/gen_01/metrics.json")
    fresh_real.run(tmp_path, validation=False)
    assert sha256(tmp_path / "seed_42/gen_01/metrics.json") == snapshot
    assert backups
    summary = list((tmp_path / "generated").glob('*/summary.json'))
    assert summary and all(read_json(path)["stability_observed"] is False for path in summary)


@pytest.mark.parametrize("change", [{"epochs": 5}, {"max_sequence_length": 2049}, {"synthetic_fraction": .5},
                                    {"source_tokens_per_cycle": 0}, {"dataset_revision": "main"}])
def test_invalid_fresh_profiles_rejected(settings, change):
    with pytest.raises(ValueError):
        validate_config({**settings, **change})


def test_recovery_verifies_indexed_controls_without_replacing_current_registry(tmp_path):
    from model_collapse.common import atomic_json
    from model_collapse.workflow.fresh_validation import verify_recovery
    historic = tmp_path / "fetched_control/original/resources.json"
    atomic_json(historic, {"run_id": "example", "pod_id": "original"})
    atomic_json(tmp_path / "resources.json", {"run_id": "example", "pod_id": "current"})
    current = (tmp_path / "resources.json").read_bytes()
    expected = sha256(historic)
    atomic_json(tmp_path / "artifact_index.json", {"files": {"resources.json": {"sha256": expected}}})
    atomic_json(tmp_path / "recovery_snapshot.json", {"preserved_control_files": {
        "resources.json": {"recovered_path": "fetched_control/original/resources.json", "sha256": expected}}})
    verify_recovery(tmp_path)
    assert (tmp_path / "resources.json").read_bytes() == current
    historic.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="corrupt"):
        verify_recovery(tmp_path)


@pytest.mark.parametrize("already_spent,admitted", [(.2, True), (.51, False)])
def test_new_smoke_cap_is_shared_across_attempts_before_provisioning(tmp_path, monkeypatch, already_spent, admitted):
    from model_collapse import remote
    from model_collapse.common import PROJECT, config
    class ReachedAdmission(Exception):
        pass
    class API:
        def request(self, method, path, body=None):
            assert method == "GET" and path == "/pods"
            return {"pods": []}
    monkeypatch.setattr(remote, "STATE", tmp_path)
    monkeypatch.setattr(remote, "Runpod", API)
    monkeypatch.setattr(remote, "load_secrets", lambda: None)
    monkeypatch.setattr(remote, "ledger", lambda: {"total_budget_usd": 50, "launches": []})
    monkeypatch.setattr(remote, "accrued", lambda *args: 1)
    monkeypatch.setattr(remote, "campaign_accrued", lambda d, name: already_spent)
    path = PROJECT / "experiments/opt125m_fineweb_edu/fresh_real_one_epoch/smoke.yaml"
    gpu = config(path)["required_gpu"]
    monkeypatch.setattr(remote, "preflight", lambda **kwargs: {"account": {"clientBalance": 10},
                        "gpu_candidates": [{"gpu": gpu, "vram_gb": 24, "data_center": "DC", "hourly_usd": .3}]})
    monkeypatch.setattr(remote, "source_archive", lambda: (Path('unused'), "runtime"))
    monkeypatch.setattr(remote, "verify_start_checkpoint", lambda c: (_ for _ in ()).throw(ReachedAdmission()))
    expected = ReachedAdmission if admitted else RuntimeError
    with pytest.raises(expected):
        remote.launch(smoke=True, config_path=path)
