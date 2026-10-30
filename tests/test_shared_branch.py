"""Real tiny OPT fork tests; no network, credentials, large models or GPUs required."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import OPTConfig, OPTForCausalLM

from model_collapse import engine
from model_collapse.common import atomic_json, checkpoint_manifest, digest, read_json, sha256, verify_files
from model_collapse.data import save_array
from model_collapse.hub import artifact_files
from model_collapse.workflow import branch
from model_collapse.workflow.lineage import token_hash, training_lineage, verify_completed_stage


@pytest.fixture
def source(tmp_path, monkeypatch, request):
    torch.set_num_threads(2)
    monkeypatch.setattr(engine, "device", lambda: "cpu")
    monkeypatch.setattr(engine, "load_model", lambda c, checkpoint=None: OPTForCausalLM.from_pretrained(checkpoint))
    c = {"seed": 42, "epochs": 1, "learning_rate": 2e-5, "min_learning_rate": 1e-6, "weight_decay": .01,
         "microbatch": 2, "effective_batch": 4, "eval_batch": 2, "precision": "fp32", "initialization": "continued",
         "generation_batch": 2, "generation_shard": 2, "kind": "full", "generations": 3,
         "model": "tiny-opt", "model_revision": "a" * 40, "dataset": "tiny-real", "dataset_revision": "b" * 40,
         "dataset_config": "raw", "block_size": 64, "new_tokens": 64, "num_beams": 5,
         "repetition_penalty": 1.0, "probe_count": 4, "train_blocks": None, "validation_blocks": None, "test_blocks": None}
    c.update(getattr(request, "param", {}))
    root = tmp_path / "source"
    real = np.random.default_rng(4).integers(4, 32, size=(8, 64), dtype=np.int32)
    for split, data in (("train", real), ("validation", real[:4]), ("test", real[4:])):
        save_array(root / f"data/{split}.npy", data)
    atomic_json(root / "data/manifest.json", {"files": {f"{s}.npy": sha256(root / f"data/{s}.npy")
                                                        for s in ("train", "validation", "test")}})
    engine.seed_all(42)
    model = OPTForCausalLM(OPTConfig(vocab_size=32, hidden_size=16, word_embed_proj_dim=16,
                                    ffn_dim=32, num_hidden_layers=1, num_attention_heads=2,
                                    max_position_embeddings=128, dropout=.1))
    gen0 = root / "seed_42/gen_00"
    model.save_pretrained(gen0 / "checkpoint")
    initial = engine.weights_hash(model)
    t0 = {"generation": 0, "initial_weights_sha256": initial, "selected_weights_sha256": initial,
          "input_tokens_sha256": None, "training_blocks": 0, "selected_epoch": None,
          "seconds": 0, "checkpoint_files": checkpoint_manifest(gen0 / "checkpoint")}
    atomic_json(gen0 / "training.json", t0)
    d1, d1m = engine.generate(root, 0, model, real, c, initial)
    t1 = engine.train(root, 1, d1, real[:4], c, initial, initial_checkpoint=gen0 / "checkpoint")
    model1 = engine.load_model(c, root / "seed_42/gen_01/checkpoint")
    d2, d2m = engine.generate(root, 1, model1, real, c, t1["selected_weights_sha256"])
    for g, training, generation in ((0, t0, d1m), (1, t1, d2m)):
        gen = root / f"seed_42/gen_{g:02d}"
        atomic_json(gen / "metrics.json", {"generation": g, "training": training, "generation_timing": generation,
                                          "test": {"perplexity": 123.0}, "validation": {"perplexity": 123.0}})
        for name in ("samples.json", "test_sequences.json", "reference_sequences.json"):
            atomic_json(gen / name, [])
        stage_files = {p.relative_to(gen).as_posix(): sha256(p) for p in gen.rglob("*")
                       if p.is_file() and "resume" not in p.parts and p.name != "stage_completion.json"}
        atomic_json(gen / "stage_completion.json", {"files": stage_files,
                                                   "selected_weights_sha256": training["selected_weights_sha256"]})
    repos = {"model": "test/models", "dataset": "test/data"}
    c["start_checkpoint"] = {"repo": repos["model"], "revision": "c" * 40, "path": "source/seed_42/gen_00/checkpoint",
                              "weights_sha256": initial, "files": t0["checkpoint_files"]}
    atomic_json(root / "config.json", c)
    atomic_json(root / "manifest.json", {"config_hash": digest(c)})
    atomic_json(root / "completion.json", {"complete": True, "generations": 3})
    index = {"run_id": "source", "repos": repos, "files": artifact_files(root)}
    atomic_json(root / "artifact_index.json", index)
    monkeypatch.setattr(branch, "resolve_snapshot", lambda repos, revisions=None: revisions or {"model": "d" * 40, "dataset": "e" * 40})
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *args, **kwargs: str(root / "artifact_index.json"))
    def download(run_id, repos, rel, meta, revisions):
        p = root / rel
        verify_files(root, {rel: meta})
        return p
    monkeypatch.setattr(branch, "download_snapshot_file", download)
    def head(url):
        rel = url.split("/source/", 1)[1]
        meta = index["files"][rel]
        return SimpleNamespace(size=meta["bytes"], etag=meta["sha256"])
    monkeypatch.setattr(branch, "get_hf_file_metadata", head)
    new = copy.deepcopy(c)
    new.update(initialization="reset_s0", kind="smoke", train_blocks=4, validation_blocks=2, test_blocks=2)
    new["branch_source"] = {"run_id": "source", "repos": repos, "first_new_cycle": 2}
    resolved = branch.resolve_branch_source(new)
    return root, resolved, real


def import_into(path, c, real):
    for split, data in (("train", real[:4]), ("validation", real[:2]), ("test", real[4:6])):
        save_array(path / f"data/{split}.npy", data)
    return branch.import_shared_prefix(path, c)


def test_shared_prefix_exact_import_and_subset_scope(tmp_path, source):
    original, c, real = source
    target = tmp_path / "branch"
    receipt = import_into(target, c, real)
    assert receipt["d2_full_tokens_sha256"] != receipt["d2_active_tokens_sha256"]
    for g in (0, 1):
        rel = f"seed_42/gen_{g:02d}/metrics.json"
        assert sha256(target / rel) == sha256(original / rel)
        assert read_json(target / f"seed_42/gen_{g:02d}/import.json")["measurement_scope"] == "original_source_splits"
    assert np.array_equal(np.load(target / "synthetic_for_gen_02/tokens.npy"),
                          np.load(original / "synthetic_for_gen_02/tokens.npy")[:4])
    assert branch.import_shared_prefix(target, c) == receipt
    (target / "seed_42/gen_01/metrics.json").write_text("{}")
    with pytest.raises(RuntimeError, match="corrupt"):
        branch.import_shared_prefix(target, c)


def test_actual_fork_uses_same_d2_but_different_initializers(tmp_path, source):
    original, c, real = source
    reset = tmp_path / "reset"
    continued = tmp_path / "continued"
    import_into(reset, c, real)
    import_into(continued, c, real)
    cont = {**c, "initialization": "continued"}
    starts = []
    adam = torch.optim.AdamW
    def optimizer(*args, **kwargs):
        opt = adam(*args, **kwargs)
        starts.append(len(opt.state))
        return opt
    # Both branches consume the exact same D2, with initializer S0 versus S1.
    from unittest.mock import patch
    with patch.object(torch.optim, "AdamW", optimizer):
        a, ah, ac, al = training_lineage(reset, 2, c, None)
        b, bh, bc, bl = training_lineage(continued, 2, cont, None)
        assert token_hash(a) == token_hash(b)
        assert ah != bh and al["generator_weights_sha256"] == bl["generator_weights_sha256"] == bh
        ar = engine.train(reset, 2, a, real[:2], c, ah, initial_checkpoint=ac, lineage=al)
        br = engine.train(continued, 2, b, real[:2], cont, bh, initial_checkpoint=bc, lineage=bl)
    assert starts == [0, 0]
    assert ar["selected_weights_sha256"] != br["selected_weights_sha256"]
    assert ar["initialization"] == "reset_s0"
    assert ar["initial_weights_sha256"] != al["generator_weights_sha256"]
    assert engine.train(reset, 2, a, real[:2], c, ah, initial_checkpoint=ac, lineage=al) == ar
    with pytest.raises(RuntimeError, match="lineage"):
        engine.train(reset, 2, a, real[:2], c, ah, initial_checkpoint=ac, lineage={**al, "generator_weights_sha256": "bad"})
    # A second reset cycle still starts from S0 and consumes A2's data, including shard recovery.
    model = engine.load_model(c, reset / "seed_42/gen_02/checkpoint")
    with pytest.raises(engine.ValidationInterruption):
        engine.generate(reset, 2, model, real[:4], c, ar["selected_weights_sha256"], inject=True)
    d3, m3 = engine.generate(reset, 2, model, real[:4], c, ar["selected_weights_sha256"], inject=True)
    assert m3["reused_continuations"] == 2
    before = sha256(reset / "seed_42/gen_02/training.json")
    blocks, expected, checkpoint, lineage = training_lineage(reset, 3, c, None)
    assert expected == ah and lineage["generator_weights_sha256"] == ar["selected_weights_sha256"]
    engine.train(reset, 3, blocks, real[:2], c, expected, initial_checkpoint=checkpoint, lineage=lineage)
    assert sha256(reset / "seed_42/gen_02/training.json") == before


def test_source_and_training_corruption_rejected(tmp_path, source):
    original, c, real = source
    changed = copy.deepcopy(c)
    changed["start_checkpoint"]["weights_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="S0 differs"):
        branch.resolve_branch_source(changed)
    changed = copy.deepcopy(c)
    changed["branch_source"]["artifact_index_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="index changed"):
        branch.resolve_branch_source(changed)
    target = tmp_path / "branch"
    import_into(target, c, real)
    manifest = read_json(target / "synthetic_for_gen_02/manifest.json")
    manifest["provenance"]["parent_weights_sha256"] = "0" * 64
    atomic_json(target / "synthetic_for_gen_02/manifest.json", manifest)
    with pytest.raises(RuntimeError, match="generator"):
        training_lineage(target, 2, c, None)
    atomic_json(target / "synthetic_for_gen_02/manifest.json", {**manifest,
                "provenance": {**manifest["provenance"], "parent_weights_sha256": c["branch_source"]["s1_weights_sha256"]}})
    np.save(target / "synthetic_for_gen_02/tokens.npy", np.zeros((4, 64), dtype=np.int32))
    with pytest.raises(RuntimeError, match="corrupt"):
        training_lineage(target, 2, c, None)


def test_generation_honours_repetition_penalty(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "device", lambda: "cpu")
    class Model:
        def eval(self):
            return self
        def generate(self, input_ids, **kwargs):
            assert kwargs["repetition_penalty"] == 1.23
            return torch.cat([input_ids, torch.full_like(input_ids, 6)], dim=1)
    values, manifest = engine.generate(tmp_path, 0, Model(), np.ones((2, 64), dtype=np.int32),
                                      {"precision": "fp32", "generation_shard": 2, "generation_batch": 2,
                                       "repetition_penalty": 1.23}, "parent")
    assert manifest["provenance"]["repetition_penalty"] == 1.23
    assert values.shape == (2, 64)
    with pytest.raises(ValueError):
        engine.decoding_settings({"repetition_penalty": float("nan")})


@pytest.mark.parametrize("kind", ["missing_completion", "incomplete", "wrong_config"])
def test_source_must_be_complete_and_match_its_configuration(source, kind):
    root, c, real = source
    if kind == "missing_completion":
        (root / "completion.json").unlink()
    elif kind == "incomplete":
        atomic_json(root / "completion.json", {"complete": False, "generations": 3})
    else:
        atomic_json(root / "manifest.json", {"config_hash": "0" * 64})
    # Keep the mock index internally consistent, then test the semantic rejection.
    index = read_json(root / "artifact_index.json")
    for rel in ("completion.json", "manifest.json"):
        if (root / rel).exists():
            index["files"][rel].update(sha256=sha256(root / rel), bytes=(root / rel).stat().st_size)
        else:
            index["files"].pop(rel, None)
    atomic_json(root / "artifact_index.json", index)
    from unittest.mock import patch
    def download(run_id, repos, rel, meta, revisions):
        verify_files(root, {rel: meta})
        return root / rel
    new = copy.deepcopy(c)
    new["branch_source"].pop("artifact_index_sha256")
    new["branch_source"].pop("required_files")
    with patch.object(branch, "download_snapshot_file", download):
        with pytest.raises(RuntimeError, match="missing required|not completed|does not match"):
            branch.resolve_branch_source(new)


def test_completed_metrics_alone_cannot_skip_stage_verification(tmp_path):
    c = {"seed": 42}
    training = {"checkpoint_files": {}, "selected_weights_sha256": "parent", "training_blocks": 1}
    gen = tmp_path / "seed_42/gen_02"
    manifest = {"files": {}, "provenance": {"parent_weights_sha256": "parent"}}
    atomic_json(gen / "metrics.json", {"training": training, "generation_timing": manifest})
    atomic_json(tmp_path / "synthetic_for_gen_03/manifest.json", manifest)
    save_array(tmp_path / "synthetic_for_gen_03/tokens.npy", np.ones((2, 64), dtype=np.int32))
    manifest["continuation_tokens_sha256"] = token_hash(np.ones((2, 64), dtype=np.int32))
    atomic_json(gen / "metrics.json", {"training": training, "generation_timing": manifest})
    atomic_json(tmp_path / "synthetic_for_gen_03/manifest.json", manifest)
    with pytest.raises(RuntimeError, match="checksum manifest"):
        verify_completed_stage(tmp_path, 2, c, training)


@pytest.mark.parametrize("source", [{"epochs": 5, "checkpoint_selection": "final_epoch"}], indirect=True)
def test_final_epoch_shared_fork(source, tmp_path):
    original, c, real = source
    reset = tmp_path / "reset-final"
    continued = tmp_path / "continue-final"
    for path in (reset, continued):
        import_into(path, c, real)
    starts = []
    for path, config in ((reset, c), (continued, {**c, "initialization": "continued"})):
        blocks, initial, checkpoint, lineage = training_lineage(path, 2, config, None)
        t = engine.train(path, 2, blocks, real[:2], config, initial, initial_checkpoint=checkpoint, lineage=lineage)
        assert t["selected_epoch"] == t["executed_epochs"] == 5
        assert t["terminal_weights_sha256"] == t["selected_weights_sha256"]
        starts.append((initial, t["input_tokens_sha256"]))
    assert starts[0][0] != starts[1][0] and starts[0][1] == starts[1][1]
    for g in (0, 1):
        rel = f"seed_42/gen_{g:02d}/metrics.json"
        assert sha256(reset / rel) == sha256(continued / rel) == sha256(original / rel)


def test_epoch_one_prefix_cannot_authorise_final_epoch(source):
    original, c, real = source
    c.update(epochs=5, checkpoint_selection="final_epoch")
    with pytest.raises(RuntimeError, match="missing required|selection policy"):
        branch.resolve_branch_source(c)


def test_smoke_source_rejected_for_full_branch(source):
    original, c, real = source
    config = read_json(original / "config.json")
    config["kind"] = "smoke"
    atomic_json(original / "config.json", config)
    atomic_json(original / "manifest.json", {"config_hash": digest(config)})
    index = read_json(original / "artifact_index.json")
    for rel in ("config.json", "manifest.json"):
        index["files"][rel].update(sha256=sha256(original / rel), bytes=(original / rel).stat().st_size)
    atomic_json(original / "artifact_index.json", index)
    c["kind"] = "full"
    c["branch_source"].pop("artifact_index_sha256")
    with pytest.raises(RuntimeError, match="smoke source"):
        branch.resolve_branch_source(c)
