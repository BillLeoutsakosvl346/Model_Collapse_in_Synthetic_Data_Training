"""Offline coordinator/recovery exercise with tiny models and in-memory cloud stand-ins."""
import copy
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from transformers import OPTConfig, OPTForCausalLM

from model_collapse import engine, hub, supervisor
from model_collapse.common import atomic_json, now, read_json, sha256, verify_files, event
from model_collapse.workflow import completion, completion_validation


def test_coordinator_releases_cuda_cache_before_child_processes(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("released"))
    completion.release_coordinator_gpu()
    assert calls == ["released"]


def test_missing_ssh_address_is_a_bounded_transient_read(monkeypatch):
    from model_collapse import remote
    clock, attempts = [1000.], []
    monkeypatch.setattr(remote, "now", lambda: clock[0])
    monkeypatch.setattr(remote.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    def read():
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("Pod direct SSH is not ready")
        return {"healthy": True}
    assert remote._transport_read(read, 1020.)["healthy"]
    assert len(attempts) == 3


def test_observation_can_reuse_its_authenticated_session_address(monkeypatch):
    from model_collapse import remote
    from types import SimpleNamespace
    endpoint = {"host": "session-host", "port": 22001, "username": "root"}
    r = {"run_id": "comparison", "pod_id": "pod", "remote_root": "/workspace/runs/comparison",
         "workflow": "completion_comparison", "verified_ssh_direct": endpoint}
    observed = []
    class Client:
        def close(self):
            pass
    def connect(pod, key):
        observed.append(pod["ssh"]["direct"])
        return Client()
    monkeypatch.setattr(remote, "Runpod", lambda: SimpleNamespace(pod=lambda _: {"id": "pod", "status": "RUNNING", "ssh": {"direct": None}}))
    monkeypatch.setattr(remote, "ssh_connect", connect)
    monkeypatch.setattr(remote, "command", lambda *args, **kwargs: '{"healthy": true}')
    assert remote._detached_observation(r)["healthy"]
    assert observed == [endpoint]


def test_completion_startup_waits_for_ssh_propagation_then_proves_progress(tmp_path, monkeypatch):
    from model_collapse import remote
    from types import SimpleNamespace
    clock, attempts = [1000.], []
    r = {"run_id": "comparison", "launch_id": "launch", "pod_id": "pod", "volume_id": "volume",
         "created_at": 1000., "session_allowance_usd": .5, "guard_hourly_usd": .6,
         "remote_root": "/workspace/runs/comparison", "workflow": "completion_comparison"}
    atomic_json(tmp_path / "runs/comparison/config.json", {"workflow": "completion_comparison"})
    def observe(_):
        attempts.append(1)
        if len(attempts) <= 6:
            raise RuntimeError("Pod direct SSH is not ready")
        return {"status": {"state": "running", "phase": "real", "active_child": "comparison-real"},
                "child_status": {"state": "running", "generation": 1, "stage": "training"},
                "event_bytes": int(clock[0]), "epochs": 1, "new_epochs": 1, "finite_new_training": True,
                "completed_new_stages": int(len(attempts) >= 9),
                "supervisor": {"time": clock[0]}, "watchdog": {"time": clock[0]}}
    class Client:
        def open_sftp(self):
            return object()
        def close(self):
            pass
    monkeypatch.setattr(remote, "PROJECT", tmp_path)
    monkeypatch.setattr(remote, "now", lambda: clock[0])
    monkeypatch.setattr(remote.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(remote, "local_resources", lambda _: r)
    monkeypatch.setattr(remote, "_detached_observation", observe)
    monkeypatch.setattr(remote, "Runpod", lambda: SimpleNamespace(pod=lambda _: {}))
    monkeypatch.setattr(remote, "ssh_connect", lambda *args: Client())
    monkeypatch.setattr(remote, "upload_json", lambda *args: None)
    monkeypatch.setattr(remote, "command", lambda *args, **kwargs: "")
    result = remote.verify_detached("comparison", max_wait_seconds=30)
    assert result["progress_after_disconnect"] and result["observations"][1]["new_epochs"] == 1
    assert len(attempts) == 9


def test_config_profiles_and_acceptance_are_distinct():
    from model_collapse.common import config
    from model_collapse.workflow.profiles import protocol_fingerprint, require_acceptance
    base = Path("comparisons/fineweb_real_vs_synthetic_completion64")
    full, smoke = config(base / "full.yaml"), config(base / "smoke.yaml")
    assert protocol_fingerprint(full) == protocol_fingerprint(smoke)
    changed = {**full, "learning_rate": 6e-6}
    assert protocol_fingerprint(changed) != protocol_fingerprint(full)
    with pytest.raises(ValueError):
        completion.validate_config({**full, "epochs": 5})
    with pytest.raises(RuntimeError, match="both completion"):
        require_acceptance(full, {"workflow": "fresh_real", "source_code_sha256": "runtime",
                         "protocol_fingerprint": protocol_fingerprint(full), "checks": {}}, "runtime")


def test_full_bank_scope_rejected_before_provider_reads(tmp_path, monkeypatch):
    c = yaml.safe_load(Path("comparisons/fineweb_real_vs_synthetic_completion64/full.yaml").read_text())
    state = tmp_path / ".state"
    state.mkdir()
    path = tmp_path / "data/fineweb_completion64/manifest.json"
    atomic_json(path, {"contract": "fixed", "collections": {f"cycle_{i:02d}": {"pairs": 256} for i in range(1, 11)}})
    c["prepared_bank"] = {"contract": "fixed", "manifest_sha256": sha256(path), "revisions": {"model": "a" * 40, "dataset": "b" * 40}}
    monkeypatch.setattr(completion, "STATE", state)
    monkeypatch.setattr(completion, "bank_contract", lambda c: "fixed")
    with pytest.raises(RuntimeError, match="complete full-data"):
        completion.resolve_inputs(c)


@pytest.mark.parametrize("mixtures", [False, True, "single", "repeat", "decoding"])
def test_entire_comparison_recovers_workers_handoff_and_preserves_stages(tmp_path, monkeypatch, mixtures):
    decoding = mixtures == "decoding"
    repeated = mixtures == "repeat"
    if repeated or decoding:
        mixtures = False
    import transformers
    torch.set_num_threads(1)
    engine.seed_all(42)
    base = tmp_path / "pretrained"
    original = OPTForCausalLM(OPTConfig(vocab_size=40, hidden_size=16, ffn_dim=32, num_hidden_layers=1,
        num_attention_heads=2, max_position_embeddings=1024, word_embed_proj_dim=16, dropout=0., attention_dropout=0.,
        pad_token_id=1, bos_token_id=2, eos_token_id=2))
    original.save_pretrained(base, safe_serialization=True)
    def load(c, checkpoint=None):
        engine.seed_all(c["seed"])
        return OPTForCausalLM.from_pretrained(checkpoint or base)
    monkeypatch.setattr(engine, "load_model", load)
    monkeypatch.setattr(engine, "device", lambda: "cpu")
    def generated(self, input_ids, attention_mask, **settings):
        assert input_ids.shape[1] == 65 and torch.all(input_ids[:, 0] == 2)
        assert settings["num_beams"] in (1, 5) and settings["max_new_tokens"] == 64
        if settings["do_sample"]:
            assert settings["temperature"] == 1 and settings["top_k"] == 0 and settings["top_p"] == 1
            return torch.cat([input_ids, torch.randint(3, 40, (len(input_ids), 64))], dim=1)
        return torch.cat([input_ids, input_ids[:, -1:].repeat(1, 64)], dim=1)
    monkeypatch.setattr(OPTForCausalLM, "generate", generated)
    class Tokenizer:
        def decode(self, ids):
            return " ".join(map(str, np.asarray(ids).tolist()))
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            atomic_json(Path(path) / "tokenizer_config.json", {"test": True})
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    root = tmp_path / "runs/comparison"
    root.mkdir(parents=True)
    bank = tmp_path / "bank"
    rng = np.random.default_rng(42)
    pairs = 10 if mixtures else 4
    for name in ("validation", "test", "cycle_01", "cycle_02", "cycle_03"):
        folder = bank / name
        folder.mkdir(parents=True)
        for file in ("prompts.npy", "targets.npy"):
            np.save(folder / file, rng.integers(3, 40, (pairs, 64), dtype=np.int32))
        atomic_json(folder / "manifest.json", {"pairs": pairs, "target_tokens": pairs * 64,
                    "files": {file: sha256(folder / file) for file in ("prompts.npy", "targets.npy")}})
    np.save(bank / "probe_indices.npy", np.arange(2))
    folder = bank / "secondary/fineweb_edu"
    folder.mkdir(parents=True)
    np.save(folder / "tokens.npy", rng.integers(3, 40, (2, 512), dtype=np.int32))
    np.save(folder / "masks.npy", np.ones((2, 512), dtype=np.int32))
    definition = "fineweb_completion64_mixtures" if mixtures else "fineweb_real_vs_synthetic_completion64"
    c = yaml.safe_load(Path(f"comparisons/{definition}/smoke.yaml").read_text())
    if decoding:
        c["decoding_arms"] = [{"id": "greedy", "decoder": "greedy"}, {"id": "sampling", "decoder": "sampling"}]
    if repeated:
        c.update(training_chunk_policy="repeat_fixed", fixed_chunk=1)
    if mixtures == "single":
        c["completion_arms"] = [{"id": "mixed80", "synthetic_fraction": .8}]
    c.update(precision="fp32", train_pairs_per_cycle=pairs, evaluation_pairs=4, probe_count=2, microbatch=1,
             effective_batch=2, eval_batch=2, generation_batch=1 if mixtures else 2,
             generation_shard=1 if mixtures else 2, secondary_corpora=["fineweb_edu"],
             prepared_bank={"run_id": "bank", "manifest_sha256": "0" * 64})
    atomic_json(root / "config.json", c)
    atomic_json(root / "resources.json", {"run_id": root.name, "launch_id": "launch", "pod_id": "pod", "volume_id": "volume",
                "created_at": now(), "session_allowance_usd": 100, "guard_hourly_usd": 1, "code_sha256": "offline",
                "repos": {"model": "private/models", "dataset": "private/data"}})
    monkeypatch.setattr(completion, "validate_config", lambda c: None)
    monkeypatch.setattr(completion, "bank_path", lambda root, c: bank)
    monkeypatch.setattr(completion, "restore_bank", lambda ref, path: {"bos_token_id": 2})
    validated = []
    monkeypatch.setattr(completion_validation, "validate_remote", lambda root: validated.append(root))
    def backup(path, repos=None):
        path = Path(path)
        repos = repos or read_json(path / "resources.json")["repos"]
        index = {"run_id": path.name, "repos": repos, "files": hub.artifact_files(path)}
        atomic_json(path / "artifact_index.json", index)
        atomic_json(path / "backup_snapshot.json", {"run_id": path.name, "repos": repos,
                    "revisions": {"model": "a" * 40, "dataset": "b" * 40}, "artifact_index_sha256": sha256(path / "artifact_index.json")})
        if read_json(path / "resources.json", {}).get("owner_completion_id"):
            completion.record_child_snapshot(path)
        return index
    monkeypatch.setattr(hub, "backup", backup)
    releases = []
    monkeypatch.setattr(completion, "release_coordinator_gpu", lambda: releases.append(True))
    def worker(path, validation=False, max_restarts=2):
        assert releases, "Coordinator must release CUDA cache before starting child workers"
        for attempt in range(max_restarts + 1):
            try:
                return completion.run_arm(path, validation)
            except engine.ValidationInterruption:
                if attempt == max_restarts:
                    raise
                event(path, "validation_worker_resumed", interruption=attempt + 1)
    monkeypatch.setattr(supervisor, "run_worker", worker)
    with pytest.raises(engine.ValidationInterruption, match="coordinator"):
        completion.run(root, validation=True)
    first_child = next(iter(completion.child_roots(root).values()))
    real_seals = {i: sha256(first_child / f"seed_42/gen_{i:02d}/stage_completion.json") for i in range(1, 4)}
    event(root, "validation_worker_resumed", interruption=1)
    completion.run(root, validation=True)
    assert read_json(root / "completion.json")["executed_epochs"] == 3 * len(completion.child_roots(root))
    assert validated == [root]
    assert real_seals == {i: sha256(first_child / f"seed_42/gen_{i:02d}/stage_completion.json") for i in range(1, 4)}
    for name, child in completion.child_roots(root).items():
        arm = read_json(child / "config.json")
        completion.verify_completed_arm(child, arm)
        assert read_json(child / "seed_42/gen_03/training.json")["global_step"] == 3 * math.ceil(pairs / 2)
        assert read_json(child / "seed_42/gen_00/metrics.json")["weights_sha256"] == engine.weights_hash(original)
        if repeated:
            inputs = [read_json(child / f"seed_42/gen_{i:02d}/training_input/manifest.json") for i in range(1, 4)]
            assert len({m["prompt_content_sha256"] for m in inputs}) == 1
            assert all(m["lineage"]["source_collection"] == "cycle_01" for m in inputs)
            if name == "real":
                assert len({m["target_content_sha256"] for m in inputs}) == 1
            assert read_json(child / "seed_42/gen_03/metrics.json")["fixed_real_anchor"]["scored_tokens"] == pairs * 64
        if mixtures:
            indexed = hub.artifact_files(child)
            for cycle in range(1, 4):
                stage = child / f"seed_42/gen_{cycle:02d}"
                source = read_json(child / f"synthetic_for_gen_{cycle:02d}/manifest.json")
                training = read_json(stage / "training.json")
                assert source["provenance"]["parent_weights_sha256"] == training["initial_weights_sha256"]
                assert source["provenance"]["prompt_count"] == round(pairs * arm["synthetic_fraction"])
                for relative, meta in read_json(stage / "stage_completion.json")["files"].items():
                    assert indexed[f"seed_42/gen_{cycle:02d}/{relative}"]["sha256"] == meta
    synthetic = list(completion.child_roots(root).values())[-1]
    assert read_json(synthetic / "synthetic_for_gen_02/manifest.json")["reused_continuations"] == (1 if mixtures else 2)
    (synthetic / "seed_42/gen_01/training_input/loss_mask.npy").write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="corrupt"):
        completion.verify_completed_arm(synthetic, read_json(synthetic / "config.json"))
