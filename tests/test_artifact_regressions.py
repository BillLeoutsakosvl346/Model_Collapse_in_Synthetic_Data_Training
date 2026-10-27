"""Offline regressions for immutable recovery and acceptance provenance."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.errors import EntryNotFoundError

from model_collapse.common import atomic_json, read_json, sha256, verify_files
import model_collapse.hub as hub


REPOS = {"model": "owner/models", "dataset": "owner/data"}
REVISIONS = {"model": "a" * 40, "dataset": "b" * 40}


def fake_snapshot(tmp_path, monkeypatch, payloads):
    cache = tmp_path / "cache"
    cache.mkdir()
    calls, info_calls = [], []
    files = {}
    metadata = {}
    for n, (relative, value) in enumerate(payloads.items()):
        p = cache / str(n)
        p.write_bytes(value)
        kind = "model" if "/checkpoint/" in relative else "dataset"
        metadata[relative] = {"sha256": sha256(p), "bytes": len(value), "repo_type": kind}
        files[(kind, "run/" + relative)] = p
    index = {"run_id": "run", "repos": REPOS, "files": metadata}
    p = cache / "index"
    # Deliberately compact bytes to prove fetch does not reserialize the index.
    import json
    p.write_text(json.dumps(index, separators=(",", ":")), encoding="utf-8")
    files[("dataset", "run/artifact_index.json")] = p

    class API:
        token = "offline-test"
        def __init__(self, **kwargs):
            pass
        def repo_info(self, repo, repo_type):
            info_calls.append(repo_type)
            return SimpleNamespace(sha=REVISIONS[repo_type])

    def download(repo, name, *, repo_type, revision, **kwargs):
        calls.append((repo_type, name, revision))
        assert revision == REVISIONS[repo_type]
        try:
            return str(files[(repo_type, name)])
        except KeyError:
            raise EntryNotFoundError("offline missing operational file")

    monkeypatch.setattr(hub, "HfApi", API)
    monkeypatch.setattr(hub, "hf_hub_download", download)
    monkeypatch.setattr(hub, "get_token", lambda: None)
    return index, files, calls, info_calls


def test_fetch_pins_both_repositories_once_and_preserves_index_bytes(tmp_path, monkeypatch):
    index, files, calls, infos = fake_snapshot(tmp_path, monkeypatch, {
        "resources.json": b'{"pod_id":"old"}',
        "seed_42/gen_01/checkpoint/model.safetensors": b"weights",
        "synthetic_for_gen_02/tokens.npy": b"tokens",
    })
    out = tmp_path / "restored"
    result = hub.fetch("run", REPOS, out)
    assert infos == ["model", "dataset"]
    assert result["revisions"] == REVISIONS
    assert result["scope"] == "full"
    assert result["verified_files"] == 3
    assert read_json(out / "recovery_snapshot.json") == result
    assert (out / "artifact_index.json").read_bytes() == files[("dataset", "run/artifact_index.json")].read_bytes()
    assert result["artifact_index_sha256"] == sha256(out / "artifact_index.json")
    assert all(rev == REVISIONS[kind] for kind, _, rev in calls)
    verify_files(out, index["files"])


def test_explicit_revisions_do_not_resolve_mutable_heads(tmp_path, monkeypatch):
    _, _, _, infos = fake_snapshot(tmp_path, monkeypatch, {"config.json": b"{}"})
    hub.fetch("run", REPOS, tmp_path / "out", revisions=REVISIONS)
    assert infos == []
    with pytest.raises(ValueError, match="immutable"):
        hub.resolve_snapshot(REPOS, {"model": "main", "dataset": "b" * 40})


def test_metadata_receipt_does_not_claim_bulk_recovery(tmp_path, monkeypatch):
    _, _, calls, _ = fake_snapshot(tmp_path, monkeypatch, {
        "config.json": b"{}", "data/train.npy": b"tokens",
        "seed_42/gen_01/checkpoint/model.safetensors": b"weights",
    })
    result = hub.fetch("run", REPOS, tmp_path / "out", metadata_only=True)
    assert result["indexed_files"] == 3 and result["verified_files"] == 1
    assert result["scope"] == "metadata_only"
    assert not any(name.endswith((".npy", ".safetensors")) for _, name, _ in calls)


def test_historical_resources_do_not_replace_current_control(tmp_path, monkeypatch):
    index, _, _, _ = fake_snapshot(tmp_path, monkeypatch, {"resources.json": b'{"pod_id":"old"}'})
    out = tmp_path / "active"
    atomic_json(out / "resources.json", {"pod_id": "current", "launch_id": "current"})
    before = (out / "resources.json").read_bytes()
    result = hub.fetch("run", REPOS, out)
    assert (out / "resources.json").read_bytes() == before
    archived = out / result["preserved_control_files"]["resources.json"]["recovered_path"]
    assert archived.read_bytes() == b'{"pod_id":"old"}'
    hub.verify_recovered_files(out, index["files"])
    assert "resources.json" in hub.artifact_files(out)
    assert not any(name.startswith("fetched_control/") for name in hub.artifact_files(out))
    archived.write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="corrupt"):
        hub.verify_recovered_files(out, index["files"])


def test_bad_download_does_not_replace_existing_file(tmp_path, monkeypatch):
    _, files, _, _ = fake_snapshot(tmp_path, monkeypatch, {"config.json": b"correct"})
    out = tmp_path / "out"
    out.mkdir()
    (out / "config.json").write_bytes(b"keep me")
    files[("dataset", "run/config.json")].write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="corrupt"):
        hub.fetch("run", REPOS, out)
    assert (out / "config.json").read_bytes() == b"keep me"
    assert not (out / "recovery_snapshot.json").exists()


@pytest.mark.parametrize("unsafe", ["../outside", "/absolute", "a/../../outside", "a\\outside", "C:/outside", "a//b"])
def test_unsafe_index_paths_rejected_before_destination_write(tmp_path, monkeypatch, unsafe):
    fake_snapshot(tmp_path, monkeypatch, {unsafe: b"bad"})
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="Unsafe"):
        hub.fetch("run", REPOS, out)
    assert not out.exists()


def test_partial_copy_never_replaces_previous_complete_file(tmp_path, monkeypatch):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.write_bytes(b"new")
    dst.write_bytes(b"old")
    def failed_copy(source, target):
        Path(target).write_bytes(b"incomplete")
        raise OSError("injected interruption")
    monkeypatch.setattr(hub.shutil, "copyfile", failed_copy)
    with pytest.raises(OSError, match="injected"):
        hub._atomic_copy(src, dst)
    assert dst.read_bytes() == b"old"
    assert not list(tmp_path.glob("*.partial"))


def test_acceptance_rejects_source_change_before_model_or_provider(tmp_path, monkeypatch):
    import model_collapse.remote as remote
    import model_collapse.validate as validation
    atomic_json(tmp_path / "config.json", {"kind": "smoke"})
    atomic_json(tmp_path / "resources.json", {"code_sha256": "actually-tested-source"})
    monkeypatch.setattr(remote, "source_archive", lambda: (None, "changed-source"))
    def forbidden(*args, **kwargs):
        raise AssertionError("Model/provider must not be touched")
    monkeypatch.setattr(validation, "load_model", forbidden)
    with pytest.raises(RuntimeError, match="Smoke source differs"):
        validation.validate(tmp_path)
    assert not (tmp_path / "validation_results.json").exists()


def test_acceptance_certificate_records_checked_source_not_a_later_archive(tmp_path, monkeypatch):
    import contextlib
    import hashlib
    import math
    import numpy as np
    import torch
    import model_collapse.common as common
    import model_collapse.remote as remote
    import model_collapse.runpod_api as runpod_api
    import model_collapse.validate as validation
    calls = []
    def archive():
        calls.append(True)
        return None, "actually-tested" if len(calls) == 1 else "later-unrelated-source"
    monkeypatch.setattr(remote, "source_archive", archive)
    monkeypatch.setattr(common, "STATE", tmp_path / "private-state")
    monkeypatch.setattr(validation, "verify_files", lambda *args: None)
    real = np.zeros((4, 64), dtype=np.int32)
    monkeypatch.setattr(validation, "load_blocks", lambda *args: real)
    score = {"loss": math.log(2), "nll_sum": math.log(2) * 252,
             "scored_tokens": 252, "perplexity": 2.0}
    metric = {"training": {"initial_weights_sha256": "base", "selected_weights_sha256": "selected",
                           "checkpoint_files": {}, "input_tokens_sha256": hashlib.sha256(real.tobytes()).hexdigest()},
              "generation_timing": {"provenance": {"prompt_tokens_sha256": hashlib.sha256(real.tobytes()).hexdigest()},
                                    "reused_continuations": 2}, "validation": score, "test": score}
    atomic_json(tmp_path / "config.json", {"kind": "smoke", "seed": 42, "generations": 0})
    atomic_json(tmp_path / "resources.json", {"code_sha256": "actually-tested", "pod_id": "owned"})
    atomic_json(tmp_path / "artifact_index.json", {"files": {}})
    atomic_json(tmp_path / "seed_42/pretrained/metrics.json", {"initial_weights_sha256": "base"})
    atomic_json(tmp_path / "seed_42/gen_00/metrics.json", metric)
    atomic_json(tmp_path / "validation_g1_snapshot.json", {})
    atomic_json(tmp_path / "watchdog_exit.json", {"reason": "validation_budget_guard", "pod_id": "owned"})
    directory = tmp_path / "synthetic_for_gen_01"
    directory.mkdir()
    np.save(directory / "tokens.npy", real)
    import json
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps({"event": name}) for name in (
        "validation_worker_resumed", "validation_worker_resumed", "validation_generation_interruption",
        "validation_training_interruption")), encoding="utf-8")
    class Model:
        def eval(self):
            return self
        def __call__(self, input_ids, **kwargs):
            return SimpleNamespace(logits=torch.zeros((*input_ids.shape, 2)))
    monkeypatch.setattr(validation, "load_model", lambda *args: Model())
    monkeypatch.setattr(validation, "weights_hash", lambda *args: "selected")
    monkeypatch.setattr(validation, "evaluate", lambda *args: score)
    monkeypatch.setattr(validation, "tensor", lambda blocks: torch.tensor(blocks, dtype=torch.long))
    monkeypatch.setattr(validation, "autocast", lambda c: contextlib.nullcontext())
    class Provider:
        def request(self, *args):
            return {"pods": [{"id": "owned", "status": "EXITED"}]}
    monkeypatch.setattr(runpod_api, "Runpod", Provider)
    result = validation.validate(tmp_path)
    assert result["source_code_sha256"] == "actually-tested"
    assert len(calls) == 1


def test_backup_receipt_is_outside_immutable_index(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    (root / "config.json").write_bytes(b"{}")
    atomic_json(root / "recovery_snapshot.json", {"old": True})
    calls = []
    class API:
        def __init__(self, **kwargs):
            pass
        def upload_folder(self, **kwargs):
            calls.append(kwargs["repo_type"])
        def repo_info(self, repo, repo_type):
            assert calls == ["model", "dataset"]
            return SimpleNamespace(sha=REVISIONS[repo_type])
    monkeypatch.setattr(hub, "HfApi", API)
    monkeypatch.setattr(hub, "get_token", lambda: None)
    index = hub.backup(root, REPOS)
    assert set(index["files"]) == {"config.json"}
    receipt = read_json(root / "backup_snapshot.json")
    assert receipt["revisions"] == REVISIONS
    assert receipt["artifact_index_sha256"] == sha256(root / "artifact_index.json")
    assert "backup_snapshot.json" not in hub.artifact_files(root)
