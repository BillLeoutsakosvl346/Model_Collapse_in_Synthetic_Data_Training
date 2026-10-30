import copy
from types import SimpleNamespace

import numpy as np
import pytest

from model_collapse import engine
from model_collapse.common import atomic_json, digest, read_json, sha256, verify_files
from model_collapse.data import save_array
from model_collapse.hub import artifact_files
from model_collapse.workflow import initial_data as data
from model_collapse.workflow.lineage import token_hash


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    root = tmp_path / "historical"
    prompts = np.random.default_rng(5).integers(4, 32, (8, 64), dtype=np.int32)
    tokens = prompts[:, ::-1].copy()
    c = {"seed": 42, "kind": "full", "initialization": "continued", "model": "test", "model_revision": "a" * 40,
         "dataset": "real", "dataset_revision": "b" * 40, "dataset_config": "raw", "block_size": 64,
         "new_tokens": 64, "num_beams": 5, "repetition_penalty": 1.0, "precision": "fp32", "generation_batch": 2,
         "start_checkpoint": {"weights_sha256": "c" * 64, "files": {"model.safetensors": "d" * 64}}}
    save_array(root / "data/train.npy", prompts)
    atomic_json(root / "data/manifest.json", {"files": {"train.npy": sha256(root / "data/train.npy")}})
    save_array(root / "synthetic_for_gen_01/tokens.npy", tokens)
    manifest = {"provenance": {"parent_weights_sha256": "c" * 64, "prompt_tokens_sha256": token_hash(prompts),
                              "prompt_count": 8, "new_tokens": 64, "num_beams": 5, "repetition_penalty": 1.0},
                "continuation_tokens_sha256": token_hash(tokens), "seconds": 123.}
    atomic_json(root / "synthetic_for_gen_01/manifest.json", manifest)
    training = {"selected_weights_sha256": "c" * 64, "checkpoint_files": c["start_checkpoint"]["files"]}
    atomic_json(root / "seed_42/gen_00/training.json", training)
    atomic_json(root / "seed_42/gen_00/metrics.json", {"training": training, "generation_timing": manifest})
    atomic_json(root / "config.json", c)
    atomic_json(root / "manifest.json", {"config_hash": digest(c)})
    repos = {"model": "test/models", "dataset": "test/data"}
    index = {"run_id": root.name, "repos": repos, "files": artifact_files(root)}
    atomic_json(root / "artifact_index.json", index)
    monkeypatch.setattr(data, "hf_hub_download", lambda *a, **k: str(root / "artifact_index.json"))
    monkeypatch.setattr(data, "resolve_snapshot", lambda *a: {"model": "e" * 40, "dataset": "f" * 40})
    def download(run_id, repos, rel, meta, revisions):
        verify_files(root, {rel: meta})
        return root / rel
    monkeypatch.setattr(data, "download_snapshot_file", download)
    def head(url):
        m = index["files"][url.split("/historical/")[1]]
        return SimpleNamespace(size=m["bytes"], etag=m["sha256"])
    monkeypatch.setattr(data, "get_hf_file_metadata", head)
    new = {**c, "kind": "smoke", "initial_dataset": {"run_id": root.name, "repos": repos, "expected_blocks": 8}}
    new = data.resolve_initial_dataset(new)
    return root, new, prompts, tokens


@pytest.mark.parametrize("count", [4, 8])
def test_import_is_exact_reused_and_scope_is_explicit(pinned, tmp_path, count):
    source, c, prompts, tokens = pinned
    root = tmp_path / "new"
    c["kind"] = "full" if count == 8 else "smoke"
    save_array(root / "data/train.npy", prompts[:count])
    receipt = data.import_initial_dataset(root, c)
    assert receipt == data.import_initial_dataset(root, c)
    class NoGeneration:
        def generate(self, *a, **k):
            pytest.fail("Pinned D1 must never be regenerated")
    active, m = engine.generate(root, 0, NoGeneration(), prompts[:count], c, "c" * 64, inject=True)
    assert np.array_equal(active, tokens[:count])
    assert m["provenance"]["prompt_tokens_sha256"] == token_hash(prompts[:count])
    assert m["imported_source"]["original_manifest"]["provenance"]["prompt_count"] == 8
    assert m["seconds"] == 0 and not (root / ".injection_generation_consumed").exists()


@pytest.mark.parametrize("bad", ["prompts", "tokens", "manifest", "source_identity"])
def test_initial_import_rejects_corruption(pinned, tmp_path, bad):
    source, c, prompts, tokens = pinned
    root = tmp_path / "new"
    save_array(root / "data/train.npy", prompts[:4])
    if bad == "prompts":
        save_array(root / "data/train.npy", prompts[4:])
    elif bad == "tokens":
        save_array(root / "synthetic_for_gen_01/tokens.npy", np.zeros((4, 64), dtype=np.int32))
    else:
        data.import_initial_dataset(root, c)
        if bad == "manifest":
            atomic_json(root / "synthetic_for_gen_01/manifest.json", {})
        else:
            c["initial_dataset"]["revisions"]["model"] = "0" * 40
    with pytest.raises(RuntimeError, match="differ|changed|corrupt"):
        data.import_initial_dataset(root, c)


def test_partial_import_resumes_only_exact_tokens(pinned, tmp_path):
    source, c, prompts, tokens = pinned
    root = tmp_path / "new"
    save_array(root / "data/train.npy", prompts[:4])
    save_array(root / "synthetic_for_gen_01/tokens.npy", tokens[:4])
    receipt = data.import_initial_dataset(root, c)
    assert receipt["active_tokens_sha256"] == token_hash(tokens[:4])
