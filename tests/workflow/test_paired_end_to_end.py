"""Real tiny OPT executes the complete paired workflow; only external boundaries are replaced.

This is CPU numerical/integration verification, not provider or OPT-125M acceptance.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import OPTConfig, OPTForCausalLM

from model_collapse import engine, hub, pipeline, supervisor
from model_collapse.common import PROJECT, atomic_json, checkpoint_manifest, config, digest, read_json, sha256, verify_files
from model_collapse.data import save_array
from model_collapse.workflow import branch, initial_data, paired
from model_collapse.workflow.lineage import token_hash


def test_actual_tiny_opt_pair_and_all_report_families(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(engine, "device", lambda: "cpu")
    root = tmp_path / "pair"
    c = config(PROJECT / "comparisons/shared_start_final_epoch5/smoke.yaml")
    repos = {"model": "test/models", "dataset": "test/data"}
    engine.seed_all(42)
    model = OPTForCausalLM(OPTConfig(vocab_size=32, hidden_size=16, word_embed_proj_dim=16,
                                   ffn_dim=32, num_hidden_layers=1, num_attention_heads=2,
                                   max_position_embeddings=128, dropout=.1))
    historical = tmp_path / "historical"
    s0 = historical / "seed_42/gen_00/checkpoint"
    model.save_pretrained(s0)
    s0hash = engine.weights_hash(model)
    real = np.random.default_rng(9).integers(4, 32, (256, 64), dtype=np.int32)
    val, test = real[:64].copy(), real[64:128].copy()
    initial_tokens = np.roll(real, 1, axis=1).copy()
    save_array(historical / "data/train.npy", real)
    save_array(historical / "synthetic_for_gen_01/tokens.npy", initial_tokens)
    original_d1 = {"provenance": {"parent_weights_sha256": s0hash, "prompt_tokens_sha256": token_hash(real),
                                  "prompt_count": 256, "new_tokens": 64, "num_beams": 5, "repetition_penalty": 1.0},
                   "continuation_tokens_sha256": token_hash(initial_tokens), "seconds": 0.0}
    atomic_json(historical / "synthetic_for_gen_01/manifest.json", original_d1)
    start = {"repo": repos["model"], "revision": "a" * 40, "path": "historical/seed_42/gen_00/checkpoint",
             "weights_sha256": s0hash, "files": checkpoint_manifest(s0)}
    required = hub.artifact_files(historical)
    source = {"run_id": "historical", "repos": repos, "revisions": {"model": "a" * 40, "dataset": "b" * 40},
              "artifact_index_sha256": "c" * 64, "required_files": required,
              "tokens_sha256": token_hash(initial_tokens), "prompt_tokens_sha256": token_hash(real),
              "parent_weights_sha256": s0hash, "full_blocks": 256, "expected_blocks": 256}
    for a in c["arms"].values():
        a.update(start_checkpoint=start, precision="fp32")
    c["arms"]["continued"]["initial_dataset"] = source
    atomic_json(root / "config.json", c)
    resources = {"run_id": root.name, "launch_id": "cpu-test", "pod_id": "mock-provider", "volume_id": "mock-volume",
                 "created_at": 1., "session_allowance_usd": .75, "guard_hourly_usd": .66,
                 "code_sha256": "cpu-test-runtime", "repos": repos}
    atomic_json(root / "resources.json", resources)

    def model_loader(c, checkpoint=None):
        return OPTForCausalLM.from_pretrained(checkpoint or s0)
    monkeypatch.setattr(engine, "load_model", model_loader)
    monkeypatch.setattr(pipeline, "load_model", model_loader)
    monkeypatch.setattr(pipeline, "environment", lambda: {"torch": "2.8.0+cu128", "packages": {"transformers": "4.37.2", "datasets": "3.4.1"}})
    monkeypatch.setattr(paired.shutil, "disk_usage", lambda *a: SimpleNamespace(free=30_000_000_000))

    class Tokenizer:
        def decode(self, values): return " ".join(map(str, values))
        def save_pretrained(self, path): atomic_json(Path(path) / "tokenizer.json", {"tiny_test": True})
    monkeypatch.setattr(pipeline.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    def prepare(path, c, tokenizer):
        dest = Path(path) / "data"
        old = read_json(dest / "manifest.json")
        if old:
            verify_files(dest, old["files"])
            return old
        arrays = {"train": real, "validation": val, "test": test}
        for name, values in arrays.items(): save_array(dest / (name + ".npy"), values)
        m = {"files": {name + ".npy": sha256(dest / (name + ".npy")) for name in arrays},
             "stats": {name: {"full_blocks": len(values), "used_blocks": len(values)} for name, values in arrays.items()}}
        atomic_json(dest / "manifest.json", m)
        return m
    monkeypatch.setattr(pipeline, "prepare", prepare)
    def benchmark(path, c, prompts, stats):
        b = {"training_target_tokens_per_second": 30000, "generation_new_tokens_per_second": 1000,
             "full_split_blocks": {k: v["full_blocks"] for k, v in stats.items()}, "gpu": "tiny CPU test"}
        atomic_json(Path(path) / "benchmark.json", b)
        return b
    monkeypatch.setattr(pipeline, "benchmark", benchmark)

    def download(run_id, repos, rel, meta, revisions):
        src = tmp_path / run_id / rel
        verify_files(tmp_path / run_id, {rel: meta})
        return src
    monkeypatch.setattr(branch, "download_snapshot_file", download)
    monkeypatch.setattr(initial_data, "download_snapshot_file", download)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, name, **kw: str(tmp_path / name))
    import model_collapse.continued as continued
    monkeypatch.setattr(continued, "hf_hub_download", lambda repo, name, **kw: str(tmp_path / name))
    def head(url):
        rel = url.split("/resolve/", 1)[1].split("/", 1)[1]
        p = tmp_path / rel
        return SimpleNamespace(size=p.stat().st_size, etag=sha256(p))
    monkeypatch.setattr(branch, "get_hf_file_metadata", head)
    monkeypatch.setattr(branch, "resolve_snapshot", lambda repos, revisions=None: revisions)
    def backup(path, repos=None):
        path = Path(path)
        index = {"run_id": path.name, "repos": resources["repos"], "files": hub.artifact_files(path)}
        atomic_json(path / "artifact_index.json", index)
        atomic_json(path / "backup_snapshot.json", {"run_id": path.name, "repos": resources["repos"],
                    "revisions": {"model": "d" * 40, "dataset": "e" * 40}, "artifact_index_sha256": sha256(path / "artifact_index.json")})
        if read_json(path / "resources.json", {}).get("owner_pair_id"):
            paired.record_child_snapshot(path)
        return index
    monkeypatch.setattr(hub, "backup", backup)
    monkeypatch.setattr(pipeline, "backup", backup)
    monkeypatch.setattr(supervisor, "run_worker", lambda child, **kw: pipeline.run(child, validation=False))
    paired.run_pair(root, validation=False)
    completion = read_json(root / "completion.json")
    assert completion["unique_training_stages"] == 4 and completion["executed_epochs"] == 20
    children = paired.child_roots(root)
    t = {name: read_json(child / "seed_42/gen_02/training.json") for name, child in children.items()}
    assert t["continued"]["input_tokens_sha256"] == t["reset"]["input_tokens_sha256"]
    assert t["continued"]["initial_weights_sha256"] != s0hash == t["reset"]["initial_weights_sha256"]
    assert all(value["selected_epoch"] == value["executed_epochs"] == 5 for value in t.values())
    for g in (0, 1):
        rel = f"seed_42/gen_{g:02d}/metrics.json"
        assert sha256(children["continued"] / rel) == sha256(children["reset"] / rel)
    output = next((root / "generated").iterdir())
    assert (output / "comparison.png").exists()
    for name in children:
        assert (output / name / "report.md").exists()
    before = {p: sha256(p) for p in children["continued"].rglob("*") if p.is_file()}
    paired.report_pair(root, out=tmp_path / "regenerated")
    assert all(sha256(p) == checksum for p, checksum in before.items())
