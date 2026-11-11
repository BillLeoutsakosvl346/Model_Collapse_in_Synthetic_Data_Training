"""Cold terminal-stage restoration never claims whole-run bulk recovery."""
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch
from transformers import OPTConfig, OPTForCausalLM

from model_collapse import engine, hub
from model_collapse.common import atomic_json, read_json, sha256
from model_collapse.completion_data import assemble
from model_collapse.workflow import completion
from model_collapse.workflow.completion_restore import verify_terminal_restore


def test_terminal_restore_pinned_checksums_optimizer_and_direct_evaluation(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(engine, "device", lambda: "cpu")
    root, child = tmp_path / "runs/parent", tmp_path / "runs/child"
    root.mkdir(parents=True)
    stage = child / "seed_42/gen_03"
    checkpoint = stage / "checkpoint"
    model = OPTForCausalLM(OPTConfig(vocab_size=20, hidden_size=8, ffn_dim=16, num_hidden_layers=1,
        num_attention_heads=2, max_position_embeddings=256, word_embed_proj_dim=8, dropout=0., attention_dropout=0.))
    model.save_pretrained(checkpoint, safe_serialization=True)
    c = {"generations": 3, "evaluation_pairs": 4, "bos_token_id": 2, "precision": "fp32", "eval_batch": 2, "seed": 42}
    bank = tmp_path / "bank"
    folder = bank / "test"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(42)
    prompt, target = (rng.integers(3, 20, (4,64), dtype=np.int32) for _ in range(2))
    np.save(folder / "prompts.npy", prompt)
    np.save(folder / "targets.npy", target)
    atomic_json(folder / "manifest.json", {"pairs": 4, "files": {p.name: sha256(p) for p in folder.glob("*.npy")}})
    tokens, attention, scoring = assemble(prompt, target, 2)
    expected = engine.evaluate(model, tokens, c, masks=attention, loss_masks=scoring)
    weights = engine.weights_hash(model)
    atomic_json(stage / "training.json", {"selected_weights_sha256": weights, "global_step": 6})
    atomic_json(stage / "metrics.json", {"test": expected})
    (stage / "optimizer_state").mkdir()
    torch.save({"model_weights_sha256": weights, "global_step": 6}, stage / "optimizer_state/cycle.pt")
    atomic_json(stage / "stage_completion.json", {"files": {
        p.relative_to(stage).as_posix(): sha256(p) for p in stage.rglob("*") if p.is_file()}})
    repos = {"model": "private/models", "dataset": "private/data"}
    atomic_json(child / "artifact_index.json", {"files": hub.artifact_files(child)})
    snapshot = {"run_id": "child", "repos": repos, "revisions": {"model": "a"*40, "dataset": "b"*40},
                "artifact_index_sha256": sha256(child / "artifact_index.json")}
    atomic_json(child / "backup_snapshot.json", snapshot)
    def fetch(run_id, actual_repos, out, **kwargs):
        assert kwargs == {"metadata_only": True, "revisions": snapshot["revisions"]}
        assert actual_repos == repos and run_id == "child"
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(child / "artifact_index.json", out / "artifact_index.json")
        return {"artifact_index_sha256": snapshot["artifact_index_sha256"]}
    monkeypatch.setattr(hub, "fetch", fetch)
    from model_collapse.artifacts import hub as canonical
    monkeypatch.setattr(canonical, "download_snapshot_file", lambda run, repos, relative, meta, commits: child / relative)
    monkeypatch.setattr(completion, "bank_path", lambda *_: bank)
    result = verify_terminal_restore(root, child, c)
    assert result["scope"] == "complete_terminal_stage_only"
    assert result["test_loss_absolute_error"] == 0
    assert result["weights_sha256"] == weights
    assert verify_terminal_restore(root, child, c) == result
    (Path(result["restored_path"]) / "optimizer_state/cycle.pt").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="corrupt"):
        verify_terminal_restore(root, child, c)
