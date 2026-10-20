import numpy as np
import pytest
import torch

from model_collapse.common import sha256, verify_files
from model_collapse.engine import ValidationInterruption, generate
from model_collapse.hub import artifact_files
from model_collapse.supervisor import deadline


def test_corruption_is_rejected(tmp_path):
    p = tmp_path / "tokens.npy"
    p.write_bytes(b"original")
    manifest = {p.name: sha256(p)}
    verify_files(tmp_path, manifest)
    p.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="corrupt"):
        verify_files(tmp_path, manifest)


def test_budget_deadline_does_not_reset_on_resume():
    resources = {"created_at": 100, "session_allowance_usd": .5, "guard_hourly_usd": .25}
    assert deadline(resources) == 7300
    assert deadline(dict(resources)) == 7300


def test_partial_generation_resumes_shards_and_excludes_prompts(tmp_path, monkeypatch):
    import model_collapse.engine as e
    monkeypatch.setattr(e, "device", lambda: "cpu")
    class Model:
        calls = 0
        def eval(self):
            return self
        def generate(self, input_ids, **kwargs):
            self.calls += 1
            assert kwargs["num_beams"] == 5
            assert kwargs["min_new_tokens"] == kwargs["max_new_tokens"] == 64
            return torch.cat([input_ids, torch.full_like(input_ids, 17)], dim=1)
    c = {"generation_shard": 2, "generation_batch": 2, "precision": "fp32"}
    prompts = np.full((6, 64), 3, dtype=np.int32)
    m = Model()
    with pytest.raises(ValidationInterruption):
        generate(tmp_path, 0, m, prompts, c, "parent", inject=True)
    assert not (tmp_path / "synthetic_for_gen_01/manifest.json").exists()
    assert m.calls == 1
    values, manifest = generate(tmp_path, 0, m, prompts, c, "parent", inject=True)
    assert m.calls == 3
    assert values.shape == prompts.shape and np.all(values == 17)
    assert manifest["reused_continuations"] == 2
    verify_files(tmp_path / "synthetic_for_gen_01", manifest["files"])
    with pytest.raises(RuntimeError, match="provenance"):
        generate(tmp_path, 0, m, prompts, c, "different-parent")


def test_backup_excludes_secrets_optimizer_and_partial_files(tmp_path):
    for rel in (".env", "secrets.json", "x.partial", "seed_42/gen_00/resume/epoch.pt", "seed_42/gen_00/checkpoint/model.safetensors", "config.json"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("test")
    files = artifact_files(tmp_path)
    assert set(files) == {"config.json", "seed_42/gen_00/checkpoint/model.safetensors"}
    assert files["seed_42/gen_00/checkpoint/model.safetensors"]["repo_type"] == "model"
