"""Evaluation-only regressions. No provider or Hugging Face access."""
import json
from pathlib import Path

import numpy as np
import pytest

from model_collapse.common import atomic_json, sha256
from scripts.data.evaluate import comparison_rows, load_inputs


def test_comparison_uses_within_corpus_changes_and_equal_target_counts():
    values = {"pretrained": {"x": {"loss": 3., "perplexity": 20., "scored_tokens": 12}},
              "fineweb_epoch1": {"x": {"loss": 2.9, "perplexity": 18., "scored_tokens": 12}}}
    row = comparison_rows(values)[0]
    assert row["perplexity_change_percent"] == pytest.approx(-10)
    assert row["nll_change"] == pytest.approx(-.1)
    values["fineweb_epoch1"]["x"]["scored_tokens"] = 13
    with pytest.raises(RuntimeError, match="different target"):
        comparison_rows(values)


def test_evaluation_rejects_changed_masks_or_arrays(tmp_path):
    directory = tmp_path / "inputs/x"
    directory.mkdir(parents=True)
    tokens = np.zeros((2, 512), dtype=np.int32)
    masks = tokens.copy()
    masks[:, :5] = 1
    np.save(directory / "tokens.npy", tokens)
    np.save(directory / "masks.npy", masks)
    spec = {"input_files": {f"inputs/x/{n}": sha256(directory / n) for n in ("tokens.npy", "masks.npy")},
            "collections": {"x": {"scored_tokens": 8}}}
    assert load_inputs(tmp_path, spec)["x"][0].shape == (2, 512)
    masks[0, 7] = 1
    np.save(directory / "masks.npy", masks)
    with pytest.raises(RuntimeError, match="corrupt"):
        load_inputs(tmp_path, spec)
    spec["input_files"]["inputs/x/masks.npy"] = sha256(directory / "masks.npy")
    with pytest.raises(RuntimeError, match="masks/counts"):
        load_inputs(tmp_path, spec)


def test_evaluation_backup_uses_only_explicit_dataset_allowlist(tmp_path, monkeypatch):
    from scripts.data.evaluate import backup_analysis
    import huggingface_hub
    calls = []
    class Api:
        def __init__(self, **kwargs):
            pass
        def upload_folder(self, **kwargs):
            calls.append(kwargs)
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    atomic_json(tmp_path / "completion.json", {"complete": True})
    (tmp_path / ".secret").write_text("excluded")
    backup_analysis(tmp_path, {"model": "private/models", "dataset": "private/data"})
    assert calls[0]["repo_id"] == "private/data"
    assert calls[0]["allow_patterns"] == ["completion.json", "artifact_index.json", "status.json", "events.jsonl"]
    (tmp_path / "seed_42/gen_00/checkpoint").mkdir(parents=True)
    (tmp_path / "seed_42/gen_00/checkpoint/weights.bin").write_bytes(b"weights")
    with pytest.raises(RuntimeError, match="must not contain"):
        backup_analysis(tmp_path, {"model": "private/models", "dataset": "private/data"})


def test_evaluation_worker_has_no_training_entry_points():
    import ast
    source = Path("scripts/data/evaluate.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not attributes.intersection({"backward", "step", "train", "AdamW", "train_model"})


def test_failed_evaluation_saves_diagnostic_and_failed_backup(tmp_path, monkeypatch):
    from scripts.data import evaluate
    (tmp_path / "evaluator.py").write_text("test worker")
    atomic_json(tmp_path / "resources.json", {"repos": {"dataset": "private/data"}})
    atomic_json(tmp_path / "experiment.json", {"evaluator_sha256": sha256(tmp_path / "evaluator.py"),
                "model_config": {}, "input_files": {"missing.npy": "0" * 64}, "collections": {}})
    backups = []
    monkeypatch.setattr(evaluate, "backup_analysis", lambda root, repos: backups.append(json.loads((root / "completion.json").read_text())))
    with pytest.raises(RuntimeError, match="corrupt"):
        evaluate.run(tmp_path)
    assert backups and backups[0]["complete"] is False
    assert "Traceback" in json.loads((tmp_path / "failure.json").read_text())["traceback"]
    assert json.loads((tmp_path / "stop_request.json").read_text())["reason"] == "failure"
