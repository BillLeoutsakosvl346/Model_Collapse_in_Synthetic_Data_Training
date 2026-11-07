from types import SimpleNamespace

import pytest

from scripts.operations.hf_storage_admission import admission, check_storage, unique_lfs_bytes


def test_storage_counts_historical_objects_once_per_repository():
    files = [SimpleNamespace(file_oid="old", size=90), SimpleNamespace(file_oid="new", size=10),
             SimpleNamespace(file_oid="old", size=90)]
    assert unique_lfs_bytes(files) == 100
    with pytest.raises(ValueError):
        unique_lfs_bytes([*files, SimpleNamespace(file_oid="old", size=91)])


def test_small_upload_access_cannot_certify_full_capacity():
    with pytest.raises(RuntimeError, match="insufficient"):
        admission(100, 95, 30, 2)
    assert admission(100, 5, 40, 10)["admitted"]
    with pytest.raises(ValueError):
        admission(100, -1, 30, 2)


def test_storage_check_binds_namespace_and_privacy():
    class Api:
        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(private=True, sha="revision", siblings=[])

        def list_lfs_files(self, *args, **kwargs):
            return [SimpleNamespace(file_oid="old", size=20)]

    result = check_storage({"model": "project/models", "dataset": "project/data"},
                           "project", 100, 40, 10, Api())
    assert result["used_bytes"] == 40
    with pytest.raises(ValueError):
        check_storage({"model": "personal/models"}, "project", 100, 40, 10, Api())


def test_resume_rejects_storage_before_the_provider_boundary(tmp_path, monkeypatch):
    from model_collapse.common import atomic_json
    from scripts.operations import resume_completion64

    run_id = "example"
    repos = {"model": "project/models", "dataset": "project/data"}
    atomic_json(tmp_path / "runs" / run_id / "config.json",
                {"workflow": "completion_comparison", "kind": "full"})
    atomic_json(tmp_path / ".state/completion64-new-backup-destination.json",
                {"organization": "project", "repos": repos, "free_plan": True})
    atomic_json(tmp_path / "docs/validation/2026-11-06_completion64-backup-capacity-repair.json",
                {"run_id": run_id, "temporary_cpu_removed": True})
    monkeypatch.setattr(resume_completion64, "PROJECT", tmp_path)
    monkeypatch.setattr(resume_completion64, "STATE", tmp_path / ".state")
    monkeypatch.setattr(resume_completion64, "load_secrets", lambda: None)
    monkeypatch.setattr(resume_completion64, "HfApi", lambda **kwargs: object())
    monkeypatch.setenv("HF_MODEL_REPO", repos["model"])
    monkeypatch.setenv("HF_DATASET_REPO", repos["dataset"])
    calls = []
    monkeypatch.setattr(resume_completion64, "launch", lambda **kwargs: calls.append(kwargs))

    def insufficient(*args):
        return admission(100, 95, 40, 10)

    monkeypatch.setattr(resume_completion64, "check_storage", insufficient)
    with pytest.raises(RuntimeError, match="insufficient"):
        resume_completion64.resume(run_id)
    assert calls == []
