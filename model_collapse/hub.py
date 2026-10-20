from __future__ import annotations

import os
import uuid
from pathlib import Path

from huggingface_hub import HfApi, get_token, hf_hub_download

from .common import STATE, atomic_json, load_secrets, read_json, sha256, verify_files


def destinations(probe=False):
    load_secrets()
    api = HfApi(token=os.environ.get("HF_TOKEN") or get_token())
    user = api.whoami()["name"]
    repos = {"model": os.environ.get("HF_MODEL_REPO") or f"{user}/iib-model-collapse-week1-models",
             "dataset": os.environ.get("HF_DATASET_REPO") or f"{user}/iib-model-collapse-week1-data"}
    for kind, repo in repos.items():
        api.create_repo(repo, repo_type=kind, private=True, exist_ok=True)
        if not api.repo_info(repo, repo_type=kind).private:
            raise RuntimeError(f"Artifact repository must be private: {repo}")
        if probe:
            payload = ("Week 1 authenticated round-trip " + uuid.uuid4().hex).encode()
            remote = "_probes/" + uuid.uuid4().hex + ".txt"
            api.upload_file(path_or_fileobj=payload, path_in_repo=remote, repo_id=repo, repo_type=kind,
                            commit_message="Verify private artifact recovery")
            p = hf_hub_download(repo, remote, repo_type=kind, token=api.token, force_download=True)
            if Path(p).read_bytes() != payload:
                raise RuntimeError("Private artifact round-trip failed")
    result = {"user": user, "repos": repos, "private_upload_download_verified": probe}
    atomic_json(STATE / "hub.json", result)
    return result


def artifact_files(root):
    root = Path(root)
    files = {}
    volatile = {"artifact_index.json", "status.json", "events.jsonl", "worker.lock", "supervisor.log",
                "watchdog.json", "supervisor.json", "pipeline.log", "stop_request.json", "watchdog_exit.json"}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name in volatile or ".partial" in p.name or "resume" in p.parts or p.name.startswith(".") or "secret" in p.name.lower():
            continue
        rel = p.relative_to(root).as_posix()
        if rel.startswith(".injection"):
            continue
        files[rel] = {"sha256": sha256(p), "bytes": p.stat().st_size,
                      "repo_type": "model" if "/checkpoint/" in rel else "dataset"}
    return files


def backup(root, repos=None):
    root = Path(root)
    repos = repos or read_json(root / "resources.json")["repos"]
    api = HfApi(token=os.environ.get("HF_TOKEN") or get_token())
    files = artifact_files(root)
    index = {"run_id": root.name, "repos": repos, "files": files}
    atomic_json(root / "artifact_index.json", index)
    for kind in ("model", "dataset"):
        paths = [p for p, meta in files.items() if meta["repo_type"] == kind]
        if kind == "dataset":
            paths += ["artifact_index.json", "events.jsonl", "status.json"]
        # Exact allow list prevents accidental credential/cache uploads.
        api.upload_folder(repo_id=repos[kind], repo_type=kind, folder_path=root, path_in_repo=root.name,
                          allow_patterns=paths, commit_message=f"Save {root.name} verified pipeline state")
    return index


def fetch(run_id, repos, out):
    out = Path(out)
    api = HfApi(token=os.environ.get("HF_TOKEN") or get_token())
    idxpath = hf_hub_download(repos["dataset"], f"{run_id}/artifact_index.json", repo_type="dataset", token=api.token, force_download=True)
    index = read_json(idxpath)
    import shutil
    for rel, meta in index["files"].items():
        src = hf_hub_download(repos[meta["repo_type"]], f"{run_id}/{rel}", repo_type=meta["repo_type"], token=api.token)
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    atomic_json(out / "artifact_index.json", index)
    verify_files(out, index["files"])
    return {"verified_files": len(index["files"]), "verified_bytes": sum(x["bytes"] for x in index["files"].values()), "path": str(out)}
