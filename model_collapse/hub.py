from __future__ import annotations

import os
import re
import shutil
import uuid
import time
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, get_token, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from requests.exceptions import RequestException

from .common import STATE, atomic_json, load_secrets, read_json, sha256, verify_files
from .artifacts.paths import (atomic_copy as _atomic_copy, safe_destination as _safe_destination,
                              safe_relative as _safe_relative)


def retry_transfer(call, *args, **kwargs):
    """Retry transient transport/server failures; permission denials remain terminal."""
    for attempt in range(4):
        try:
            return call(*args, **kwargs)
        except EntryNotFoundError:
            raise  # An optional absent file is not a transient transport failure.
        except RequestException as e:
            response = getattr(e, "response", None)
            status = response.status_code if response is not None else None
            if attempt == 3 or (status is not None and status < 500 and status != 429):
                raise
            time.sleep(min(2 ** (attempt + 1), 10))


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
                "watchdog.json", "supervisor.json", "pipeline.log", "stop_request.json", "watchdog_exit.json",
                "recovery_snapshot.json", "backup_snapshot.json"}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name in volatile or ".partial" in p.name or "resume" in p.parts or p.name.startswith(".") or "secret" in p.name.lower():
            continue
        rel = p.relative_to(root).as_posix()
        if any(part.startswith(".") for part in PurePosixPath(rel).parts) or rel.startswith("fetched_control/"):
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
        retry_transfer(api.upload_folder, repo_id=repos[kind], repo_type=kind, folder_path=root, path_in_repo=root.name,
                          allow_patterns=paths, commit_message=f"Save {root.name} verified pipeline state")
    resolved = resolve_snapshot(repos)
    atomic_json(root / "backup_snapshot.json", {"run_id": root.name, "repos": repos,
                "revisions": resolved, "artifact_index_sha256": sha256(root / "artifact_index.json")})
    return index


def _check_meta(meta):
    if not isinstance(meta, dict) or meta.get("repo_type") not in ("model", "dataset"):
        raise ValueError("Invalid artifact repository type")
    if not isinstance(meta.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", meta["sha256"]):
        raise ValueError("Invalid artifact checksum")
    if not isinstance(meta.get("bytes"), int) or isinstance(meta["bytes"], bool) or meta["bytes"] < 0:
        raise ValueError("Invalid artifact size")


def resolve_snapshot(repos, revisions=None):
    """Resolve each Hub HEAD once, or accept an explicitly immutable commit pair.

    No repository is created or changed. A supplied revision must already be a
    full commit SHA; branch names are not sufficiently stable for recovery.
    """
    revisions = revisions or {}
    api = HfApi(token=os.environ.get("HF_TOKEN") or get_token())
    resolved = {}
    for kind in ("model", "dataset"):
        candidate = revisions.get(kind)
        if candidate is None:
            candidate = retry_transfer(api.repo_info, repos[kind], repo_type=kind).sha
        if not isinstance(candidate, str) or not re.fullmatch(r"[0-9a-f]{40}", candidate):
            raise ValueError(f"{kind} artifact revision must be an immutable commit SHA")
        resolved[kind] = candidate
    return resolved


def download_snapshot_file(run_id, repos, rel, meta, revisions):
    """Download one pinned file to the Hub cache and verify it before use."""
    if _safe_relative(run_id) != run_id or len(PurePosixPath(run_id).parts) != 1:
        raise ValueError("Unsafe run ID")
    _safe_relative(rel)
    _check_meta(meta)
    kind = meta["repo_type"]
    revision = revisions.get(kind)
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Snapshot download requires pinned revisions")
    src = Path(retry_transfer(hf_hub_download, repos[kind], f"{run_id}/{rel}", repo_type=kind,
                             revision=revision, token=os.environ.get("HF_TOKEN") or get_token()))
    if src.stat().st_size != meta["bytes"] or sha256(src) != meta["sha256"]:
        raise RuntimeError(f"Missing or corrupt artifact: {rel}")
    return src


def verify_recovered_files(root, files):
    """Verify index bytes, following explicitly recorded historical controls.

    This is needed only when recovering into an existing working directory whose
    authoritative resource association must not be overwritten by an old backup.
    Empty temporary restores continue to work with common.verify_files directly.
    """
    root = Path(root)
    receipt = read_json(root / "recovery_snapshot.json", {})
    redirects = receipt.get("preserved_control_files", {})
    for relative, expected in files.items():
        _safe_relative(relative)
        digest = expected["sha256"] if isinstance(expected, dict) else expected
        p = _safe_destination(root, relative)
        if p.is_file() and sha256(p) == digest:
            continue
        historic = redirects.get(relative, {})
        target = historic.get("recovered_path")
        if not target or historic.get("sha256") != digest:
            raise RuntimeError(f"Missing or corrupt artifact: {relative}")
        _safe_destination(root, target)
        verify_files(root, {target: digest})


def fetch(run_id, repos, out, *, metadata_only=False, revisions=None):
    """Recover a consistent pinned snapshot without changing live control state."""
    if _safe_relative(run_id) != run_id or len(PurePosixPath(run_id).parts) != 1:
        raise ValueError("Unsafe run ID")
    out = Path(out)
    api = HfApi(token=os.environ.get("HF_TOKEN") or get_token())
    resolved = resolve_snapshot(repos, revisions)
    idxpath = Path(retry_transfer(hf_hub_download, repos["dataset"], f"{run_id}/artifact_index.json",
                                 repo_type="dataset", revision=resolved["dataset"], token=api.token,
                                 force_download=True))
    index = read_json(idxpath)
    if index.get("run_id") != run_id or index.get("repos") != repos:
        raise RuntimeError("Artifact index run/repository identity mismatch")
    # Validate the complete index before selecting or creating any destination.
    for rel, meta in index["files"].items():
        _safe_destination(out, rel)
        _check_meta(meta)
    selected = {p: m for p, m in index["files"].items()
                if not (metadata_only and p.endswith((".npy", ".safetensors")))}
    protected = {"resources.json", "events.jsonl", "status.json", "watchdog_exit.json",
                 "supervisor.json", "watchdog.json", "stop_request.json"}
    redirected = {}
    def store(src, relative, meta=None):
        dst = _safe_destination(out, relative)
        source_hash = sha256(src)
        if relative in protected and dst.exists() and sha256(dst) != source_hash:
            alternate = f"fetched_control/{source_hash}/{relative}"
            _atomic_copy(src, _safe_destination(out, alternate), meta)
            redirected[relative] = {"recovered_path": alternate, "sha256": source_hash,
                                    "authoritative_local_sha256": sha256(dst)}
        else:
            _atomic_copy(src, dst, meta)
    for rel, meta in selected.items():
        src = download_snapshot_file(run_id, repos, rel, meta, resolved)
        store(src, rel, meta)
    # Preserve the original index bytes rather than reserializing its contents.
    _atomic_copy(idxpath, _safe_destination(out, "artifact_index.json"))
    operational = []
    from huggingface_hub.errors import EntryNotFoundError
    for name in ("events.jsonl", "status.json", "watchdog_exit.json"):
        try:
            src = retry_transfer(hf_hub_download, repos["dataset"], f"{run_id}/{name}", repo_type="dataset",
                                 revision=resolved["dataset"], token=api.token, force_download=True)
            store(src, name)
            operational.append(name)
        except EntryNotFoundError:
            pass
    result = {"run_id": run_id, "repos": repos, "revisions": resolved,
              "artifact_index_sha256": sha256(idxpath), "indexed_files": len(index["files"]),
              "verified_files": len(selected), "verified_bytes": sum(x["bytes"] for x in selected.values()),
              "metadata_only": metadata_only, "scope": "metadata_only" if metadata_only else "full",
              "preserved_control_files": redirected,
              "operational_files_outside_immutable_index": operational, "path": str(out)}
    atomic_json(out / "recovery_snapshot.json", result)
    verify_recovered_files(out, selected)
    return result
