"""Private namespace capacity admission including historical LFS objects."""
from __future__ import annotations

def unique_lfs_bytes(files):
    sizes = {}
    for item in files:
        oid, size = item.file_oid, item.size
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("Invalid remote LFS size")
        if oid in sizes and sizes[oid] != size:
            raise ValueError("Conflicting sizes for one LFS object")
        sizes[oid] = size
    return sum(sizes.values())


def admission(quota_bytes, used_bytes, required_bytes, reserve_bytes):
    values = (quota_bytes, used_bytes, required_bytes, reserve_bytes)
    if any(type(value) is not int or value < 0 for value in values) or quota_bytes == 0:
        raise ValueError("Storage amounts must be nonnegative integer byte counts")
    available = quota_bytes - used_bytes - reserve_bytes
    if required_bytes > available:
        raise RuntimeError("Private backup capacity is insufficient for the remaining experiment")
    return {"quota_bytes": quota_bytes, "used_bytes": used_bytes,
            "required_additional_bytes": required_bytes, "reserve_bytes": reserve_bytes,
            "available_after_reserve_bytes": available, "admitted": True}


def check_storage(repos, namespace, quota_bytes, required_bytes, reserve_bytes, api):
    from ..common import now
    evidence = {}
    for kind, repo in repos.items():
        if kind not in ("model", "dataset") or repo.split("/")[0] != namespace:
            raise ValueError("Backup repositories must belong to the declared dedicated namespace")
        info = api.repo_info(repo, repo_type=kind, files_metadata=True)
        if not info.private:
            raise RuntimeError("Backup repositories must be private")
        lfs_bytes = unique_lfs_bytes(api.list_lfs_files(repo, repo_type=kind))
        native_bytes = sum((item.size or 0) for item in info.siblings if not item.lfs)
        evidence[kind] = {"repo": repo, "revision": info.sha,
                          "lfs_bytes_including_history": lfs_bytes,
                          "current_native_bytes": native_bytes}
    # Count separately per repository, conservatively allowing no cross-repo deduplication.
    used = sum(item["lfs_bytes_including_history"] + item["current_native_bytes"]
               for item in evidence.values())
    result = admission(quota_bytes, used, required_bytes, reserve_bytes)
    return {**result, "namespace": namespace, "repositories": evidence, "checked_at": now(),
            "scope": "All repositories in the dedicated experiment namespace must be supplied. "
                     "Native Git history and ordinary metadata growth are covered by the reserve."}

