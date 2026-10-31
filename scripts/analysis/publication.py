"""Offline publication helpers, outside the GPU-accepted runtime bundle."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_destination(source, out):
    source, out = Path(source).resolve(), Path(out).resolve()
    published = PROJECT / "reports"
    if out == source or source.is_relative_to(out):
        raise ValueError("Output must be separate from source evidence")
    if out.is_relative_to(source) and not out.is_relative_to(source / "generated"):
        raise ValueError("Only a separate generated directory may be inside source evidence")
    if out.is_relative_to(published):
        parts = out.relative_to(published).parts
        if len(parts) < 3 or parts[1] != "generated":
            raise ValueError("Published reports are protected; use reports/<run>/generated/<new-id>")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Output directory must be new or empty")
    return out


def verified_publication(source):
    source = Path(source).resolve()
    path = source / "publication_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for rel, record in manifest["files"].items():
        target = (source / rel).resolve()
        if not target.is_relative_to(source):
            raise ValueError("Publication manifest path escapes source")
        if not target.is_file() or checksum(target) != record["sha256"]:
            raise RuntimeError(f"Missing/corrupt publication evidence: {rel}")
        if target.stat().st_size != record["bytes"]:
            raise RuntimeError(f"Publication evidence size differs: {rel}")
    return manifest


def write_provenance(source, out, script, inputs, *, scope):
    source, out, script = Path(source).resolve(), Path(out).resolve(), Path(script).resolve()
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT,
                                           stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    versions = {}
    for package in ("numpy", "matplotlib", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    records = {Path(p).resolve().relative_to(source).as_posix(): {
        "sha256": checksum(p), "bytes": Path(p).stat().st_size} for p in sorted(set(map(Path, inputs)))}
    outputs = {p.relative_to(out).as_posix(): {"sha256": checksum(p), "bytes": p.stat().st_size}
               for p in sorted(out.rglob("*")) if p.is_file() and p != out / "publication_reproduction.json"}
    metadata = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_root": str(source), "scope": scope, "inputs": records,
                "analysis": {"revision": revision, "script": script.name, "script_sha256": checksum(script),
                             "helpers": {"publication.py": checksum(__file__)},
                             "python": platform.python_version(), "packages": versions},
                "outputs": outputs, "hash_scope": "Output hashes exclude this self-referential manifest."}
    (out / "publication_reproduction.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
