"""Enumerate and preserve deployment archives independently of provisioning."""
import hashlib
import io
import json
import subprocess
import tarfile

from ..common import atomic_json, digest, sha256


def source_archive(project, state):
    files = [p for p in (project / "model_collapse").rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")
             and not any(part.startswith(".") for part in p.relative_to(project).parts)]
    files += [project / "pyproject.toml"]
    code_hash = digest({p.relative_to(project).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in sorted(files)})
    archive = state / f"code-{code_hash}.tar.gz"
    state.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {"schema_version": 1, "runtime_sha256": code_hash, "git_commit": commit,
                "runtime_identity_excludes": ["configs", "experiments", "comparisons", "docs", "reports"],
                "files": {p.relative_to(project).as_posix(): sha256(p) for p in sorted(files)}}
    if archive.exists():
        # Runtime identity can stay unchanged across documentation-only commits. Never
        # overwrite the exact archive that an earlier acceptance test actually deployed.
        with tarfile.open(archive, "r:gz") as existing:
            member = existing.extractfile("source_bundle.json")
            saved = json.load(member) if member is not None else None
        if not saved or saved.get("runtime_sha256") != code_hash:
            raise RuntimeError("Existing source archive lacks the matching bundle manifest")
        atomic_json(state / "source_bundle.json", saved)
        return archive, code_hash
    atomic_json(state / "source_bundle.json", manifest)
    with tarfile.open(archive, "w:gz") as tar:
        for p in files:
            tar.add(p, arcname=p.relative_to(project).as_posix())
        payload = (json.dumps(manifest, indent=2) + "\n").encode()
        info = tarfile.TarInfo("source_bundle.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return archive, code_hash
