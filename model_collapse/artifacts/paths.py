"""Pure filesystem safety used by Hub recovery and artifact imports.

This module has no provider, model, or credential dependency. hub.py retains its
historical private helper names as compatibility aliases.
"""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path, PurePosixPath

from ..common import sha256


def safe_relative(relative):
    """Treat remote names as portable paths, including when running on Windows."""
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative or "\0" in relative:
        raise ValueError("Unsafe artifact path")
    parts = PurePosixPath(relative).parts
    if PurePosixPath(relative).is_absolute() or any(p in ("", ".", "..") for p in relative.split("/")):
        raise ValueError("Unsafe artifact path")
    if any(Path(p).is_reserved() for p in parts):
        raise ValueError("Unsafe artifact path")
    return relative


def safe_destination(root, relative):
    relative = safe_relative(relative)
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact path escapes recovery directory")
    return path


def atomic_copy(src, dst, expected=None):
    """Do not expose an incomplete file or replace good evidence with bad bytes."""
    src, dst = Path(src), Path(dst)
    actual_hash = sha256(src)
    if expected is not None and (actual_hash != expected["sha256"] or src.stat().st_size != expected["bytes"]):
        raise RuntimeError(f"Missing or corrupt artifact: {dst.name}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + "." + uuid.uuid4().hex + ".partial")
    try:
        shutil.copyfile(src, tmp)
        if sha256(tmp) != actual_hash:
            raise RuntimeError(f"Recovery copy checksum mismatch: {dst.name}")
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


