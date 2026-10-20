from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
STATE = PROJECT / ".state"


def now():
    return time.time()


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def event(root, name, **details):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": now(), "event": name, **details}, allow_nan=False) + "\n")


def load_secrets():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return  # The bootstrap watchdog uses only stdlib and inherited secrets.
    load_dotenv(PROJECT / ".env", override=False)


def config(path):
    import yaml
    c = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_config(c)
    return c


def validate_config(c):
    for key in ("model_revision", "dataset_revision"):
        if len(c[key]) != 40:
            raise ValueError(f"{key} must be a pinned commit SHA")
    if c["block_size"] != 64 or c["new_tokens"] != 64 or c["num_beams"] != 5:
        raise ValueError("Week 1 requires 64-token blocks/continuations and five beams")
    if c["effective_batch"] % c["microbatch"] or c["microbatch"] <= 0:
        raise ValueError("microbatch must divide the effective batch")
    if c["generations"] < 0 or c["epochs"] < 1 or c["precision"] not in ("bf16", "fp32"):
        raise ValueError("Invalid training settings")
    if c["kind"] == "full" and any(c[k] is not None for k in ("train_blocks", "validation_blocks", "test_blocks")):
        raise ValueError("Full experiments must use all three real splits")


def checkpoint_manifest(directory):
    directory = Path(directory)
    return {p.name: sha256(p) for p in sorted(directory.iterdir()) if p.is_file()}


def verify_files(root, files):
    for relative, expected in files.items():
        p = Path(root) / relative
        expected = expected["sha256"] if isinstance(expected, dict) else expected
        if not p.is_file() or sha256(p) != expected:
            raise RuntimeError(f"Missing or corrupt artifact: {relative}")
