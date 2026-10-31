from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

from .paths import PROJECT, STATE


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
    limit = c.get("max_session_usd")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, (int, float))
                              or not math.isfinite(limit) or limit <= 0):
        raise ValueError("max_session_usd must be a positive finite amount")
    if c.get("workflow") == "paired":
        from .workflow.paired import validate_pair_config
        return validate_pair_config(c)
    from .mixing import mixture_enabled, synthetic_fraction
    synthetic_fraction(c)
    if mixture_enabled(c) and c.get("initialization") != "continued":
        raise ValueError("Mixed-data training currently requires continued initialization")
    from .workflow.selection import selection_policy
    selection_policy(c)
    if c.get("initialization", "fresh_base") not in ("fresh_base", "continued", "reset_s0"):
        raise ValueError("Unknown model initialization mode")
    if c.get("initialization") in ("continued", "reset_s0"):
        start = c.get("start_checkpoint", {})
        if len(start.get("revision", "")) != 40 or len(start.get("weights_sha256", "")) != 64 or not start.get("files"):
            raise ValueError("Checkpoint initialization requires a pinned, checksummed starting checkpoint")
    if c.get("initialization") == "reset_s0":
        branch = c.get("branch_source", {})
        if not branch.get("run_id") or set(branch.get("repos", {})) != {"model", "dataset"}:
            raise ValueError("reset_s0 requires a source run and model/dataset repositories")
        if branch.get("first_new_cycle") != 2 or c["generations"] < 2:
            raise ValueError("The shared-start branch imports cycles 0/1 and first trains cycle 2")
        if any(len(v) != 40 for v in branch.get("revisions", {}).values()):
            raise ValueError("Branch repository revisions must be immutable commit SHAs")
    for key in ("model_revision", "dataset_revision"):
        if len(c[key]) != 40:
            raise ValueError(f"{key} must be a pinned commit SHA")
    if c["block_size"] != 64 or c["new_tokens"] != 64 or c["num_beams"] != 5:
        raise ValueError("These protocols require 64-token blocks/continuations and five beams")
    if c.get("repetition_penalty", 1.0) != 1.0:
        raise ValueError("These validated protocols require repetition_penalty=1.0")
    if c["microbatch"] <= 0 or c["effective_batch"] <= 0 or c["effective_batch"] % c["microbatch"]:
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
