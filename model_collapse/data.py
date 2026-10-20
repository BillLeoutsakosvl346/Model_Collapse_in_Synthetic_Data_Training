from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .common import atomic_json, digest, read_json, sha256, verify_files


def save_array(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("wb") as f:
        np.save(f, np.asarray(values, dtype=np.int32), allow_pickle=False)
    os.replace(tmp, path)


def pack_rows(rows, tokenizer, block_size=64):
    """One deterministic concatenation per split, preserving every source row."""
    ids = []
    for start in range(0, len(rows), 1000):
        batch = tokenizer(rows[start:start + 1000], add_special_tokens=True, truncation=False)["input_ids"]
        for row in batch:
            ids.extend(row)
    count = len(ids) // block_size
    return np.asarray(ids[:count * block_size], dtype=np.int32).reshape(count, block_size), len(ids) % block_size


def prepare(root, c, tokenizer):
    root = Path(root)
    dest = root / "data"
    existing = read_json(dest / "manifest.json")
    if existing:
        if existing["preprocessing_config_hash"] != digest({k: c[k] for k in ("dataset", "dataset_revision", "dataset_config", "model_revision", "block_size", "train_blocks", "validation_blocks", "test_blocks")}):
            raise RuntimeError("Data configuration changed")
        verify_files(dest, existing["files"])
        return existing
    from datasets import load_dataset
    ds = load_dataset(c["dataset"], c["dataset_config"], revision=c["dataset_revision"])
    stats, files = {}, {}
    for split in ("train", "validation", "test"):
        blocks, remainder = pack_rows(ds[split]["text"], tokenizer, c["block_size"])
        total = len(blocks)
        limit = c[split + "_blocks"]
        if limit is not None:
            blocks = blocks[:limit]
        path = dest / (split + ".npy")
        save_array(path, blocks)
        files[path.name] = sha256(path)
        stats[split] = {"rows": len(ds[split]), "full_blocks": total, "used_blocks": len(blocks),
                        "discarded_remainder_tokens": remainder, "fingerprint": ds[split]._fingerprint}
    manifest = {"preprocessing_config_hash": digest({k: c[k] for k in ("dataset", "dataset_revision", "dataset_config", "model_revision", "block_size", "train_blocks", "validation_blocks", "test_blocks")}),
                "stats": stats, "files": files, "policy": "Preserve row strings; OPT default special tokens per row; concatenate once per split; disjoint 64-token blocks; discard split remainder only."}
    atomic_json(dest / "manifest.json", manifest)
    return manifest


def load_blocks(root, split):
    return np.load(Path(root) / "data" / (split + ".npy"), mmap_mode="r", allow_pickle=False)
