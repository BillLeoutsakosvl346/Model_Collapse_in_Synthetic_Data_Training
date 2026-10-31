"""Reconstructible block-level mixtures of real training text and synthetic targets."""
from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np

from .common import atomic_json, read_json, sha256, verify_files
from .data import load_blocks, save_array
from .workflow.lineage import token_hash

VERSION = "block_mixture_v1"
CAMPAIGN = "continued_mixed_50_final5"


def synthetic_fraction(c):
    value = c.get("synthetic_fraction", 1.0)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("synthetic_fraction must be a finite number between zero and one")
    return float(value)


def mixture_enabled(c):
    return synthetic_fraction(c) < 1.0


def mixture_counts(total, fraction):
    fraction = synthetic_fraction({"synthetic_fraction": fraction})
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise ValueError("Mixture total must be a positive integer")
    n_synthetic = int((Decimal(str(fraction)) * total).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return total - n_synthetic, n_synthetic


def reconstruct(real, synthetic, selection):
    """The unshuffled order is real rows followed by synthetic rows."""
    real_ids = np.asarray(selection["real_indices"], dtype=np.int64)
    synthetic_ids = np.asarray(selection["synthetic_indices"], dtype=np.int64)
    order = np.asarray(selection["permutation"], dtype=np.int64)
    rows = np.concatenate((real[real_ids], synthetic[synthetic_ids]), axis=0)
    labels = np.concatenate((np.zeros(len(real_ids), dtype=np.int32), np.ones(len(synthetic_ids), dtype=np.int32)))
    return np.asarray(rows[order], dtype=np.int32), labels[order].tolist()


def sample_mixture(real, synthetic, fraction, seed, cycle):
    for name, values in (("Real", real), ("Synthetic", synthetic)):
        if values.ndim != 2 or values.shape[1] != 64 or values.dtype != np.int32:
            raise ValueError(f"{name} source must contain int32 64-token blocks")
    total = len(real)
    n_real, n_synthetic = mixture_counts(total, fraction)
    if len(synthetic) < n_synthetic:
        raise ValueError("Insufficient synthetic source blocks for sampling without replacement")
    if not isinstance(cycle, int) or isinstance(cycle, bool) or cycle < 1:
        raise ValueError("Mixture cycle must be a positive integer")
    streams = [np.random.default_rng(np.random.SeedSequence([seed, cycle, stream])) for stream in (0, 1, 2)]
    selection = {"real_indices": streams[0].choice(total, n_real, replace=False).tolist(),
                 "synthetic_indices": streams[1].choice(len(synthetic), n_synthetic, replace=False).tolist(),
                 "permutation": streams[2].permutation(total).tolist()}
    blocks, labels = reconstruct(real, synthetic, selection)
    selection["source_labels"] = labels  # 0 = real; 1 = synthetic, in training-array order.
    return blocks, selection


def mixture_input(root, cycle, c, synthetic, source_manifest, generator_hash, *, create=True):
    """Verify sources, then build or verify exactly one immutable mixture per cycle."""
    root = Path(root)
    data_manifest = read_json(root / "data/manifest.json")
    if not data_manifest:
        raise RuntimeError("Real training source manifest is missing")
    verify_files(root / "data", data_manifest["files"])
    real = load_blocks(root, "train")
    source = root / f"synthetic_for_gen_{cycle:02d}"
    # Callers also check generator lineage; this function independently verifies source bytes.
    verify_files(source, source_manifest["files"])
    if token_hash(synthetic) != source_manifest["continuation_tokens_sha256"]:
        raise RuntimeError("Synthetic source token hash differs")
    if source_manifest["provenance"]["parent_weights_sha256"] != generator_hash:
        raise RuntimeError("Mixture generator identity differs")
    fraction = synthetic_fraction(c)
    expected, selection = sample_mixture(real, synthetic, fraction, c["seed"], cycle)
    n_real, n_synthetic = mixture_counts(len(real), fraction)
    sources = {"real_split": "train", "real_file_sha256": sha256(root / "data/train.npy"),
               "real_manifest_sha256": sha256(root / "data/manifest.json"),
               "real_tokens_sha256": token_hash(real), "real_pool_blocks": len(real),
               "synthetic_dataset": source.name, "synthetic_file_sha256": sha256(source / "tokens.npy"),
               "synthetic_manifest_sha256": sha256(source / "manifest.json"),
               "synthetic_tokens_sha256": token_hash(synthetic), "synthetic_pool_blocks": len(synthetic),
               "generator_weights_sha256": generator_hash}
    description = {"version": VERSION, "generation": cycle, "seed": c["seed"],
                   "synthetic_fraction": fraction, "effective_synthetic_fraction": n_synthetic / len(real),
                   "training_blocks": len(real), "block_size": 64, "real_blocks": n_real,
                   "synthetic_blocks": n_synthetic, "real_tokens": n_real * 64,
                   "synthetic_tokens": n_synthetic * 64, "sources": sources,
                   "training_tokens_sha256": token_hash(expected)}
    dest = root / f"mixed_for_gen_{cycle:02d}"
    old = read_json(dest / "manifest.json")
    if old is not None:
        if {k: v for k, v in old.items() if k != "files"} != description:
            raise RuntimeError("Mixture configuration or source identity changed")
        verify_files(dest, old["files"])
        actual_selection = read_json(dest / "selection.json")
        actual = np.load(dest / "tokens.npy", allow_pickle=False)
        if actual_selection != selection or actual.dtype != np.int32 or not np.array_equal(actual, expected):
            raise RuntimeError("Mixture tokens or selection differ from deterministic reconstruction")
        return actual, old
    if not create:
        raise RuntimeError("Mixed training dataset completion manifest is missing")
    save_array(dest / "tokens.npy", expected)
    atomic_json(dest / "selection.json", selection)
    manifest = {**description, "files": {name: sha256(dest / name) for name in ("tokens.npy", "selection.json")}}
    atomic_json(dest / "manifest.json", manifest)  # Commit only after all reconstructible data exists.
    return expected, manifest


def certificate_name(c):
    return "validation_continued_mixed.json" if mixture_enabled(c) else {
        "continued": "validation_continued.json", "reset_s0": "validation_reset_s0.json"}.get(
            c.get("initialization"), "validation.json")
