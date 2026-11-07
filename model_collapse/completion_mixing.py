"""Frozen continuation replacement with nested, reproducible prompt selections."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .science.identity import token_hash

from .common import atomic_json, read_json, sha256, verify_files
from .data import save_array
from .science.sampling import mixture_counts

POLICY = "nested_seedsequence_6401_v1"


def select_rows(total, fraction, seed, cycle):
    real_count, synthetic_count = mixture_counts(total, fraction)
    if type(cycle) is not int or cycle < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Mixture selection requires a nonnegative seed and positive cycle")
    permutation = np.random.default_rng(np.random.SeedSequence([seed, cycle, 6401])).permutation(total)
    indices = permutation[:synthetic_count]
    labels = np.zeros(total, dtype=np.int32)
    labels[indices] = 1
    return {"permutation": permutation, "synthetic_indices": indices,
            "source_labels": labels, "row_indices": np.arange(total, dtype=np.int64)}, real_count


def freeze_selection(stage, prompt, real, fraction, seed, cycle, generator_hash, *, create=True):
    """Persist the row mapping before any generated shard can be reused."""
    folder = Path(stage) / "mixture"
    if prompt.shape != real.shape or prompt.ndim != 2 or prompt.shape[1] != 64:
        raise ValueError("Completion mixtures require aligned 64-token prompt/target pairs")
    arrays, real_count = select_rows(len(prompt), fraction, seed, cycle)
    description = {"selection_policy": POLICY, "cycle": cycle, "seed": seed,
                   "synthetic_fraction": fraction, "pairs": len(prompt),
                   "real_pairs": real_count, "synthetic_pairs": len(arrays["synthetic_indices"]),
                   "effective_synthetic_fraction": len(arrays["synthetic_indices"]) / len(prompt),
                   "prompt_tokens_sha256": token_hash(prompt), "real_targets_sha256": token_hash(real),
                   "generator_weights_sha256": generator_hash if len(arrays["synthetic_indices"]) else None}
    path = folder / "selection.json"
    previous = read_json(path)
    if previous is not None:
        if {key: previous.get(key) for key in description} != description:
            raise RuntimeError("Frozen completion mixture selection changed")
        verify_files(folder, previous["files"])
        for name, expected in arrays.items():
            if not np.array_equal(np.load(folder / (name + ".npy"), allow_pickle=False), expected):
                raise RuntimeError("Saved completion mixture row mapping differs")
    else:
        if not create:
            raise RuntimeError("Completed mixture selection is missing")
        for name, values in arrays.items():
            save_array(folder / (name + ".npy"), values)
        atomic_json(path, {**description, "files": {name + ".npy": sha256(folder / (name + ".npy")) for name in arrays}})
    return arrays, description


def freeze_targets(stage, real, generated, selection, description, generation_hash, *, create=True):
    """Scatter generated continuations into their recorded original prompt rows."""
    folder = Path(stage) / "mixture"
    indices = selection["synthetic_indices"]
    if len(indices) and (generated is None or generated.shape != (len(indices), 64)):
        raise ValueError("Generated continuations do not match the selected prompt rows")
    targets = np.array(real, dtype=np.int32, copy=True)
    if len(indices):
        targets[indices] = generated
    meta = {**description, "selection_sha256": sha256(folder / "selection.json"),
            "generation_manifest_sha256": generation_hash,
            "generated_targets_sha256": token_hash(generated) if len(indices) else None,
            "mixed_targets_sha256": token_hash(targets), "scored_targets": len(real) * 64}
    path = folder / "manifest.json"
    previous = read_json(path)
    if previous is not None:
        if {key: previous.get(key) for key in meta} != meta:
            raise RuntimeError("Frozen mixed completion targets or provenance changed")
        verify_files(folder, previous["files"])
        if not np.array_equal(np.load(folder / "targets.npy", allow_pickle=False), targets):
            raise RuntimeError("Mixed targets differ from their selected sources")
    else:
        if not create:
            raise RuntimeError("Completed mixture targets are missing")
        save_array(folder / "targets.npy", targets)
        files = {"selection.json": sha256(folder / "selection.json"), "targets.npy": sha256(folder / "targets.npy")}
        files.update(read_json(folder / "selection.json")["files"])
        atomic_json(path, {**meta, "files": files})
    return targets, sha256(path)
