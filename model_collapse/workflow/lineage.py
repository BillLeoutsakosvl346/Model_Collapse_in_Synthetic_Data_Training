"""Verify the distinction between a data generator and a training initializer."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..common import read_json, sha256, verify_files


def token_hash(values):
    return hashlib.sha256(np.asarray(values, dtype=np.int32).tobytes()).hexdigest()


def training_lineage(root, generation, c, base_hash):
    """Return verified input data and initializer; never trust the metrics-file sentinel."""
    root = Path(root)
    mode = c.get("initialization", "fresh_base")
    if generation == 0:
        return None, base_hash, None, None
    source = root / f"synthetic_for_gen_{generation:02d}"
    manifest = read_json(source / "manifest.json")
    if not manifest:
        raise RuntimeError("Training dataset completion manifest is missing")
    verify_files(source, manifest["files"])
    blocks = np.load(source / "tokens.npy", mmap_mode="r", allow_pickle=False)
    if blocks.ndim != 2 or blocks.shape[1] != 64 or not len(blocks):
        raise RuntimeError("Training continuations must be nonempty 64-token blocks")
    if token_hash(blocks) != manifest["continuation_tokens_sha256"]:
        raise RuntimeError("Training dataset token content differs from its manifest")
    parent = root / f"seed_{c['seed']}" / f"gen_{generation - 1:02d}"
    parent_training = read_json(parent / "training.json")
    if not parent_training:
        raise RuntimeError("Dataset generator training record is missing")
    verify_files(parent / "checkpoint", parent_training["checkpoint_files"])
    generator_hash = parent_training["selected_weights_sha256"]
    if manifest["provenance"]["parent_weights_sha256"] != generator_hash:
        raise RuntimeError("Training dataset generator differs from the preceding selected checkpoint")
    if mode == "continued":
        checkpoint, expected = parent / "checkpoint", generator_hash
    elif mode == "reset_s0":
        checkpoint = root / f"seed_{c['seed']}" / "gen_00" / "checkpoint"
        initial = read_json(checkpoint.parent / "training.json")
        verify_files(checkpoint, initial["checkpoint_files"])
        expected = c["start_checkpoint"]["weights_sha256"]
        if initial["selected_weights_sha256"] != expected:
            raise RuntimeError("Fixed-S0 initializer differs from the pinned checkpoint")
    else:
        checkpoint, expected = None, base_hash
    lineage = {"initializer_weights_sha256": expected,
               "initializer_checkpoint": str(checkpoint.relative_to(root)) if checkpoint else c.get("model"),
               "training_dataset": source.name, "training_dataset_file_sha256": sha256(source / "tokens.npy"),
               "training_tokens_sha256": manifest["continuation_tokens_sha256"],
               "generator_checkpoint": str((parent / "checkpoint").relative_to(root)),
               "generator_weights_sha256": generator_hash,
               "branch_origin": c.get("branch_source")}
    from ..mixing import mixture_enabled, mixture_input
    if mixture_enabled(c):
        completed_training = root / f"seed_{c['seed']}/gen_{generation:02d}/training.json"
        blocks, mixed = mixture_input(root, generation, c, blocks, manifest, generator_hash,
                                      create=not completed_training.exists())
        source = root / f"mixed_for_gen_{generation:02d}"
        lineage.update(training_dataset=source.name, training_dataset_file_sha256=sha256(source / "tokens.npy"),
                       training_tokens_sha256=mixed["training_tokens_sha256"],
                       mixture_manifest_sha256=sha256(source / "manifest.json"),
                       synthetic_source_dataset=f"synthetic_for_gen_{generation:02d}",
                       synthetic_source_tokens_sha256=manifest["continuation_tokens_sha256"],
                       mixture={k: mixed[k] for k in ("version", "synthetic_fraction", "effective_synthetic_fraction",
                                                      "real_blocks", "synthetic_blocks", "real_tokens", "synthetic_tokens")})
    return blocks, expected, checkpoint, lineage


def verify_completed_stage(root, generation, c, training):
    gen = Path(root) / f"seed_{c['seed']}" / f"gen_{generation:02d}"
    metrics = read_json(gen / "metrics.json")
    if not metrics or metrics["training"] != training:
        raise RuntimeError("Completed stage metrics disagree with verified training provenance")
    verify_files(gen / "checkpoint", training["checkpoint_files"])
    from ..mixing import mixture_enabled
    if generation and mixture_enabled(c):
        blocks, expected, _, lineage = training_lineage(root, generation, c, None)
        if (lineage != training.get("lineage") or token_hash(blocks) != training["input_tokens_sha256"]
                or expected != training["initial_weights_sha256"]):
            raise RuntimeError("Completed stage mixed training provenance differs")
    output = Path(root) / f"synthetic_for_gen_{generation + 1:02d}"
    manifest = read_json(output / "manifest.json")
    if not manifest or manifest != metrics["generation_timing"]:
        raise RuntimeError("Completed stage generation manifest differs from its metrics")
    verify_files(output, manifest["files"])
    if manifest["provenance"]["parent_weights_sha256"] != training["selected_weights_sha256"]:
        raise RuntimeError("Completed stage output was generated by a different checkpoint")
    blocks = np.load(output / "tokens.npy", mmap_mode="r", allow_pickle=False)
    if token_hash(blocks) != manifest["continuation_tokens_sha256"]:
        raise RuntimeError("Completed stage output token hash differs")
    completion = read_json(gen / "stage_completion.json")
    if not completion:
        raise RuntimeError("Completed stage checksum manifest is missing")
    verify_files(gen, completion["files"])
    names = ["samples.json", "test_sequences.json", "reference_sequences.json"]
    if training["training_blocks"]:
        names.append("training_history.json")
    for name in names:
        if not (gen / name).is_file():
            raise RuntimeError(f"Completed stage evidence is missing: {name}")
    return metrics
