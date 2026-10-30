"""Import the exact historical S0 continuations, without regenerating D1."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url

from ..common import atomic_json, digest, read_json, sha256, verify_files
from ..data import load_blocks, save_array
from ..hub import download_snapshot_file, resolve_snapshot
from .lineage import token_hash


def resolve_initial_dataset(c):
    c = copy.deepcopy(c)
    source = c.get("initial_dataset")
    if not source:
        return c
    run_id = source["run_id"]
    if Path(run_id).name != run_id or "/" in run_id or "\\" in run_id or run_id in (".", ".."):
        raise ValueError("Unsafe initial dataset run ID")
    repos = source["repos"]
    revisions = resolve_snapshot(repos, source.get("revisions"))
    idx = Path(hf_hub_download(repos["dataset"], f"{run_id}/artifact_index.json",
                               repo_type="dataset", revision=revisions["dataset"]))
    index = read_json(idx)
    if index.get("run_id") != run_id or index.get("repos") != repos:
        raise RuntimeError("Initial dataset index identity differs")
    if source.get("artifact_index_sha256") and sha256(idx) != source["artifact_index_sha256"]:
        raise RuntimeError("Initial dataset index changed")
    seed = f"seed_{c['seed']}/gen_00"
    required = {"config.json", "manifest.json", "data/train.npy", "data/manifest.json",
                "synthetic_for_gen_01/tokens.npy", "synthetic_for_gen_01/manifest.json",
                f"{seed}/training.json", f"{seed}/metrics.json"}
    if not required.issubset(index["files"]):
        raise RuntimeError("Initial dataset source evidence is incomplete")
    files = {p: index["files"][p] for p in sorted(required)}
    metadata = {}
    for rel, meta in files.items():
        if rel.endswith(".npy"):
            head = get_hf_file_metadata(hf_hub_url(repos[meta["repo_type"]], f"{run_id}/{rel}",
                                                  repo_type=meta["repo_type"], revision=revisions[meta["repo_type"]]))
            if head.size != meta["bytes"] or head.etag.strip('"') != meta["sha256"]:
                raise RuntimeError("Initial dataset object checksum differs")
        else:
            metadata[rel] = read_json(download_snapshot_file(run_id, repos, rel, meta, revisions))
    original = metadata["config.json"]
    if original["kind"] != "full" or original.get("initialization") != "continued":
        raise RuntimeError("D1 must come from the historical full continued run")
    if metadata["manifest.json"]["config_hash"] != digest(original):
        raise RuntimeError("Initial dataset source config differs from manifest")
    for key in ("seed", "model", "model_revision", "dataset", "dataset_revision", "dataset_config",
                "block_size", "new_tokens", "num_beams", "repetition_penalty", "precision", "generation_batch"):
        if original[key] != c[key]:
            raise RuntimeError(f"Initial dataset scientific setting differs: {key}")
    training, metrics = metadata[f"{seed}/training.json"], metadata[f"{seed}/metrics.json"]
    manifest = metadata["synthetic_for_gen_01/manifest.json"]
    if (training["selected_weights_sha256"] != c["start_checkpoint"]["weights_sha256"] or
            training["checkpoint_files"] != c["start_checkpoint"]["files"] or
            metrics["training"] != training or metrics["generation_timing"] != manifest or
            manifest["provenance"]["parent_weights_sha256"] != training["selected_weights_sha256"]):
        raise RuntimeError("Initial dataset generator is not the exact fixed S0")
    p = manifest["provenance"]
    if p["new_tokens"] != 64 or p["num_beams"] != 5 or p.get("repetition_penalty", 1.0) != 1.0:
        raise RuntimeError("Initial dataset decoding differs")
    if p["prompt_count"] != source.get("expected_blocks", p["prompt_count"]):
        raise RuntimeError("Initial dataset full token budget differs")
    if metadata["data/manifest.json"]["files"]["train.npy"] != files["data/train.npy"]["sha256"]:
        raise RuntimeError("Initial dataset source split manifest differs")
    if source.get("required_files") and files != source["required_files"]:
        raise RuntimeError("Initial dataset pinned evidence changed")
    source.update(revisions=revisions, artifact_index_sha256=sha256(idx), required_files=files,
                  tokens_sha256=manifest["continuation_tokens_sha256"],
                  full_blocks=p["prompt_count"],
                  prompt_tokens_sha256=p["prompt_tokens_sha256"], parent_weights_sha256=p["parent_weights_sha256"])
    return c


def import_initial_dataset(root, c):
    root = Path(root)
    source = c["initial_dataset"]
    existing = read_json(root / "initial_dataset_import.json")
    if existing:
        if existing["source_identity"] != digest(source):
            raise RuntimeError("Initial dataset source changed")
        verify_files(root, existing["files"])
        return existing
    def get(rel):
        return download_snapshot_file(source["run_id"], source["repos"], rel,
                                      source["required_files"][rel], source["revisions"])
    full_prompts = np.load(get("data/train.npy"), allow_pickle=False)
    full_tokens = np.load(get("synthetic_for_gen_01/tokens.npy"), allow_pickle=False)
    original = read_json(get("synthetic_for_gen_01/manifest.json"))
    prompts = load_blocks(root, "train")
    if (full_tokens.shape != full_prompts.shape or full_tokens.ndim != 2 or full_tokens.shape[1] != 64 or
            len(full_tokens) != source["full_blocks"] or
            token_hash(full_tokens) != source["tokens_sha256"] or
            token_hash(full_prompts) != source["prompt_tokens_sha256"] or
            not np.array_equal(prompts, full_prompts[:len(prompts)]) or
            (c["kind"] == "full" and prompts.shape != full_prompts.shape)):
        raise RuntimeError("Initial dataset tokens, prompts or shape differ")
    dest = root / "synthetic_for_gen_01"
    active = full_tokens[:len(prompts)]
    if (dest / "tokens.npy").exists() and not np.array_equal(np.load(dest / "tokens.npy", allow_pickle=False), active):
        raise RuntimeError("Existing initial dataset differs from pinned source")
    save_array(dest / "tokens.npy", active)
    provenance = {**original["provenance"], "prompt_count": len(prompts), "prompt_tokens_sha256": token_hash(prompts)}
    manifest = {"provenance": provenance, "continuation_tokens_sha256": token_hash(active),
                "files": {"tokens.npy": sha256(dest / "tokens.npy")}, "seconds": 0.0,
                "reused_continuations": len(active), "new_tokens_per_second": 0.0,
                "imported_source": {"identity": copy.deepcopy(source), "original_manifest": original,
                                    "used_blocks": len(active), "full_blocks": len(full_tokens)}}
    previous = read_json(dest / "manifest.json")
    if previous is not None and previous != manifest:
        raise RuntimeError("Existing initial dataset manifest differs from pinned source")
    atomic_json(dest / "manifest.json", manifest)
    receipt = {"source_identity": digest(source), "files": {
        f"synthetic_for_gen_01/{name}": sha256(dest / name) for name in ("tokens.npy", "manifest.json")},
        "full_tokens_sha256": token_hash(full_tokens), "active_tokens_sha256": token_hash(active),
        "new_generation_seconds": 0}
    atomic_json(root / "initial_dataset_import.json", receipt)
    return receipt
