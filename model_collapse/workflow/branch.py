"""Pinned shared-prefix import. Large source objects are downloaded only on the worker."""
from __future__ import annotations

import copy
import gc
import shutil
from pathlib import Path, PurePosixPath

import numpy as np
from huggingface_hub import get_hf_file_metadata, hf_hub_url

from ..common import atomic_json, digest, read_json, sha256, verify_files
from ..data import load_blocks, save_array
from ..hub import download_snapshot_file, resolve_snapshot
from .lineage import token_hash
from .selection import FINAL, selection_policy, verify_selection


def _relative(value):
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or "\\" in value:
        raise ValueError("Source artifact path must be a safe relative POSIX path")
    return value


def resolve_branch_source(c):
    """Freeze source commits/index and prove large-file hashes without downloading weights."""
    c = copy.deepcopy(c)
    if c.get("initialization") != "reset_s0":
        return c
    source = c["branch_source"]
    if len(PurePosixPath(_relative(source["run_id"])).parts) != 1:
        raise ValueError("Shared source run ID must identify one run")
    if source.get("first_new_cycle") != 2:
        raise ValueError("The shared-prefix branch must begin new training at cycle 2")
    repos = source["repos"]
    revisions = resolve_snapshot(repos, source.get("revisions"))
    from huggingface_hub import hf_hub_download
    index_path = Path(hf_hub_download(repos["dataset"], f"{source['run_id']}/artifact_index.json",
                                     repo_type="dataset", revision=revisions["dataset"]))
    index = read_json(index_path)
    if index.get("run_id") != source["run_id"] or index.get("repos") != repos:
        raise RuntimeError("Shared source index identifies a different run or repository")
    frozen_index = source.get("artifact_index_sha256")
    if frozen_index and sha256(index_path) != frozen_index:
        raise RuntimeError("Shared source artifact index changed")
    files = index["files"]
    seed = f"seed_{c['seed']}"
    required = {"config.json", "manifest.json", "completion.json", "data/manifest.json", "data/train.npy", "data/validation.npy", "data/test.npy",
                "synthetic_for_gen_02/tokens.npy", "synthetic_for_gen_02/manifest.json"}
    for g in (0, 1):
        prefix = f"{seed}/gen_{g:02d}/"
        required.update(p for p in files if p.startswith(prefix) and "/resume/" not in p)
        required.update(prefix + name for name in ("training.json", "metrics.json", "test_sequences.json",
                                                   "reference_sequences.json", "samples.json"))
        if selection_policy(c) == FINAL:
            required.add(prefix + "stage_completion.json")
            if g == 1:
                required.add(prefix + "training_history.json")
    if not required.issubset(files):
        raise RuntimeError("Shared source is missing required S0/S1/D2 evidence")
    chosen = {p: files[p] for p in sorted(required)}
    metadata = {}
    for rel, meta in chosen.items():
        _relative(rel)
        if len(meta.get("sha256", "")) != 64 or meta["repo_type"] not in ("model", "dataset"):
            raise RuntimeError("Shared source contains invalid file metadata")
        if rel.endswith((".npy", ".safetensors")):
            head = get_hf_file_metadata(hf_hub_url(repos[meta["repo_type"]], f"{source['run_id']}/{rel}",
                                                repo_type=meta["repo_type"], revision=revisions[meta["repo_type"]]))
            if head.size != meta["bytes"] or head.etag.strip('"') != meta["sha256"]:
                raise RuntimeError(f"Remote source object checksum or size differs: {rel}")
        else:
            path = download_snapshot_file(source["run_id"], repos, rel, meta, revisions)
            if rel.endswith(".json"):
                metadata[rel] = read_json(path)
    original = metadata["config.json"]
    completion = metadata["completion.json"]
    if not completion.get("complete") or completion.get("generations", -1) < 2:
        raise RuntimeError("Shared source experiment has not completed the required prefix")
    if metadata["manifest.json"].get("config_hash") != digest(original):
        raise RuntimeError("Shared source manifest does not match its recorded configuration")
    if original.get("initialization") != "continued":
        raise RuntimeError("Shared source must be the continued-weight experiment")
    if c["kind"] == "full" and original["kind"] != "full":
        raise RuntimeError("Full reset branch cannot import a smoke source")
    if selection_policy(original) != selection_policy(c):
        raise RuntimeError("Shared source checkpoint selection policy differs")
    for key in ("seed", "model", "model_revision", "dataset", "dataset_revision", "dataset_config", "block_size",
                "new_tokens", "num_beams", "repetition_penalty", "precision", "learning_rate", "min_learning_rate",
                "weight_decay", "microbatch", "effective_batch", "generation_batch", "eval_batch"):
        if original[key] != c[key]:
            raise RuntimeError(f"Shared-source scientific setting differs: {key}")
    if c["kind"] == "full" and original["epochs"] != c["epochs"]:
        raise RuntimeError("Full reset branch must match the source epoch schedule")
    s0 = metadata[f"{seed}/gen_00/training.json"]
    s1 = metadata[f"{seed}/gen_01/training.json"]
    d2 = metadata["synthetic_for_gen_02/manifest.json"]
    if selection_policy(c) == FINAL:
        if original["epochs"] != c["epochs"]:
            raise RuntimeError("Final-epoch source must match the epoch schedule")
        verify_selection(s1, metadata[f"{seed}/gen_01/training_history.json"], c)
        for g in (0, 1):
            prefix = f"{seed}/gen_{g:02d}/"
            stage = metadata[prefix + "stage_completion.json"]
            if stage["selected_weights_sha256"] != metadata[prefix + "training.json"]["selected_weights_sha256"]:
                raise RuntimeError("Shared source completed stage checkpoint differs")
            for rel, checksum in stage["files"].items():
                if files.get(prefix + rel, {}).get("sha256") != checksum:
                    raise RuntimeError("Shared source completed stage output differs")
    if s0["selected_weights_sha256"] != c["start_checkpoint"]["weights_sha256"]:
        raise RuntimeError("Shared source S0 differs from the configured fixed initializer")
    if s0["checkpoint_files"] != c["start_checkpoint"]["files"]:
        raise RuntimeError("Shared source S0 checkpoint files differ from the configured fixed initializer")
    if s1["initial_weights_sha256"] != s0["selected_weights_sha256"]:
        raise RuntimeError("Shared source S1 was not initialized from S0")
    if d2["provenance"]["parent_weights_sha256"] != s1["selected_weights_sha256"]:
        raise RuntimeError("Shared source D2 was not generated by S1")
    m0, m1 = (metadata[f"{seed}/gen_{g:02d}/metrics.json"] for g in (0, 1))
    if m0["training"] != s0 or m1["training"] != s1 or m1["generation_timing"] != d2:
        raise RuntimeError("Shared source metrics disagree with their stage provenance")
    if s1["input_tokens_sha256"] != m0["generation_timing"]["continuation_tokens_sha256"]:
        raise RuntimeError("Shared source S1 did not consume S0's exact continuations")
    previous_files = source.get("required_files")
    if previous_files and previous_files != chosen:
        raise RuntimeError("Pinned shared source required files changed")
    source.update(revisions=revisions, artifact_index_sha256=sha256(index_path), required_files=chosen,
                  s0_weights_sha256=s0["selected_weights_sha256"], s1_weights_sha256=s1["selected_weights_sha256"],
                  d2_tokens_sha256=d2["continuation_tokens_sha256"], source_config_sha256=files["config.json"]["sha256"])
    return c


def import_shared_prefix(root, c):
    """Import immutable historical measurements; explicitly label their original split scope."""
    from ..engine import load_model, weights_hash
    root = Path(root)
    source = c["branch_source"]
    required = source.get("required_files")
    if not required or not source.get("artifact_index_sha256"):
        raise RuntimeError("Resolve and pin the shared source before launching a worker")
    existing = read_json(root / "shared_prefix.json")
    if existing:
        if existing["source_identity"] != digest(source):
            raise RuntimeError("Cannot resume with a different shared-prefix source")
        verify_files(root, existing["files"])
        return existing
    stage_files, source_arrays = {}, {}
    for rel, meta in required.items():
        src = download_snapshot_file(source["run_id"], source["repos"], rel, meta, source["revisions"])
        if rel.startswith("data/") and rel.endswith(".npy"):
            source_arrays[Path(rel).stem] = np.load(src, mmap_mode="r", allow_pickle=False)
            continue
        if rel.startswith(f"seed_{c['seed']}/"):
            dest = root / rel
            if dest.exists() and sha256(dest) != meta["sha256"]:
                raise RuntimeError(f"Existing shared-prefix artifact differs: {rel}")
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest.with_name(dest.name + ".partial"))
                dest.with_name(dest.name + ".partial").replace(dest)
            stage_files[rel] = meta["sha256"]
        elif rel == "synthetic_for_gen_02/tokens.npy":
            source_arrays["d2"] = np.load(src, mmap_mode="r", allow_pickle=False)
        else:
            dest = root / "shared_source" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            stage_files[dest.relative_to(root).as_posix()] = meta["sha256"]
    for split in ("train", "validation", "test"):
        actual = load_blocks(root, split)
        original = source_arrays[split]
        if not np.array_equal(actual, original[:len(actual)]) or (c["kind"] == "full" and actual.shape != original.shape):
            raise RuntimeError(f"Prepared {split} blocks differ from the shared source")
    full = source_arrays["d2"]
    if token_hash(full) != source["d2_tokens_sha256"] or full.shape != source_arrays["train"].shape:
        raise RuntimeError("Pinned D2 token hash or source dataset shape differs")
    active = np.asarray(full[:len(load_blocks(root, "train"))])
    dataset = root / "synthetic_for_gen_02"
    original_manifest = read_json(root / "shared_source/synthetic_for_gen_02/manifest.json")
    if original_manifest["provenance"]["prompt_tokens_sha256"] != token_hash(source_arrays["train"]):
        raise RuntimeError("Shared source D2 prompts differ from the fixed real training blocks")
    save_array(dataset / "tokens.npy", active)
    imported_manifest = {**original_manifest, "files": {"tokens.npy": sha256(dataset / "tokens.npy")},
                         "continuation_tokens_sha256": token_hash(active),
                         "imported_source": {"run_id": source["run_id"], "revisions": source["revisions"],
                                             "full_tokens_sha256": token_hash(full), "full_blocks": len(full),
                                             "used_blocks": len(active), "subset": "first_blocks" if len(active) < len(full) else None}}
    atomic_json(dataset / "manifest.json", imported_manifest)
    for name in ("tokens.npy", "manifest.json"):
        stage_files[f"synthetic_for_gen_02/{name}"] = sha256(dataset / name)
    for g in (0, 1):
        gen = root / f"seed_{c['seed']}" / f"gen_{g:02d}"
        model = load_model(c, gen / "checkpoint")
        expected = source[f"s{g}_weights_sha256"]
        if weights_hash(model) != expected:
            raise RuntimeError("Imported checkpoint tensor hash differs from the shared source")
        del model
        gc.collect()
        receipt = {"source_run_id": source["run_id"], "source_stage_index": g, "source_revisions": source["revisions"],
                   "measurement_scope": "original_source_splits", "execution": "imported_shared_stage",
                   "new_training_seconds": 0, "new_generation_seconds": 0}
        atomic_json(gen / "import.json", receipt)
        stage_files[(gen / "import.json").relative_to(root).as_posix()] = sha256(gen / "import.json")
    result = {"source_identity": digest(source), "source_run_id": source["run_id"], "source_revisions": source["revisions"],
              "first_new_cycle": 2, "files": stage_files,
              "d2_full_tokens_sha256": token_hash(full), "d2_active_tokens_sha256": token_hash(active),
              "shared_metrics_scope": "original_source_splits_not_recomputed_for_smoke",
              "newly_executed_shared_stages": 0}
    atomic_json(root / "shared_prefix.json", result)
    return result
