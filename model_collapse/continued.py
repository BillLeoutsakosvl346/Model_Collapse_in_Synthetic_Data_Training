"""Weight-carryover initialization and real remote smoke acceptance checks.

S0 is the previous full experiment's real-data G0, pinned by Hub commit and SHA.
S(k) generates D(k+1), then its selected weights initialize training on D(k+1).
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from .common import atomic_json, checkpoint_manifest, read_json, sha256, verify_files


def initialize(root, c):
    from .engine import load_model, weights_hash
    dest = Path(root) / f"seed_{c['seed']}" / "gen_00"
    start = c["start_checkpoint"]
    done = read_json(dest / "training.json")
    if done:
        if done.get("source_checkpoint") != start:
            raise RuntimeError("Starting checkpoint provenance changed")
        verify_files(dest / "checkpoint", start["files"])
        return done
    for name, expected in start["files"].items():
        if Path(name).name != name:
            raise ValueError("Starting checkpoint filenames must be flat")
        path = dest / "checkpoint" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            src = hf_hub_download(start["repo"], f"{start['path']}/{name}", revision=start["revision"])
            if sha256(src) != expected:
                raise RuntimeError("Pinned source checkpoint checksum mismatch")
            shutil.copyfile(src, path.with_suffix(path.suffix + ".partial"))
            path.with_suffix(path.suffix + ".partial").replace(path)
    verify_files(dest / "checkpoint", start["files"])
    model = load_model(c, dest / "checkpoint")
    actual = weights_hash(model)
    if actual != start["weights_sha256"]:
        raise RuntimeError("Starting model's tensor hash differs from the recorded real-data G0")
    del model
    gc.collect()
    done = {"generation": 0, "initialization": "existing_real_data_g0", "source_checkpoint": start,
            "initial_weights_sha256": actual, "selected_weights_sha256": actual,
            "input_tokens_sha256": None, "training_blocks": 0, "selected_epoch": None,
            "seconds": 0.0, "checkpoint_files": checkpoint_manifest(dest / "checkpoint")}
    atomic_json(dest / "training.json", done)
    return done


def audit_chain(root):
    """Verify the stored tensors, token arrays, both handoffs, and recovery evidence."""
    from .data import load_blocks
    root = Path(root)
    c = read_json(root / "config.json")
    if c.get("initialization") != "continued" or c["kind"] != "smoke":
        raise ValueError("This acceptance audit requires a continued smoke experiment")
    index = read_json(root / "artifact_index.json")
    verify_files(root, index["files"])
    seed = root / f"seed_{c['seed']}"
    real = load_blocks(root, "train")
    previous_hash = c["start_checkpoint"]["weights_sha256"]
    handoffs = []
    metrics = []
    for g in range(c["generations"] + 1):
        dest = seed / f"gen_{g:02d}"
        m = read_json(dest / "metrics.json")
        t = m["training"]
        verify_files(dest / "checkpoint", t["checkpoint_files"])
        assert t["initial_weights_sha256"] == previous_hash
        if g:
            data = np.load(root / f"synthetic_for_gen_{g:02d}/tokens.npy", allow_pickle=False)
            parent = read_json(root / f"synthetic_for_gen_{g:02d}/manifest.json")
            actual = hashlib.sha256(data.astype(np.int32).tobytes()).hexdigest()
            assert data.shape == real.shape == (256, 64)
            assert t["input_tokens_sha256"] == actual == parent["continuation_tokens_sha256"]
            assert parent["provenance"]["parent_weights_sha256"] == previous_hash
            assert t["selected_weights_sha256"] != previous_hash
            assert t["optimizer_initialization"] == "fresh_each_cycle"
            handoffs.append({"cycle": g, "generator_and_training_initial_weights_sha256": previous_hash,
                             "training_data_sha256": actual, "trained_weights_sha256": t["selected_weights_sha256"]})
        else:
            assert t["source_checkpoint"] == c["start_checkpoint"] and t["training_blocks"] == 0
        output = np.load(root / f"synthetic_for_gen_{g+1:02d}/tokens.npy", allow_pickle=False)
        assert output.shape == (256, 64)
        p = m["generation_timing"]["provenance"]
        assert p["prompt_count"] == 256 and p["new_tokens"] == 64
        assert p["prompt_tokens_sha256"] == hashlib.sha256(np.asarray(real).tobytes()).hexdigest()
        assert p["parent_weights_sha256"] == t["selected_weights_sha256"]
        for split in ("validation", "test"):
            a = m[split]
            assert a["scored_tokens"] == len(load_blocks(root, split)) * 63
            assert math.isfinite(a["loss"])
            assert math.isclose(a["perplexity"], math.exp(a["nll_sum"] / a["scored_tokens"]), rel_tol=1e-12)
        metrics.append(m)
        previous_hash = t["selected_weights_sha256"]
    snapshot = read_json(root / "validation_g1_snapshot.json")
    assert snapshot
    verify_files(root, snapshot)
    names = [json.loads(x)["event"] for x in (root / "events.jsonl").read_text().splitlines()]
    assert names.count("validation_worker_resumed") == 2
    assert "validation_generation_interruption" in names and "validation_training_interruption" in names
    assert metrics[0]["generation_timing"]["reused_continuations"] > 0
    return {"checks": {"pinned_previous_real_g0_start": True, "both_weights_and_data_handoffs": True,
                        "weights_changed_each_training_cycle": True, "fixed_real_prompts_and_evaluation": True,
                        "continuation_only_64_token_datasets": True, "finite_token_weighted_perplexity": True,
                        "two_real_worker_exits_and_resume": True, "partial_generation_shard_reused": True,
                        "completed_cycle_1_unchanged_after_resume": True}, "handoffs": handoffs,
            "recorded_test": metrics[-1]["test"]}


def audit_reset_chain(root):
    """Audit actual reset updates independently from the imported full-data measurements."""
    from .data import load_blocks
    from .workflow.lineage import token_hash, training_lineage, verify_completed_stage
    root = Path(root)
    c = read_json(root / "config.json")
    if c.get("initialization") != "reset_s0" or c["kind"] != "smoke":
        raise ValueError("This acceptance audit requires a reset-S0 smoke experiment")
    index = read_json(root / "artifact_index.json")
    verify_files(root, index["files"])
    shared = read_json(root / "shared_prefix.json")
    assert shared["first_new_cycle"] == 2
    verify_files(root, shared["files"])
    real = load_blocks(root, "train")
    assert real.shape == (256, 64)
    expected = c["start_checkpoint"]["weights_sha256"]
    handoffs = []
    for g in range(2, c["generations"] + 1):
        blocks, initializer, checkpoint, lineage = training_lineage(root, g, c, expected)
        gen = root / f"seed_{c['seed']}" / f"gen_{g:02d}"
        t = read_json(gen / "training.json")
        m = verify_completed_stage(root, g, c, t)
        assert initializer == t["initial_weights_sha256"] == expected
        assert t["lineage"] == lineage
        assert lineage["generator_weights_sha256"] != expected
        assert t["selected_weights_sha256"] != expected
        assert t["input_tokens_sha256"] == token_hash(blocks)
        assert t["optimizer_initialization"] == "fresh_each_cycle"
        assert t["initialization"] == "reset_s0"
        assert blocks.shape == real.shape
        output = np.load(root / f"synthetic_for_gen_{g+1:02d}/tokens.npy", allow_pickle=False)
        assert output.shape == (256, 64)
        p = m["generation_timing"]["provenance"]
        assert p["parent_weights_sha256"] == t["selected_weights_sha256"]
        assert p["prompt_count"] == 256 and p["new_tokens"] == 64
        assert p["prompt_tokens_sha256"] == hashlib.sha256(np.asarray(real).tobytes()).hexdigest()
        for split in ("validation", "test"):
            a = m[split]
            assert a["scored_tokens"] == len(load_blocks(root, split)) * 63
            assert math.isfinite(a["loss"])
            assert math.isclose(a["perplexity"], math.exp(a["nll_sum"] / a["scored_tokens"]), rel_tol=1e-12)
        handoffs.append({"cycle": g, **lineage, "trained_weights_sha256": t["selected_weights_sha256"]})
    snapshot = read_json(root / "validation_g1_snapshot.json")
    assert snapshot
    verify_files(root, snapshot)
    names = [json.loads(x)["event"] for x in (root / "events.jsonl").read_text().splitlines()]
    assert names.count("validation_worker_resumed") == 2
    assert "validation_generation_interruption" in names and "validation_training_interruption" in names
    first = read_json(root / f"seed_{c['seed']}/gen_02/metrics.json")
    assert first["generation_timing"]["reused_continuations"] > 0
    return {"checks": {"exact_pinned_shared_prefix_import": True, "shared_evidence_scope_preserved": True,
                        "fixed_s0_weights_and_separate_generator_handoffs": True,
                        "weights_changed_each_training_cycle": True, "fixed_real_prompts_and_evaluation": True,
                        "continuation_only_64_token_datasets": True, "finite_token_weighted_perplexity": True,
                        "two_real_worker_exits_and_resume": True, "partial_generation_shard_reused": True,
                        "completed_first_new_cycle_unchanged_after_resume": True},
            "handoffs": handoffs, "recorded_test": m["test"], "shared_prefix": shared,
            "source_revisions": c["branch_source"]["revisions"], "first_new_cycle": 2}


def validate_remote(root):
    """Download actual private backups on the Pod and independently reload/evaluate S2."""
    from .engine import autocast, evaluate, load_model, shifted_nll, tensor, weights_hash
    from .data import load_blocks
    from .hub import backup, fetch
    root = Path(root)
    backup(root)
    c = read_json(root / "config.json")
    result = audit_reset_chain(root) if c.get("initialization") == "reset_s0" else audit_chain(root)
    repos = read_json(root / "resources.json")["repos"]
    with tempfile.TemporaryDirectory(prefix="iib-hub-restore-", dir=root.parent) as temp:
        restored = Path(temp)
        recovery = fetch(root.name, repos, restored)
        result["artifact_recovery"] = recovery
        index = read_json(restored / "artifact_index.json")
        # Hash evidence for every object downloaded from the Hub, not the live working copy.
        result["verified_recovered_files"] = {p: m["sha256"] for p, m in index["files"].items()}
        last = restored / f"seed_{c['seed']}" / f"gen_{c['generations']:02d}"
        model = load_model(c, last / "checkpoint")
        assert weights_hash(model) == read_json(last / "training.json")["selected_weights_sha256"]
        reevaluated = evaluate(model, load_blocks(restored, "test"), c)
        assert abs(reevaluated["loss"] - result["recorded_test"]["loss"]) < 1e-7
        ids = tensor(load_blocks(restored, "test")[:2])
        with torch.inference_mode(), autocast(c):
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        direct = -torch.log_softmax(logits[:, :-1].float(), -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
        error = float((direct - shifted_nll(logits, ids)).abs().max())
        assert error < 2e-6
        result.update(restored_test=reevaluated, max_nll_calculation_error=error,
                      source_code_sha256=read_json(root / "resources.json")["code_sha256"])
        result["checks"].update(private_hub_download_hashes_verified=True,
                                restored_checkpoint_tensor_hash_and_evaluation=True,
                                direct_shifted_nll_calculation=True)
        del model
        gc.collect()
    atomic_json(root / "remote_validation.json", result)
    return result


def verify_recovery(root):
    """Metadata stays local; remote acceptance evidence verifies the cloud-only bulk files."""
    from .hub import verify_recovered_files
    root = Path(root)
    index = read_json(root / "artifact_index.json")
    recovery = read_json(root / "recovery_verified.json", {})
    audit = read_json(root / "remote_validation.json", {})
    if not index:
        raise RuntimeError("Artifact index is missing")
    for rel, meta in index["files"].items():
        if (root / rel).is_file():
            verify_recovered_files(root, {rel: meta})
        elif (recovery.get("metadata_only") and rel.endswith((".npy", ".safetensors"))
              and audit.get("verified_recovered_files", {}).get(rel) == meta["sha256"]):
            continue
        else:
            raise RuntimeError(f"Artifact has no verified recovery copy: {rel}")


def validate_local(root):
    from .common import STATE
    from .remote import source_archive, local_resources
    from .runpod_api import Runpod
    root = Path(root)
    verify_recovery(root)
    result = read_json(root / "remote_validation.json")
    if not result or not all(result["checks"].values()):
        raise RuntimeError("Remote acceptance checks are missing or failed")
    r = local_resources(root.name)
    guard = read_json(root / "watchdog_exit.json")
    pod = Runpod().pod(r["pod_id"])
    if not guard or guard["pod_id"] != r["pod_id"] or guard["reason"] != "validation_budget_guard" or pod["status"] != "EXITED":
        raise RuntimeError("Independent budget guard has not verified actual GPU shutdown")
    code_hash = source_archive()[1]
    if result["source_code_sha256"] != code_hash:
        raise RuntimeError("Source changed after the GPU test")
    result["checks"].update(independent_budget_guard_and_provider_stop_verified=True)
    mode = read_json(root / "config.json").get("initialization", "continued")
    result.update(provider_pod_state=pod["status"], validated_run_id=root.name, initialization=mode)
    atomic_json(root / "validation_results.json", result)
    atomic_json(STATE / f"validation_{mode}.json", result)
    return result
