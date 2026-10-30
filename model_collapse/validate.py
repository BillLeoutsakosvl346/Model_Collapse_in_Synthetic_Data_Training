"""Audit the real smoke chain and re-evaluate a checkpoint recovered from the Hub."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .common import atomic_json, read_json, verify_files
from .data import load_blocks
from .engine import autocast, evaluate, load_model, shifted_nll, tensor, weights_hash


def validate(root):
    root = Path(root)
    c = read_json(root / "config.json")
    if c["kind"] != "smoke":
        raise ValueError("validate audits the fault-injected smoke configuration")
    # An acceptance certificate must describe the runtime that produced this run.
    # Check before checkpoint loading, expensive evaluation, or provider reads.
    from .remote import source_archive
    resources = read_json(root / "resources.json") or {}
    tested_source = resources.get("code_sha256")
    current_source = source_archive()[1]
    if not tested_source or tested_source != current_source:
        raise RuntimeError("Smoke source differs from current runtime; run a new acceptance test")
    index = read_json(root / "artifact_index.json")
    verify_files(root, index["files"])
    seed = root / f"seed_{c['seed']}"
    baseline = read_json(seed / "pretrained/metrics.json")
    real = load_blocks(root, "train")
    base_hash = baseline["initial_weights_sha256"]
    checks = {}
    metrics = []
    for g in range(c["generations"] + 1):
        m = read_json(seed / f"gen_{g:02d}/metrics.json")
        if not m:
            raise RuntimeError(f"G{g} is incomplete")
        training = m["training"]
        assert training["initial_weights_sha256"] == base_hash
        assert training["selected_weights_sha256"] != base_hash
        verify_files(seed / f"gen_{g:02d}/checkpoint", training["checkpoint_files"])
        source = real if g == 0 else np.load(root / f"synthetic_for_gen_{g:02d}/tokens.npy", allow_pickle=False)
        assert training["input_tokens_sha256"] == hashlib.sha256(np.asarray(source, dtype=np.int32).tobytes()).hexdigest()
        generated = np.load(root / f"synthetic_for_gen_{g + 1:02d}/tokens.npy", allow_pickle=False)
        provenance = m["generation_timing"]["provenance"]
        assert generated.shape[1] == 64
        assert provenance["prompt_tokens_sha256"] == hashlib.sha256(np.asarray(real[:len(generated)]).tobytes()).hexdigest()
        assert len(source) == len(real)
        for split in ("validation", "test"):
            assert m[split]["scored_tokens"] == len(load_blocks(root, split)) * 63
            assert math.isfinite(m[split]["loss"])
            assert math.isclose(m[split]["perplexity"], math.exp(m[split]["nll_sum"] / m[split]["scored_tokens"]), rel_tol=1e-12)
        metrics.append(m)
    checks["fresh_base_every_generation_and_weights_changed"] = True
    checks["both_exact_dataset_handoffs_and_fixed_real_prompts"] = True
    checks["continuations_exactly_64_tokens"] = True
    checks["finite_token_weighted_perplexity_all_generations"] = True
    snapshot = read_json(root / "validation_g1_snapshot.json")
    verify_files(root, snapshot)
    checks["completed_g1_unchanged_after_resume"] = True
    events = [json.loads(line) for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    names = [e["event"] for e in events]
    assert names.count("validation_worker_resumed") == 2
    assert "validation_generation_interruption" in names and "validation_training_interruption" in names
    assert metrics[0]["generation_timing"]["reused_continuations"] > 0
    checks["two_real_worker_exits_and_resume"] = True
    checks["partial_generation_reused_saved_shard"] = True
    guard = read_json(root / "watchdog_exit.json")
    if guard is None or guard["reason"] != "validation_budget_guard":
        raise RuntimeError("Independent budget-guard shutdown evidence is missing")
    checks["independent_budget_guard_requested_actual_shutdown"] = True
    model = load_model(c, seed / f"gen_{c['generations']:02d}/checkpoint")
    assert weights_hash(model) == metrics[-1]["training"]["selected_weights_sha256"]
    restored = evaluate(model, load_blocks(root, "test"), c)
    # Different supported hardware/PyTorch versions may round BF16 differently.
    assert abs(restored["loss"] - metrics[-1]["test"]["loss"]) < .02
    ids = tensor(load_blocks(root, "test")[:2])
    model.eval()
    with torch.inference_mode(), autocast(c):
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    expected = -torch.log_softmax(logits[:, :-1].float(), dim=-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    actual = shifted_nll(logits, ids)
    error = float((expected - actual).abs().max())
    assert error < 2e-6
    checks["restored_checkpoint_hash_and_evaluation"] = True
    checks["nll_matches_direct_log_probability_calculation"] = True
    result = {"checks": checks, "restored_test": restored, "recorded_test": metrics[-1]["test"],
              "max_nll_calculation_error": error, "verified_artifact_files": len(index["files"])}
    from .runpod_api import Runpod
    from .common import STATE
    pods = Runpod().request("GET", "/pods")["pods"]
    state = next((p["status"] for p in pods if p["id"] == resources["pod_id"]), "ABSENT")
    if state not in ("EXITED", "TERMINATED", "ABSENT") or guard["pod_id"] != resources["pod_id"]:
        raise RuntimeError("Provider has not verified stopped task compute")
    checks["provider_readback_confirms_compute_stopped"] = True
    result["provider_pod_state"] = state
    result["source_code_sha256"] = tested_source
    from .workflow.profiles import protocol_fingerprint
    result["protocol_fingerprint"] = protocol_fingerprint(c)
    result["validated_run_id"] = root.name
    atomic_json(root / "validation_results.json", result)
    atomic_json(STATE / "validation.json", result)
    return result
