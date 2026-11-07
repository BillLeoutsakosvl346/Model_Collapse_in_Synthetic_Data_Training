"""Independent recovery and numerical acceptance for fresh-real training."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from ..common import STATE, atomic_json, digest, read_json, sha256, verify_files
from .fresh_real import CHECKS, verify_cycle


def audit_data(root, c, tokenizer):
    from ..fresh_data import document_keys, encode_sequences, load_collection, passage_hashes
    root = Path(root)
    manifest = read_json(root / "data/manifest.json")
    verify_files(root / "data", manifest["files"])
    seen, spans = set(), set()
    for name, info in manifest["collections"].items():
        tokens, masks, _ = load_collection(root, name)
        records = [json.loads(line) for line in (root / "data" / name / "documents.jsonl").read_text(encoding="utf-8").splitlines()]
        sequences, source_count, current = [], 0, set()
        for record in records:
            keys = document_keys(record)
            assert keys and not seen.intersection(keys)
            seen.update(keys)
            ids = tokenizer.encode(record["text"], add_special_tokens=False)[:record["used_source_tokens"]]
            assert len(ids) == record["used_source_tokens"]
            assert hashlib.sha256(np.asarray(ids, dtype=np.int32).tobytes()).hexdigest() == record["used_tokens_sha256"]
            fingerprints = passage_hashes(ids)
            assert not spans.intersection(fingerprints)
            current.update(fingerprints)
            sequences.extend(encode_sequences(ids, tokenizer, c["max_sequence_length"], genuine_end=not record["trimmed"]))
            source_count += len(ids)
        spans.update(current)
        expected = np.full_like(tokens, tokenizer.pad_token_id)
        expected_masks = np.zeros_like(masks)
        for row, ids in enumerate(sequences):
            expected[row, :len(ids)] = ids
            expected_masks[row, :len(ids)] = 1
        assert np.array_equal(tokens, expected) and np.array_equal(masks, expected_masks)
        assert source_count == info["source_text_tokens"]
        if name.startswith("cycle_"):
            assert source_count == c["source_tokens_per_cycle"]
    return {name: {k: value[k] for k in ("documents", "source_text_tokens", "non_padding_tokens", "scored_tokens")}
            for name, value in manifest["collections"].items()}


def validate_remote(root):
    import torch
    from transformers import AutoTokenizer
    from .. import engine
    from ..fresh_data import load_collection
    from ..hub import backup, fetch
    from .profiles import protocol_fingerprint
    root = Path(root)
    c = read_json(root / "config.json")
    assert c["kind"] == "smoke" and c["generations"] == 3
    tokenizer = AutoTokenizer.from_pretrained(c["model"], revision=c["model_revision"])
    data = audit_data(root, c, tokenizer)
    seed = root / f"seed_{c['seed']}"
    baseline = read_json(seed / "gen_00/metrics.json")
    original = engine.load_model(c)
    assert engine.weights_hash(original) == baseline["weights_sha256"]
    del original
    gc.collect()
    previous, parent_hash, global_step = baseline["weights_sha256"], None, 0
    handoffs = []
    for cycle in range(1, 4):
        metrics = verify_cycle(root, cycle, c, previous, parent_hash)
        stage = seed / f"gen_{cycle:02d}"
        t = metrics["training"]
        state_path = stage / "optimizer_state/cycle.pt"
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        assert state["global_step"] == t["global_step"] > global_step
        assert state["scheduler"]["last_epoch"] == state["global_step"]
        assert state["model_weights_sha256"] == t["selected_weights_sha256"] != previous
        assert all(int(item["step"]) == state["global_step"] for item in state["optimizer"]["state"].values())
        history = read_json(stage / "training_history.json")
        assert len(history) == 1 and t["selected_epoch"] == 1
        for split in ("validation", "test"):
            value = metrics[split]
            assert math.isfinite(value["loss"]) and math.isclose(value["perplexity"], math.exp(value["nll_sum"] / value["scored_tokens"]), rel_tol=1e-12)
            assert value["scored_tokens"] == data[split]["scored_tokens"]
        handoffs.append({"cycle": cycle, "initial_weights_sha256": previous, "selected_weights_sha256": t["selected_weights_sha256"],
                         "optimizer_parent_sha256": parent_hash, "global_step": state["global_step"],
                         "learning_rate_end": history[0]["learning_rate_end"]})
        previous, parent_hash, global_step = t["selected_weights_sha256"], sha256(state_path), state["global_step"]
    assert global_step == state["total_steps"]
    assert math.isclose(state["optimizer"]["param_groups"][0]["lr"], c["min_learning_rate"], rel_tol=1e-10)
    verify_files(seed / "gen_01", read_json(root / "validation_cycle1_snapshot.json"))
    events = [json.loads(line)["event"] for line in (root / "events.jsonl").read_text().splitlines()]
    assert events.count("validation_worker_resumed") == 2 and "fresh_real_midcycle_interruption" in events
    backup(root)
    repos = read_json(root / "resources.json")["repos"]
    with tempfile.TemporaryDirectory(prefix="fresh-real-restored-", dir=root.parent) as temporary:
        restored = Path(temporary)
        recovery = fetch(root.name, repos, restored)
        index = read_json(restored / "artifact_index.json")
        verify_files(restored, index["files"])
        last = restored / f"seed_{c['seed']}/gen_03"
        model = engine.load_model(c, last / "checkpoint")
        assert engine.weights_hash(model) == previous
        optimizer_state = torch.load(last / "optimizer_state/cycle.pt", map_location=engine.device(), weights_only=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"])
        optimizer.load_state_dict(optimizer_state["optimizer"])
        assert optimizer_state["global_step"] == global_step
        assert sha256(last / "optimizer_state/cycle.pt") == parent_hash
        test, masks, _ = load_collection(restored, "test")
        evaluated = engine.evaluate(model, test, c, masks=masks)
        assert abs(evaluated["loss"] - metrics["test"]["loss"]) < 1e-7
        ids, attention = engine.tensor(test[:2]), engine.tensor(masks[:2])
        with torch.inference_mode(), engine.autocast(c):
            logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
        direct = -torch.log_softmax(logits[:, :-1].float(), -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
        direct *= attention[:, 1:]
        error = float((direct - engine.shifted_nll(logits, ids, attention)).abs().max())
        assert error < 2e-6
        result = {"workflow": "fresh_real", "source_code_sha256": read_json(root / "resources.json")["code_sha256"],
                  "protocol_fingerprint": protocol_fingerprint(c), "data": data, "handoffs": handoffs,
                  "restored_test": evaluated, "max_nll_calculation_error": error, "artifact_recovery": recovery,
                  "verified_recovered_files": {name: meta["sha256"] for name, meta in index["files"].items()},
                  "checks": {key: True for key in CHECKS - {"detached_progress_verified", "independent_budget_guard_and_provider_stop_verified"}}}
        del optimizer, model, optimizer_state
        gc.collect()
    atomic_json(root / "remote_validation.json", result)
    return result


def verify_recovery(root):
    from ..hub import verify_recovered_files
    root = Path(root)
    index = read_json(root / "artifact_index.json")
    remote = read_json(root / "remote_validation.json", {})
    recovered = read_json(root / "recovery_verified.json", {})
    if not index:
        raise RuntimeError("FineWeb artifact index is missing")
    for name, meta in index["files"].items():
        if (root / name).exists():
            verify_recovered_files(root, {name: meta})
        elif (recovered.get("metadata_only") and name.endswith((".npy", ".safetensors", ".pt", "documents.jsonl"))
              and remote.get("verified_recovered_files", {}).get(name) == meta["sha256"]):
            continue
        else:
            raise RuntimeError(f"Unverified FineWeb recovery object: {name}")


def validate_local(root):
    from ..remote import local_resources, source_archive
    from ..runpod_api import Runpod
    from ..mixing import certificate_name
    from .profiles import protocol_fingerprint, require_acceptance
    root = Path(root)
    c = read_json(root / "config.json")
    r = local_resources(root.name)
    if any(read_json(root / "resources.json", {}).get(key) != r[key] for key in ("run_id", "launch_id", "pod_id", "volume_id", "code_sha256")):
        raise RuntimeError("FineWeb recovery resource ownership differs")
    verify_recovery(root)
    result = read_json(root / "remote_validation.json")
    guard = read_json(root / "watchdog_exit.json", {})
    pod = Runpod().pod(r["pod_id"])
    if guard.get("pod_id") != r["pod_id"] or guard.get("reason") != "validation_budget_guard" or pod["status"] != "EXITED":
        raise RuntimeError("FineWeb watchdog shutdown is not verified")
    runtime = source_archive()[1]
    if r["code_sha256"] != runtime or result.get("source_code_sha256") != runtime:
        raise RuntimeError("Runtime changed after FineWeb acceptance")
    detached = read_json(root / "detached_verification.json", {})
    observations = detached.get("observations", [])
    if (any(detached.get(key) != r[key] for key in ("run_id", "pod_id", "launch_id")) or
            detached.get("progress_after_disconnect") is not True or len(observations) != 2 or
            detached.get("observations_sha256") != digest(observations) or
            observations[-1].get("finite_new_training") is not True):
        raise RuntimeError("FineWeb detached progress was not verified")
    result["checks"].update(detached_progress_verified=True, independent_budget_guard_and_provider_stop_verified=True)
    result.update(validated_run_id=root.name, provider_pod_state=pod["status"], protocol_fingerprint=protocol_fingerprint(c))
    require_acceptance(c, result, runtime)
    atomic_json(root / "validation_results.json", result)
    atomic_json(STATE / certificate_name(c, runtime), result)
    atomic_json(STATE / "benchmark_fresh_real.json", read_json(root / "benchmark.json"))
    return result
