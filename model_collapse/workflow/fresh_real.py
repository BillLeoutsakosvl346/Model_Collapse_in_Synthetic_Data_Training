"""One real-only trajectory with fresh documents and continuous optimization."""
from __future__ import annotations

import gc
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
from filelock import FileLock

from ..common import atomic_json, checkpoint_manifest, digest, event, now, read_json, sha256, verify_files

ACCEPTANCE = "fineweb_fresh_real_one_epoch"
CHECKS = {"three_real_cycles", "disjoint_documents_and_passages", "exact_source_token_quotas",
          "masked_token_weighted_loss", "terminal_checkpoints_reload", "continuous_optimizer_and_scheduler",
          "worker_interruptions_recovered", "completed_cycle_unchanged", "private_bulk_restore",
          "restored_model_and_optimizer_state", "restored_evaluation", "direct_shifted_nll_calculation",
          "detached_progress_verified", "independent_budget_guard_and_provider_stop_verified"}


def validate_config(c):
    from ..workflow.selection import selection_policy
    if c.get("dataset") != "HuggingFaceFW/fineweb-edu" or c.get("dataset_config") != "sample-10BT":
        raise ValueError("Fresh-real requires the named FineWeb-Edu sample")
    if (c.get("model") != "facebook/opt-125m" or c.get("epochs") != 1 or
            selection_policy(c) != "final_epoch" or c.get("initialization") != "continued" or
            c.get("synthetic_fraction") != 0.0):
        raise ValueError("Fresh-real requires pretrained OPT, one terminal epoch and real-only continued weights")
    for key in ("model_revision", "dataset_revision"):
        if not isinstance(c.get(key), str) or len(c[key]) != 40 or any(x not in "0123456789abcdef" for x in c[key]):
            raise ValueError("Fresh-real model and dataset revisions must be pinned SHAs")
    if (not c.get("dataset_files") or any(not isinstance(p, str) or not p.startswith("sample/10BT/")
            or ".." in p or not p.endswith(".parquet") for p in c["dataset_files"])):
        raise ValueError("Fresh-real requires pinned sample shard paths")
    for key in ("generations", "source_tokens_per_cycle", "validation_documents", "test_documents",
                "shuffle_buffer", "microbatch", "effective_batch", "eval_batch", "max_sequence_length"):
        if type(c.get(key)) is not int or c[key] <= 0:
            raise ValueError(f"Invalid fresh-real count: {key}")
    if c["max_sequence_length"] > 2048 or c["max_sequence_length"] < 3 or c["effective_batch"] % c["microbatch"]:
        raise ValueError("Invalid sequence length or accumulation")
    if c.get("precision") not in ("bf16", "fp32") or c.get("kind") not in ("smoke", "full"):
        raise ValueError("Invalid fresh-real execution profile")
    for key in ("learning_rate", "min_learning_rate", "weight_decay"):
        if not isinstance(c.get(key), (int, float)) or not math.isfinite(c[key]) or c[key] < 0:
            raise ValueError("Invalid fresh-real optimizer setting")
    if not 0 < c["min_learning_rate"] <= c["learning_rate"]:
        raise ValueError("Invalid global learning-rate schedule")
    if c.get("storage_backend", "network") != "network":
        raise ValueError("Fresh-real acceptance uses Standard network storage")


def training_input_hash(tokens, masks):
    import hashlib
    token_hash = hashlib.sha256(np.asarray(tokens, dtype=np.int32).tobytes()).hexdigest()
    mask_hash = hashlib.sha256(np.asarray(masks, dtype=np.int32).tobytes()).hexdigest()
    return hashlib.sha256((token_hash + mask_hash).encode()).hexdigest()


def verify_cycle(root, cycle, c, previous_hash, parent_state_hash):
    from ..fresh_data import load_collection
    from ..workflow.selection import verify_selection
    stage = Path(root) / f"seed_{c['seed']}" / f"gen_{cycle:02d}"
    sealed = read_json(stage / "stage_completion.json")
    if not sealed:
        raise RuntimeError("Incomplete fresh-real cycle")
    verify_files(stage, sealed["files"])
    metrics = read_json(stage / "metrics.json")
    t = metrics["training"]
    tokens, masks, manifest = load_collection(root, f"cycle_{cycle:02d}")
    verify_selection(t, read_json(stage / "training_history.json"), c)
    if (t["initial_weights_sha256"] != previous_hash or t["optimizer_parent_sha256"] != parent_state_hash or
            t["input_tokens_sha256"] != training_input_hash(tokens, masks) or
            t["lineage"]["data_manifest_sha256"] != sha256(Path(root) / "data" / f"cycle_{cycle:02d}" / "manifest.json") or
            manifest["source_text_tokens"] != c["source_tokens_per_cycle"]):
        raise RuntimeError("Fresh-real cycle initializer, optimization state or data changed")
    return metrics


def benchmark(root, c, tokens, masks, data):
    import torch
    from .. import engine
    path = Path(root) / "benchmark.json"
    if path.exists():
        return read_json(path)
    model = engine.load_model(c)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"])
    order = np.argsort(-np.asarray(masks).sum(1))
    rows = np.resize(order, c["effective_batch"])
    times = []
    targets = int(np.asarray(masks[rows])[:, 1:].sum())
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        model.train()
        engine.sync()
        start = time.perf_counter()
        for offset in range(0, len(rows), c["microbatch"]):
            chosen = rows[offset:offset + c["microbatch"]]
            ids, attention = engine.tensor(tokens[chosen]), engine.tensor(masks[chosen])
            labels = ids.masked_fill(~attention.bool(), -100)
            with engine.autocast(c):
                loss = model(input_ids=ids, attention_mask=attention, labels=labels, use_cache=False).loss
            (loss * (int(attention[:, 1:].sum()) / targets)).backward()
        optimizer.step()
        engine.sync()
        times.append(time.perf_counter() - start)
    validation = engine.evaluate(model, tokens[rows], c, masks=masks[rows])
    result = {"workflow": "fresh_real", "training_target_tokens_per_second": targets / float(np.median(times[2:])),
              "evaluation_target_tokens_per_second": validation["scored_tokens"] / validation["seconds"],
              "training_step_seconds": times, "preparation_seconds": data["seconds"],
              "sampled_source_tokens": sum(x["source_text_tokens"] for x in data["collections"].values()),
              "sampled_documents": sum(x["documents"] for x in data["collections"].values()),
              "collections": data["collections"], "max_sequence_length": c["max_sequence_length"],
              "microbatch": c["microbatch"], "effective_batch": c["effective_batch"], "precision": c["precision"],
              "gpu": torch.cuda.get_device_name() if engine.device() == "cuda" else "cpu"}
    atomic_json(path, result)
    del optimizer, model
    gc.collect()
    if engine.device() == "cuda":
        torch.cuda.empty_cache()
    return result


def forecast(bench, c, hourly_usd):
    cycles = c["generations"]
    n = c["source_tokens_per_cycle"] * cycles
    training = n * 1.02 / bench["training_target_tokens_per_second"]
    collections = bench["collections"]
    heldout = sum(collections[name]["scored_tokens"] * c[f"{name}_documents"] / collections[name]["documents"]
                  for name in ("validation", "test"))
    evaluation = ((cycles + 1) * heldout + n * 1.02) / bench["evaluation_target_tokens_per_second"]
    mean_source = bench["sampled_source_tokens"] / max(1, bench["sampled_documents"])
    full_documents = n / mean_source + c["validation_documents"] + c["test_documents"]
    preparation = bench["preparation_seconds"] * max(1, (c["shuffle_buffer"] + full_documents) /
                                                       (c["shuffle_buffer"] + bench["sampled_documents"]))
    seconds = (training + evaluation + preparation) * 1.5 + 300 + 120 * (cycles + 1)
    return {"workflow": "fresh_real", "estimated_hours": seconds / 3600,
            "estimated_compute_and_running_storage_usd": seconds / 3600 * hourly_usd,
            "training_cycles": cycles, "source_training_tokens": n, "training_seconds": training,
            "evaluation_seconds": evaluation, "preparation_seconds": preparation,
            "method": "Measured masked training/evaluation, document/shuffle preparation scaling, 1.5x margin, 300s setup and 120s per checkpoint/backup."}


def run(root, validation=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "worker.lock"), timeout=0):
        _run(root, validation)


def _run(root, validation):
    import torch
    from transformers import AutoTokenizer
    from .. import engine
    from ..fresh_data import prepare_fresh, load_collection
    from ..hub import backup
    from ..pipeline import environment
    from ..supervisor import deadline
    c, resources = read_json(root / "config.json"), read_json(root / "resources.json")
    validate_config(c)
    frozen = read_json(root / "manifest.json")
    if frozen and frozen["config_hash"] != digest(c):
        raise RuntimeError("Fresh-real configuration changed")
    atomic_json(root / "status.json", {"state": "running", "stage": "preparation", "time": now(), "pid": os.getpid()})
    if not (root / "environment.json").exists():
        atomic_json(root / "environment.json", environment())
    tokenizer = AutoTokenizer.from_pretrained(c["model"], revision=c["model_revision"])
    data = prepare_fresh(root, c, tokenizer)
    atomic_json(root / "manifest.json", {"config_hash": digest(c), "data_manifest_sha256": sha256(root / "data/manifest.json"),
                                         "code_sha256": resources["code_sha256"]})
    val, val_masks, _ = load_collection(root, "validation")
    test, test_masks, _ = load_collection(root, "test")
    first, first_masks, _ = load_collection(root, "cycle_01")
    bench = benchmark(root, c, first, first_masks, data)
    total_steps = sum(math.ceil(data["collections"][f"cycle_{g:02d}"]["sequences"] / c["effective_batch"])
                      for g in range(1, c["generations"] + 1))
    seed_root = root / f"seed_{c['seed']}"
    base = seed_root / "gen_00"
    baseline = read_json(base / "metrics.json")
    if baseline is None:
        model = engine.load_model(c)
        model.save_pretrained(base / "checkpoint", safe_serialization=True)
        tokenizer.save_pretrained(base / "tokenizer")
        baseline = {"generation": 0, "weights_sha256": engine.weights_hash(model),
                    "validation": engine.evaluate(model, val, c, masks=val_masks),
                    "test": engine.evaluate(model, test, c, masks=test_masks),
                    "checkpoint_files": checkpoint_manifest(base / "checkpoint")}
        atomic_json(base / "metrics.json", baseline)
        del model
        gc.collect()
    verify_files(base / "checkpoint", baseline["checkpoint_files"])
    previous_hash = baseline["weights_sha256"]
    parent_checkpoint, parent_state = base / "checkpoint", None
    for cycle in range(1, c["generations"] + 1):
        stage = seed_root / f"gen_{cycle:02d}"
        tokens, masks, collection = load_collection(root, f"cycle_{cycle:02d}")
        if parent_state is not None:
            parent_record = read_json(parent_checkpoint.parent / "training.json")
            verify_files(parent_checkpoint.parent, parent_record["state_files"])
            verify_files(parent_checkpoint, parent_record["checkpoint_files"])
        parent_state_hash = sha256(parent_state) if parent_state else None
        if (stage / "stage_completion.json").exists():
            metrics = verify_cycle(root, cycle, c, previous_hash, parent_state_hash)
        else:
            estimate = (collection["scored_tokens"] / bench["training_target_tokens_per_second"] +
                        (collection["scored_tokens"] + int(val_masks[:, 1:].sum()) + int(test_masks[:, 1:].sum())) /
                        bench["evaluation_target_tokens_per_second"]) * 1.5 + 120
            if c["kind"] == "full" and estimate > deadline(resources) - now():
                atomic_json(root / "status.json", {"state": "budget_paused", "generation": cycle, "time": now()})
                backup(root)
                return
            if shutil.disk_usage(root).free < 3_000_000_000:
                raise RuntimeError("Insufficient free space for recoverable checkpoints")
            atomic_json(root / "status.json", {"state": "running", "generation": cycle, "stage": "training", "time": now()})
            lineage = {"initializer_weights_sha256": previous_hash, "optimizer_parent_sha256": parent_state_hash,
                       "data_manifest_sha256": sha256(root / "data" / f"cycle_{cycle:02d}/manifest.json")}
            training = engine.train(root, cycle, tokens, val, c, previous_hash, initial_checkpoint=parent_checkpoint,
                                    lineage=lineage, masks=masks, val_masks=val_masks,
                                    optimizer_checkpoint=parent_state, total_steps=total_steps, inject=validation and cycle == 2)
            model = engine.load_model(c, stage / "checkpoint")
            tokenizer.save_pretrained(stage / "tokenizer")
            metrics = {"generation": cycle, "training": training, "data": collection,
                       "training_evaluation": engine.evaluate(model, tokens, c, masks=masks),
                       "validation": engine.evaluate(model, val, c, masks=val_masks),
                       "test": engine.evaluate(model, test, c, stage / "test_sequences.json", masks=test_masks)}
            atomic_json(stage / "metrics.json", metrics)
            files = {p.relative_to(stage).as_posix(): sha256(p) for p in stage.rglob("*")
                     if p.is_file() and "resume" not in p.parts and p.name != "stage_completion.json" and not p.name.endswith(".partial")}
            atomic_json(stage / "stage_completion.json", {"files": files, "generation": cycle})
            (stage / "resume/epoch.pt").unlink(missing_ok=True)
            del model
            gc.collect()
            if engine.device() == "cuda":
                torch.cuda.empty_cache()
            event(root, "generation_complete", generation=cycle, validation_perplexity=metrics["validation"]["perplexity"])
        previous_hash = metrics["training"]["selected_weights_sha256"]
        parent_checkpoint, parent_state = stage / "checkpoint", stage / "optimizer_state/cycle.pt"
        backup(root)  # Repairs backup-only interruptions before any following cycle.
        if validation and cycle == 1 and not (root / f".injected_{cycle}").exists():
            (root / f".injected_{cycle}").touch()
            if cycle == 1:
                atomic_json(root / "validation_cycle1_snapshot.json", read_json(stage / "stage_completion.json")["files"])
            event(root, "fresh_real_validation_interruption", generation=cycle)
            raise engine.ValidationInterruption(f"Fresh-real acceptance restart after cycle {cycle}")
    atomic_json(root / "completion.json", {"complete": True, "workflow": "fresh_real", "cycles": c["generations"], "time": now()})
    atomic_json(root / "status.json", {"state": "finalizing", "time": now()})
    from ..report import report
    report(root, for_backup=True)
    try:
        backup(root)
        if validation:
            from .fresh_validation import validate_remote
            validate_remote(root)
        atomic_json(root / "status.json", {"state": "complete", "time": now()})
        backup(root)
    except BaseException:
        atomic_json(root / "completion.json", {"complete": False, "workflow": "fresh_real", "finalization_pending": True, "time": now()})
        raise
