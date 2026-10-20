from __future__ import annotations

import gc
import importlib.metadata
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from transformers import AutoTokenizer

from .common import atomic_json, digest, event, now, read_json, sha256
from .data import load_blocks, prepare
from .engine import (ValidationInterruption, autocast, device, diversity, evaluate, generate,
                     load_model, seed_all, sync, tensor, train, weights_hash)
from .hub import backup


def environment():
    driver = None
    if torch.cuda.is_available():
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "driver_version": driver,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "packages": {x: importlib.metadata.version(x) for x in ("transformers", "datasets", "numpy", "huggingface-hub", "matplotlib")},
            "determinism": "Fixed seeds and epoch-local shuffle; TF32 disabled; no promise of bitwise equality across GPU/library versions."}


def benchmark(root, c, prompts, stats):
    dest = Path(root) / "benchmark.json"
    if not (Path(root) / "benchmark_resources.json").exists():
        atomic_json(Path(root) / "benchmark_resources.json", read_json(Path(root) / "resources.json"))
    if dest.exists():
        return read_json(dest)
    model = load_model(c)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"])
    times = []
    for step in range(5):
        optimizer.zero_grad(set_to_none=True)
        sync()
        start = time.perf_counter()
        for offset in range(0, c["effective_batch"], c["microbatch"]):
            rows = np.arange(offset, offset + c["microbatch"]) % len(prompts)
            ids = tensor(prompts[rows])
            with autocast(c):
                loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids, use_cache=False).loss
            (loss / (c["effective_batch"] / c["microbatch"])).backward()
        optimizer.step()
        sync()
        times.append(time.perf_counter() - start)
    train_tps = c["effective_batch"] * 63 / float(np.median(times[2:]))
    del optimizer
    gc.collect()
    model.eval()
    gen_times = []
    with torch.inference_mode():
        for step in range(3):
            ids = tensor(prompts[:c["generation_batch"]])
            sync()
            start = time.perf_counter()
            with autocast(c):
                model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), num_beams=5, do_sample=False,
                               min_new_tokens=64, max_new_tokens=64, repetition_penalty=1.0,
                               forced_eos_token_id=None, use_cache=True)
            sync()
            gen_times.append(time.perf_counter() - start)
    result = {"training_target_tokens_per_second": train_tps,
              "generation_new_tokens_per_second": c["generation_batch"] * 64 / float(np.median(gen_times[1:])),
              "training_step_seconds": times, "generation_batch_seconds": gen_times,
              "effective_batch": c["effective_batch"], "microbatch": c["microbatch"], "generation_batch": c["generation_batch"],
              "num_beams": 5, "precision": c["precision"], "full_split_blocks": {k: v["full_blocks"] for k, v in stats.items()},
              "gpu": torch.cuda.get_device_name() if device() == "cuda" else "cpu"}
    atomic_json(dest, result)
    del model
    gc.collect()
    if device() == "cuda":
        torch.cuda.empty_cache()
    event(root, "benchmark_complete", **result)
    return result


def forecast(bench, generations, hourly_usd):
    n = bench["full_split_blocks"]["train"]
    # Includes 1.5x compute margin, epoch validation/test estimates, checkpoint saves/uploads, and setup.
    train_seconds = (generations + 1) * 5 * n * 63 / bench["training_target_tokens_per_second"]
    gen_seconds = (generations * n + 256) * 64 / bench["generation_new_tokens_per_second"]
    evaluation_seconds = (generations + 2) * (6 * bench["full_split_blocks"]["validation"] + bench["full_split_blocks"]["test"]) * 63 / bench["training_target_tokens_per_second"]
    seconds = (train_seconds + gen_seconds + evaluation_seconds) * 1.5 + 300 + 180 * (generations + 1)
    return {"synthetic_generations": generations, "estimated_hours": seconds / 3600,
            "estimated_compute_and_running_storage_usd": seconds / 3600 * hourly_usd,
            "method": "Warmed real training/five-beam generation; 1.5x compute margin plus setup/checkpoint/upload allowance. Estimate, not a measured full experiment."}


def run(root, validation=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "worker.lock"), timeout=0):
        _run(root, validation)


def _run(root, validation):
    c = read_json(root / "config.json")
    resources = read_json(root / "resources.json")
    frozen = read_json(root / "manifest.json")
    if frozen and frozen["config_hash"] != digest(c):
        raise RuntimeError("Cannot resume with changed experimental settings")
    seed_all(c["seed"])
    session_environment = environment()
    if not (root / "environment.json").exists():
        atomic_json(root / "environment.json", session_environment)
    atomic_json(root / ("environment_" + resources["launch_id"] + ".json"), session_environment)
    atomic_json(root / "status.json", {"state": "running", "pid": os.getpid(), "time": now()})
    tokenizer = AutoTokenizer.from_pretrained(c["model"], revision=c["model_revision"])
    data_manifest = prepare(root, c, tokenizer)
    real, val, test = (load_blocks(root, s) for s in ("train", "validation", "test"))
    atomic_json(root / "manifest.json", {"config_hash": digest(c), "data_manifest_sha256": sha256(root / "data" / "manifest.json"),
                                         "code_sha256": resources["code_sha256"], "kind": c["kind"]})
    bench = benchmark(root, c, real, data_manifest["stats"])
    baseline_path = root / f"seed_{c['seed']}" / "pretrained" / "metrics.json"
    baseline = read_json(baseline_path)
    if baseline is None:
        model = load_model(c)
        baseline = {"initial_weights_sha256": weights_hash(model), "validation": evaluate(model, val, c),
                    "test": evaluate(model, test, c, baseline_path.parent / "test_sequences.json")}
        atomic_json(baseline_path, baseline)
        del model
        gc.collect()
    expected = baseline["initial_weights_sha256"]
    for g in range(c["generations"] + 1):
        gen = root / f"seed_{c['seed']}" / f"gen_{g:02d}"
        previous = root / f"synthetic_for_gen_{g:02d}"
        blocks = real if g == 0 else np.load(previous / "tokens.npy", mmap_mode="r")
        metrics_path = gen / "metrics.json"
        if metrics_path.exists():
            training = train(root, g, blocks, val, c, expected)
            event(root, "generation_reused", generation=g, metrics_sha256=sha256(metrics_path))
            continue
        if c["kind"] == "full":
            from .supervisor import deadline
            remaining = deadline(resources) - now()
            next_seconds = (5 * len(real) * 63 / bench["training_target_tokens_per_second"] +
                            len(real) * 64 / bench["generation_new_tokens_per_second"]) * 1.5 + 240
            if next_seconds > remaining:
                atomic_json(root / "status.json", {"state": "budget_paused", "next_generation": g, "time": now()})
                event(root, "budget_stage_declined", generation=g, estimated_stage_seconds=next_seconds, remaining_seconds=remaining)
                if g:
                    from .report import report
                    report(root)
                backup(root)
                return
        atomic_json(root / "status.json", {"state": "running", "generation": g, "stage": "training", "pid": os.getpid(), "time": now()})
        if not (gen / "config.json").exists():
            atomic_json(gen / "config.json", {"training_config": c, "environment": session_environment,
                        "launch_id": resources["launch_id"], "code_sha256": resources["code_sha256"]})
        training = train(root, g, blocks, val, c, expected)
        if g:
            parent = read_json(previous / "manifest.json")
            if training["input_tokens_sha256"] != parent["continuation_tokens_sha256"]:
                raise RuntimeError("Previous generation's continuations were not used exactly")
        model = load_model(c, gen / "checkpoint")
        tokenizer.save_pretrained(gen / "tokenizer")
        metrics = {"generation": g, "training": training, "validation": evaluate(model, val, c),
                   "test": evaluate(model, test, c, gen / "test_sequences.json")}
        prompts = real if g < c["generations"] else real[:c["probe_count"]]
        atomic_json(root / "status.json", {"state": "running", "generation": g, "stage": "generation", "pid": os.getpid(), "time": now()})
        values, generation = generate(root, g, model, prompts, c, training["selected_weights_sha256"], inject=validation and g == 0)
        probe = values[:c["probe_count"]]
        metrics["diversity"] = diversity(probe)
        metrics["generation_timing"] = generation
        atomic_json(gen / "samples.json", [{"prompt_index": i, "prompt": tokenizer.decode(real[i]),
                                            "continuation": tokenizer.decode(probe[i])} for i in range(len(probe))])
        reference = model if g == 0 else load_model(c, root / f"seed_{c['seed']}" / "gen_00" / "checkpoint")
        metrics["frozen_g0_reference"] = evaluate(reference, probe, c, gen / "reference_sequences.json")
        atomic_json(metrics_path, metrics)
        resume_state = gen / "resume" / "epoch.pt"
        if resume_state.exists():
            resume_state.unlink()
        del model, reference
        gc.collect()
        if device() == "cuda":
            torch.cuda.empty_cache()
        event(root, "generation_complete", generation=g, validation_perplexity=metrics["validation"]["perplexity"],
              test_perplexity=metrics["test"]["perplexity"], initial_weights_sha256=expected,
              input_tokens_sha256=training["input_tokens_sha256"])
        backup(root)
        if validation and g == 1 and not (root / ".injection_g1_consumed").exists():
            snapshot = {str(p.relative_to(root)): sha256(p) for p in gen.rglob("*") if p.is_file() and "resume" not in p.parts}
            atomic_json(root / "validation_g1_snapshot.json", snapshot)
            (root / ".injection_g1_consumed").touch()
            event(root, "validation_training_interruption", completed_generation=1)
            raise ValidationInterruption("Injected process exit after completed G1")
    snapshot = read_json(root / "validation_g1_snapshot.json")
    if snapshot:
        from .common import verify_files
        verify_files(root, snapshot)
    from .report import report
    report(root)
    atomic_json(root / "status.json", {"state": "complete", "generations": c["generations"], "time": now()})
    atomic_json(root / "completion.json", {"complete": True, "generations": c["generations"], "time": now(),
                                            "validation_g1_unchanged": bool(snapshot)})
    backup(root)
    event(root, "pipeline_complete", generations=c["generations"])
