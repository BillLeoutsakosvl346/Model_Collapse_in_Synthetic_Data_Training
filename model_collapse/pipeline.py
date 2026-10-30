from __future__ import annotations

import gc
import importlib.metadata
import os
import platform
import subprocess
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from transformers import AutoTokenizer

from .common import atomic_json, digest, event, now, read_json, sha256
from .data import load_blocks, prepare
from .engine import (ValidationInterruption, autocast, decoding_settings, device, diversity, evaluate, generate,
                     load_model, seed_all, sync, tensor, train, weights_hash)
from .hub import backup
from .workflow.lineage import training_lineage, verify_completed_stage
from .analysis.forecasts import forecast


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
                model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **decoding_settings(c))
            sync()
            gen_times.append(time.perf_counter() - start)
    result = {"training_target_tokens_per_second": train_tps,
              "generation_new_tokens_per_second": c["generation_batch"] * 64 / float(np.median(gen_times[1:])),
              "training_step_seconds": times, "generation_batch_seconds": gen_times,
              "effective_batch": c["effective_batch"], "microbatch": c["microbatch"], "generation_batch": c["generation_batch"],
              "num_beams": c.get("num_beams", 5), "repetition_penalty": c.get("repetition_penalty", 1.0),
              "precision": c["precision"], "full_split_blocks": {k: v["full_blocks"] for k, v in stats.items()},
              "gpu": torch.cuda.get_device_name() if device() == "cuda" else "cpu"}
    atomic_json(dest, result)
    del model
    gc.collect()
    if device() == "cuda":
        torch.cuda.empty_cache()
    event(root, "benchmark_complete", **result)
    return result


def run(root, validation=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "worker.lock"), timeout=0):
        _run(root, validation)


def _run(root, validation):
    c = read_json(root / "config.json")
    continued = c.get("initialization") == "continued"
    branch = c.get("initialization") == "reset_s0"
    resources = read_json(root / "resources.json")
    frozen = read_json(root / "manifest.json")
    if frozen and frozen["config_hash"] != digest(c):
        raise RuntimeError("Cannot resume with changed experimental settings")
    seed_all(c["seed"])
    session_environment = environment()
    for key, expected_value in c.get("required_environment", {}).items():
        actual_value = session_environment.get(key, session_environment["packages"].get(key))
        if actual_value != expected_value:
            raise RuntimeError(f"Matched environment requirement failed for {key}: expected {expected_value}, got {actual_value}")
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
    if branch:
        from .workflow.branch import import_shared_prefix
        import_shared_prefix(root, c)
    baseline_path = root / f"seed_{c['seed']}" / "pretrained" / "metrics.json"
    baseline = read_json(baseline_path)
    if baseline is None and not branch:
        model = load_model(c)
        baseline = {"initial_weights_sha256": weights_hash(model), "validation": evaluate(model, val, c),
                    "test": evaluate(model, test, c, baseline_path.parent / "test_sequences.json")}
        atomic_json(baseline_path, baseline)
        del model
        gc.collect()
    base_hash = c["start_checkpoint"]["weights_sha256"] if branch else baseline["initial_weights_sha256"]
    first_new = 2 if branch else 0
    first_training = first_new if branch else 1
    for g in range(first_new, c["generations"] + 1):
        gen = root / f"seed_{c['seed']}" / f"gen_{g:02d}"
        previous = root / f"synthetic_for_gen_{g:02d}"
        blocks, expected, parent_checkpoint, lineage = training_lineage(root, g, c, base_hash)
        if g == 0:
            blocks = real
        def training_stage():
            if continued and g == 0:
                from .continued import initialize
                return initialize(root, c)
            return train(root, g, blocks, val, c, expected, initial_checkpoint=parent_checkpoint, lineage=lineage)
        metrics_path = gen / "metrics.json"
        if metrics_path.exists():
            training = training_stage()
            verify_completed_stage(root, g, c, training)
            event(root, "generation_reused", generation=g, metrics_sha256=sha256(metrics_path))
            continue
        if c["kind"] == "full":
            from .supervisor import deadline
            remaining = deadline(resources) - now()
            next_prompt_count = len(real) if g < c["generations"] else min(len(real), c["probe_count"])
            next_seconds = (c["epochs"] * len(real) * (c["block_size"] - 1) / bench["training_target_tokens_per_second"] +
                            next_prompt_count * c["new_tokens"] / bench["generation_new_tokens_per_second"]) * 1.5 + 240
            if next_seconds > remaining:
                atomic_json(root / "status.json", {"state": "budget_paused", "next_generation": g, "time": now()})
                event(root, "budget_stage_declined", generation=g, estimated_stage_seconds=next_seconds, remaining_seconds=remaining)
                if g:
                    from .report import report
                    report(root, for_backup=True)
                backup(root)
                return
        atomic_json(root / "status.json", {"state": "running", "generation": g, "stage": "training", "pid": os.getpid(), "time": now()})
        if resources.get("owner_pair_id") and shutil.disk_usage(root).free < 3_000_000_000:
            raise RuntimeError("Insufficient recoverable checkpoint space; last committed stages remain preserved")
        if not (gen / "config.json").exists():
            atomic_json(gen / "config.json", {"training_config": c, "environment": session_environment,
                        "launch_id": resources["launch_id"], "code_sha256": resources["code_sha256"]})
        training = training_stage()
        if g:
            parent = read_json(previous / "manifest.json")
            if training["input_tokens_sha256"] != parent["continuation_tokens_sha256"]:
                raise RuntimeError("Previous generation's continuations were not used exactly")
            if continued and parent["provenance"]["parent_weights_sha256"] != expected:
                raise RuntimeError("The model generating these data must also initialize their training")
        model = load_model(c, gen / "checkpoint")
        if g == 0 and c.get("initial_dataset"):
            from .workflow.initial_data import import_initial_dataset
            import_initial_dataset(root, c)
        tokenizer.save_pretrained(gen / "tokenizer")
        metrics = {"generation": g, "training": training, "validation": evaluate(model, val, c),
                   "test": evaluate(model, test, c, gen / "test_sequences.json")}
        prompts = real if g < c["generations"] else real[:c["probe_count"]]
        atomic_json(root / "status.json", {"state": "running", "generation": g, "stage": "generation", "pid": os.getpid(), "time": now()})
        values, generation = generate(root, g, model, prompts, c, training["selected_weights_sha256"],
                                      inject=validation and g == (1 if c.get("initial_dataset") else first_new))
        probe = values[:c["probe_count"]]
        metrics["diversity"] = diversity(probe)
        metrics["generation_timing"] = generation
        atomic_json(gen / "samples.json", [{"prompt_index": i, "prompt": tokenizer.decode(real[i]),
                                            "continuation": tokenizer.decode(probe[i])} for i in range(len(probe))])
        reference = model if g == 0 else load_model(c, root / f"seed_{c['seed']}" / "gen_00" / "checkpoint")
        metrics["frozen_g0_reference"] = evaluate(reference, probe, c, gen / "reference_sequences.json")
        atomic_json(metrics_path, metrics)
        stage_files = {p.relative_to(gen).as_posix(): sha256(p) for p in gen.rglob("*")
                       if p.is_file() and "resume" not in p.parts and p.name != "stage_completion.json"
                       and not p.name.endswith(".partial")}
        atomic_json(gen / "stage_completion.json", {"generation": g, "files": stage_files,
                                                     "selected_weights_sha256": training["selected_weights_sha256"]})
        resume_state = gen / "resume" / "epoch.pt"
        if resume_state.exists():
            resume_state.unlink()
        del model, reference
        gc.collect()
        if device() == "cuda":
            torch.cuda.empty_cache()
        event(root, "generation_complete", generation=g, validation_perplexity=metrics["validation"]["perplexity"],
              test_perplexity=metrics["test"]["perplexity"], initial_weights_sha256=training["initial_weights_sha256"],
              input_tokens_sha256=training["input_tokens_sha256"])
        backup(root)
        if validation and g == first_training and not (root / ".injection_g1_consumed").exists():
            snapshot = {str(p.relative_to(root)): sha256(p) for p in gen.rglob("*") if p.is_file() and "resume" not in p.parts}
            atomic_json(root / "validation_g1_snapshot.json", snapshot)
            (root / ".injection_g1_consumed").touch()
            event(root, "validation_training_interruption", completed_generation=g)
            raise ValidationInterruption(f"Injected process exit after completed cycle {g}")
    snapshot = read_json(root / "validation_g1_snapshot.json")
    if snapshot:
        from .common import verify_files
        verify_files(root, snapshot)
    from .report import report
    report(root, for_backup=True)
    if validation and (continued or branch):
        from .continued import validate_remote
        validate_remote(root)
    atomic_json(root / "status.json", {"state": "complete", "generations": c["generations"], "time": now()})
    atomic_json(root / "completion.json", {"complete": True, "generations": c["generations"], "time": now(),
                                            "validation_g1_unchanged": bool(snapshot),
                                            "validation_completed_cycle_unchanged": bool(snapshot),
                                            "first_new_cycle": first_new})
    backup(root)
    event(root, "pipeline_complete", generations=c["generations"])
