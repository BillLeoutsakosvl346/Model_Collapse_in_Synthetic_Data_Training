from __future__ import annotations

import contextlib
import gc
import hashlib
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from .common import atomic_json, checkpoint_manifest, event, read_json, sha256, verify_files
from .data import save_array


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def autocast(c):
    if device() == "cuda" and c["precision"] == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Configured BF16 is unsupported; select a compatible GPU")
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def sync():
    if device() == "cuda":
        torch.cuda.synchronize()


def load_model(c, checkpoint=None):
    seed_all(c["seed"])
    kwargs = {"torch_dtype": torch.float32}
    if checkpoint is None:
        kwargs["revision"] = c["model_revision"]
    return AutoModelForCausalLM.from_pretrained(str(checkpoint or c["model"]), **kwargs).to(device())


def weights_hash(model):
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def shifted_nll(logits, ids):
    return F.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                           ids[:, 1:].reshape(-1), reduction="none").reshape(len(ids), -1)


def tensor(rows):
    return torch.tensor(np.array(rows, copy=True), dtype=torch.long, device=device())


@torch.inference_mode()
def evaluate(model, blocks, c, export=None):
    model.eval()
    losses = []
    sync()
    start = time.monotonic()
    for offset in range(0, len(blocks), c["eval_batch"]):
        ids = tensor(blocks[offset:offset + c["eval_batch"]])
        with autocast(c):
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        nll = shifted_nll(logits, ids)
        losses.extend(nll.double().sum(1).cpu().tolist())
    sync()
    scored = len(blocks) * (blocks.shape[1] - 1)
    total = math.fsum(losses)
    if not scored or not math.isfinite(total):
        raise RuntimeError("Empty evaluation or non-finite loss")
    seq_nll = np.asarray(losses) / (blocks.shape[1] - 1)
    result = {"nll_sum": total, "scored_tokens": scored, "loss": total / scored,
              "perplexity": math.exp(total / scored), "seconds": time.monotonic() - start,
              "mean_sequence_perplexity": float(np.exp(seq_nll).mean()),
              "std_sequence_perplexity": float(np.exp(seq_nll).std()), "context_tokens": blocks.shape[1]}
    if export:
        atomic_json(export, {"sequence_mean_nll": seq_nll.tolist(), "scored_tokens_per_sequence": blocks.shape[1] - 1})
    return result


def train(root, generation, blocks, val, c, expected_base_hash=None):
    dest = Path(root) / f"seed_{c['seed']}" / f"gen_{generation:02d}"
    dest.mkdir(parents=True, exist_ok=True)
    done = read_json(dest / "training.json")
    input_hash = hashlib.sha256(np.asarray(blocks, dtype=np.int32).tobytes()).hexdigest()
    if done:
        if done["input_tokens_sha256"] != input_hash or done["initial_weights_sha256"] != expected_base_hash:
            raise RuntimeError("Generation provenance changed")
        verify_files(dest / "checkpoint", done["checkpoint_files"])
        event(root, "training_reused", generation=generation, input_tokens_sha256=input_hash)
        return done
    model = load_model(c)
    initial = weights_hash(model)
    if expected_base_hash and initial != expected_base_hash:
        raise RuntimeError("Generation did not start from original pretrained weights")
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"], betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=c["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c["epochs"], eta_min=c["min_learning_rate"])
    resume = dest / "resume" / "epoch.pt"
    best, selected_epoch, first_epoch, history = math.inf, None, 0, []
    if resume.exists():
        state = torch.load(resume, map_location=device(), weights_only=False)
        if state["input_hash"] != input_hash:
            raise RuntimeError("Interrupted training input changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        first_epoch, best, selected_epoch, history = state["epoch"], state["best"], state["selected_epoch"], state["history"]
    seed_all(c["seed"])
    if device() == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    for epoch in range(first_epoch, c["epochs"]):
        # Epoch-local deterministic order/dropout makes epoch-boundary resume exact.
        seed_all(c["seed"] + epoch)
        order = np.random.default_rng(c["seed"] + epoch).permutation(len(blocks))
        model.train()
        total_loss, count = 0.0, 0
        lr = optimizer.param_groups[0]["lr"]
        sync()
        epoch_start = time.monotonic()
        for start_idx in range(0, len(order), c["effective_batch"]):
            window = order[start_idx:start_idx + c["effective_batch"]]
            optimizer.zero_grad(set_to_none=True)
            for j in range(0, len(window), c["microbatch"]):
                rows = window[j:j + c["microbatch"]]
                ids = tensor(blocks[rows])
                with autocast(c):
                    loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                (loss * (len(rows) / len(window))).backward()
                total_loss += float(loss.detach()) * len(rows)
                count += len(rows)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            if not torch.isfinite(norm):
                raise RuntimeError("Non-finite gradients")
            optimizer.step()
        sync()
        train_seconds = time.monotonic() - epoch_start
        val_metrics = evaluate(model, val, c)
        history.append({"epoch": epoch + 1, "learning_rate": lr, "train_loss": total_loss / count,
                        "validation_loss": val_metrics["loss"], "train_seconds": train_seconds,
                        "training_target_tokens_per_second": count * 63 / train_seconds})
        if val_metrics["loss"] < best:
            best, selected_epoch = val_metrics["loss"], epoch + 1
            model.save_pretrained(dest / "checkpoint", safe_serialization=True)
        scheduler.step()
        resume.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1, "best": best, "selected_epoch": selected_epoch, "history": history,
                    "input_hash": input_hash}, resume.with_suffix(".partial"))
        resume.with_suffix(".partial").replace(resume)
        atomic_json(dest / "training_history.json", history)
        event(root, "epoch_complete", generation=generation, **history[-1])
    del model, optimizer, scheduler
    gc.collect()
    if device() == "cuda":
        torch.cuda.empty_cache()
    selected = load_model(c, dest / "checkpoint")
    final_hash = weights_hash(selected)
    if initial == final_hash:
        raise RuntimeError("Training did not change any model weights")
    done = {"generation": generation, "initial_weights_sha256": initial, "selected_weights_sha256": final_hash,
            "input_tokens_sha256": input_hash, "training_blocks": len(blocks), "selected_epoch": selected_epoch,
            "best_validation_loss": best, "seconds": time.monotonic() - start,
            "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated() if device() == "cuda" else 0,
            "checkpoint_files": checkpoint_manifest(dest / "checkpoint")}
    atomic_json(dest / "training.json", done)
    # Retain one resume state until the entire generation is committed; the backup excludes it.
    del selected
    gc.collect()
    return done


class ValidationInterruption(Exception):
    pass


@torch.inference_mode()
def generate(root, generation, model, prompts, c, parent_hash, inject=False):
    dest = Path(root) / f"synthetic_for_gen_{generation + 1:02d}"
    dest.mkdir(parents=True, exist_ok=True)
    provenance = {"parent_weights_sha256": parent_hash, "prompt_tokens_sha256": hashlib.sha256(np.asarray(prompts).tobytes()).hexdigest(),
                  "prompt_count": len(prompts), "new_tokens": 64, "num_beams": 5, "repetition_penalty": 1.0}
    done = read_json(dest / "manifest.json")
    if done:
        if done["provenance"] != provenance:
            raise RuntimeError("Synthetic dataset provenance changed")
        verify_files(dest, done["files"])
        return np.load(dest / "tokens.npy", mmap_mode="r"), done
    model.eval()
    generated, shards = [], {}
    sync()
    start_time = time.monotonic()
    reused = 0
    for start in range(0, len(prompts), c["generation_shard"]):
        end = min(start + c["generation_shard"], len(prompts))
        path = dest / f"shard_{start:07d}.npy"
        meta_path = path.with_suffix(".json")
        meta = read_json(meta_path)
        if meta:
            if meta["provenance"] != provenance or meta["sha256"] != sha256(path):
                raise RuntimeError("Corrupt or mismatched generation shard")
            shard = np.load(path, allow_pickle=False)
            reused += len(shard)
        else:
            parts = []
            for offset in range(start, end, c["generation_batch"]):
                ids = tensor(prompts[offset:min(end, offset + c["generation_batch"])])
                with autocast(c):
                    outputs = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), num_beams=5,
                                             do_sample=False, min_new_tokens=64, max_new_tokens=64,
                                             repetition_penalty=1.0, no_repeat_ngram_size=0,
                                             forced_eos_token_id=None, use_cache=True)
                continuation = outputs[:, 64:].cpu().numpy()
                if outputs.shape[1] != 128 or continuation.shape[1] != 64:
                    raise RuntimeError("Generation length invariant failed")
                parts.append(continuation)
            shard = np.concatenate(parts)
            save_array(path, shard)
            atomic_json(meta_path, {"provenance": provenance, "sha256": sha256(path)})
        if shard.shape != (end - start, 64):
            raise RuntimeError("Generation shard shape mismatch")
        shards[path.name] = sha256(path)
        generated.append(shard)
        event(root, "generation_shard_complete", generation=generation, completed=end, total=len(prompts), reused=bool(meta))
        marker = Path(root) / ".injection_generation_consumed"
        if inject and not marker.exists():
            marker.touch()
            event(root, "validation_generation_interruption", generation=generation, completed=end)
            raise ValidationInterruption("Injected generation interruption after an atomic shard")
    values = np.concatenate(generated)
    save_array(dest / "tokens.npy", values)
    sync()
    elapsed = time.monotonic() - start_time
    done = {"provenance": provenance, "continuation_tokens_sha256": hashlib.sha256(values.astype(np.int32).tobytes()).hexdigest(),
            "files": {"tokens.npy": sha256(dest / "tokens.npy"), **shards},
            "seconds": elapsed, "reused_continuations": reused,
            "new_tokens_per_second": (len(prompts) - reused) * 64 / max(elapsed, 1e-9)}
    atomic_json(dest / "manifest.json", done)
    return values, done


def diversity(values):
    values = np.asarray(values)
    result = {}
    for n in (1, 2, 3, 4):
        grams = Counter(tuple(row[i:i + n]) for row in values.tolist() for i in range(len(row) - n + 1))
        total = sum(grams.values())
        if n < 4:
            result[f"distinct_{n}"] = len(grams) / total
        else:
            result["repeated_4gram_rate"] = (total - len(grams)) / total
            result["repeated_4gram_denominator"] = total
    result["duplicate_continuation_rate"] = 1 - len(set(map(tuple, values.tolist()))) / len(values)
    counts = np.asarray(list(Counter(values.flatten().tolist()).values()), dtype=float)
    p = counts / counts.sum()
    result["token_entropy_nats"] = float(-(p * np.log(p)).sum())
    result["probe_count"] = len(values)
    return result
