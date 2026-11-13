from __future__ import annotations

import contextlib
import gc
import hashlib
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from ..common import atomic_json, checkpoint_manifest, digest, event, read_json, sha256, verify_files
from ..data import save_array
from ..science.metrics import diversity  # Compatibility import for published scripts.
from ..workflow.optimization import carries_optimizer


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


def shifted_nll(logits, ids, masks=None):
    targets = ids[:, 1:]
    if masks is not None:
        targets = targets.masked_fill(~masks[:, 1:].bool(), -100)
    return F.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                           targets.reshape(-1), reduction="none").reshape(len(ids), -1)


def tensor(rows):
    return torch.tensor(np.array(rows, copy=True), dtype=torch.long, device=device())


@torch.inference_mode()
def evaluate(model, blocks, c, export=None, *, masks=None, loss_masks=None):
    model.eval()
    losses, counts = [], []
    sync()
    start = time.perf_counter()
    for offset in range(0, len(blocks), c["eval_batch"]):
        ids = tensor(blocks[offset:offset + c["eval_batch"]])
        attention = torch.ones_like(ids) if masks is None else tensor(masks[offset:offset + c["eval_batch"]])
        scoring = attention if loss_masks is None else tensor(loss_masks[offset:offset + c["eval_batch"]])
        if loss_masks is not None and (torch.any(scoring > attention) or torch.any((scoring != 0) & (scoring != 1))):
            raise ValueError("Loss mask must select visible binary target positions")
        with autocast(c):
            logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
        nll = shifted_nll(logits, ids, scoring if masks is not None or loss_masks is not None else None)
        losses.extend(nll.double().sum(1).cpu().tolist())
        counts.extend(scoring[:, 1:].sum(1).cpu().tolist())
    sync()
    scored = sum(counts)
    total = math.fsum(losses)
    if not scored or not math.isfinite(total):
        raise RuntimeError("Empty evaluation or non-finite loss")
    if any(count <= 0 for count in counts):
        raise RuntimeError("Evaluation sequence has no prediction targets")
    seq_nll = np.asarray(losses) / np.asarray(counts)
    result = {"nll_sum": total, "scored_tokens": scored, "loss": total / scored,
              "perplexity": math.exp(total / scored), "seconds": time.perf_counter() - start,
              "mean_sequence_perplexity": float(np.exp(seq_nll).mean()),
              "std_sequence_perplexity": float(np.exp(seq_nll).std()), "context_tokens": blocks.shape[1]}
    if export:
        atomic_json(export, {"sequence_mean_nll": seq_nll.tolist(),
                             "scored_tokens_per_sequence": blocks.shape[1] - 1 if masks is None and loss_masks is None else counts})
    return result


def train(root, generation, blocks, val, c, expected_base_hash=None, *, initial_checkpoint=None, lineage=None,
          masks=None, val_masks=None, optimizer_checkpoint=None, total_steps=None, inject=False,
          loss_masks=None, val_loss_masks=None, source_labels=None, update_probe=None, progress_callback=None):
    from ..workflow.selection import FINAL, selection_policy, training_contract, verify_selection
    policy, contract = selection_policy(c), training_contract(c)
    from ..workflow.optimization import is_plain_sgd, optimizer_and_scheduler, restore_optimizer
    sgd = is_plain_sgd(c)
    observing = sgd and c.get("update_diagnostics", {}).get("enabled", False)
    persistent = carries_optimizer(c)
    if persistent:
        if not total_steps or masks is None or val_masks is None:
            raise ValueError("Fresh-real training requires masks and a complete optimizer-step schedule")
    dest = Path(root) / f"seed_{c['seed']}" / f"gen_{generation:02d}"
    dest.mkdir(parents=True, exist_ok=True)
    done = read_json(dest / "training.json")
    input_hash = hashlib.sha256(np.asarray(blocks, dtype=np.int32).tobytes()).hexdigest()
    if masks is not None:
        input_hash = hashlib.sha256((input_hash + hashlib.sha256(np.asarray(masks, dtype=np.int32).tobytes()).hexdigest()).encode()).hexdigest()
    if loss_masks is not None:
        if (np.asarray(loss_masks).shape != np.asarray(blocks).shape or masks is None or
                np.any(np.asarray(loss_masks) > np.asarray(masks)) or
                np.any((np.asarray(loss_masks) != 0) & (np.asarray(loss_masks) != 1))):
            raise ValueError("Invalid training loss mask")
        input_hash = hashlib.sha256((input_hash + hashlib.sha256(np.asarray(loss_masks, dtype=np.int32).tobytes()).hexdigest()).encode()).hexdigest()
    if observing:
        if (source_labels is None or np.asarray(source_labels).shape != (len(blocks),) or
                not np.isin(source_labels, [0, 1]).all() or update_probe is None):
            raise ValueError("Update diagnostics require aligned source labels and a frozen probe")
        input_hash = hashlib.sha256((input_hash + hashlib.sha256(np.asarray(source_labels, dtype=np.uint8).tobytes()).hexdigest()
                                    + update_probe["identity"]).encode()).hexdigest()
    parent_state_hash = sha256(optimizer_checkpoint) if optimizer_checkpoint else None
    observation_binding = {"training_contract_sha256": contract, "input_tokens_sha256": input_hash,
                           "initial_weights_sha256": expected_base_hash, "optimizer_parent_sha256": parent_state_hash,
                           "lineage_sha256": digest(lineage), "total_optimizer_steps": total_steps} if observing else None
    if done:
        verify_selection(done, read_json(dest / "training_history.json", []), c)
        if done["input_tokens_sha256"] != input_hash or done["initial_weights_sha256"] != expected_base_hash:
            raise RuntimeError("Generation provenance changed")
        if lineage is not None and done.get("lineage") != lineage:
            raise RuntimeError("Generation generator/initializer lineage changed")
        verify_files(dest / "checkpoint", done["checkpoint_files"])
        if persistent:
            if (done.get("training_contract_sha256") != contract or done.get("optimizer_parent_sha256") != parent_state_hash
                    or done.get("total_optimizer_steps") != total_steps):
                raise RuntimeError("Persistent optimizer provenance changed")
            verify_files(dest, done["state_files"])
        if observing:
            from .sgd_effects import verify_seal
            verify_seal(dest, expected_weights=done["selected_weights_sha256"],
                        expected_steps=math.ceil(len(blocks) / c["effective_batch"]) * c["epochs"],
                        expected_probe_identity=update_probe["identity"], expected_binding=observation_binding)
        event(root, "training_reused", generation=generation, input_tokens_sha256=input_hash)
        return done
    if c.get("initialization") in ("continued", "reset_s0") and initial_checkpoint is None:
        raise RuntimeError("Continued training requires the preceding model checkpoint or fixed-S0 checkpoint")
    model = load_model(c, initial_checkpoint)
    initial = weights_hash(model)
    if expected_base_hash and initial != expected_base_hash:
        raise RuntimeError("Generation did not start from the expected initial weights")
    optimizer, scheduler = optimizer_and_scheduler(model, c, total_steps, persistent)
    global_step = 0
    if optimizer_checkpoint:
        parent = torch.load(optimizer_checkpoint, map_location=device(), weights_only=False)
        if parent["model_weights_sha256"] != initial or parent["total_steps"] != total_steps:
            raise RuntimeError("Optimizer state differs from its model or schedule")
        restore_optimizer(optimizer, scheduler, parent, c)
        global_step = parent["global_step"]
        restore_rng(parent["rng"])
    resume = dest / "resume" / "epoch.pt"
    best, best_epoch, selected_epoch, first_epoch, history = math.inf, None, None, 0, []
    if resume.exists() and observing:
        event(root, "sgd_unfinished_cycle_replayed", generation=generation)
    if resume.exists() and not observing:
        state = torch.load(resume, map_location=device(), weights_only=False)
        if state.get("training_contract_sha256") != contract:
            raise RuntimeError("Interrupted training selection policy or settings changed; legacy state requires its original runtime")
        if persistent and state.get("total_steps") != total_steps:
            raise RuntimeError("Interrupted optimizer schedule changed")
        if state["input_hash"] != input_hash:
            raise RuntimeError("Interrupted training input changed")
        if state.get("initial_hash", initial) != initial:
            raise RuntimeError("Interrupted training parent weights changed")
        model.load_state_dict(state["model"])
        restore_optimizer(optimizer, scheduler, state, c)
        first_epoch, best, selected_epoch, history = state["epoch"], state["best"], state["selected_epoch"], state["history"]
        if persistent:
            global_step = state["global_step"]
            restore_rng(state["rng"])
        best_epoch = state.get("best_epoch")
        if not 0 <= first_epoch <= c["epochs"] or len(history) != first_epoch:
            raise RuntimeError("Invalid epoch resume history")
    if not persistent:
        seed_all(c["seed"])
    if device() == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    observer = None
    if observing:
        from .sgd_effects import UpdateObserver, prepare_replay
        recovered = prepare_replay(dest, cycle=generation, probe_identity=update_probe["identity"],
                                   binding=observation_binding,
                                   expected_steps=range(global_step + 1, global_step + 1 +
                                                       math.ceil(len(blocks) / c["effective_batch"]) * c["epochs"]))
        if recovered:
            event(root, "sgd_uncommitted_seal_archived", generation=generation, **recovered)
        observer = UpdateObserver(model, probe=(update_probe["tokens"], update_probe["attention"], update_probe["scoring"]),
                                  root=dest, cycle=generation, probe_identity=update_probe["identity"],
                                  documents=update_probe["documents"], eval_batch=c["eval_batch"], binding=observation_binding)
    for epoch in range(first_epoch, c["epochs"]):
        # Epoch-local deterministic order/dropout makes epoch-boundary resume exact.
        if not persistent:
            seed_all(c["seed"] + epoch)
        order = np.random.default_rng(c["seed"] + epoch + (generation if persistent else 0)).permutation(len(blocks))
        model.train()
        total_loss, count = 0.0, 0
        lr = optimizer.param_groups[0]["lr"]
        sync()
        epoch_start = time.perf_counter()
        for start_idx in range(0, len(order), c["effective_batch"]):
            window_start = time.perf_counter()
            window = order[start_idx:start_idx + c["effective_batch"]]
            scoring_masks = loss_masks if loss_masks is not None else masks
            window_targets = int(np.asarray(scoring_masks[window])[:, 1:].sum()) if scoring_masks is not None else len(window) * (blocks.shape[1] - 1)
            if window_targets <= 0:
                raise ValueError("Training batch has no scored targets")
            optimizer.zero_grad(set_to_none=True)
            accumulator = None
            if observer is not None:
                from .sgd_effects import SourceAccumulator
                accumulator = SourceAccumulator(model, window_targets)
            for j in range(0, len(window), c["microbatch"]):
                rows = window[j:j + c["microbatch"]]
                ids = tensor(blocks[rows])
                attention = torch.ones_like(ids) if masks is None else tensor(masks[rows])
                scoring = attention if loss_masks is None else tensor(loss_masks[rows])
                labels = ids if masks is None and loss_masks is None else ids.masked_fill(~scoring.bool(), -100)
                targets = int(scoring[:, 1:].sum())
                with autocast(c):
                    if observer is not None:
                        logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
                        sequence_sums = shifted_nll(logits, ids, scoring).sum(1)
                        loss = sequence_sums.sum() / targets
                    else:
                        loss = model(input_ids=ids, attention_mask=attention, labels=labels, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                if observer is not None:
                    accumulator.backward(sequence_sums, scoring[:, 1:].sum(1), np.asarray(source_labels)[rows])
                else:
                    (loss * (targets / window_targets)).backward()
                total_loss += float(loss.detach()) * targets
                count += targets
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            if not torch.isfinite(norm):
                raise RuntimeError("Non-finite gradients")
            if observer is not None:
                observer.before_step(accumulator, learning_rate=optimizer.param_groups[0]["lr"],
                                     step=global_step + 1, batch_indices=window)
            optimizer.step()
            if observer is not None:
                observer.after_step(step_seconds=time.perf_counter() - window_start)
                del accumulator
            if progress_callback is not None:
                progress_callback(global_step + 1, generation=generation, examples=len(window))
            if persistent:
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                marker = Path(root) / ".fresh_training_exit_consumed"
                if inject and not marker.exists():
                    marker.touch()
                    event(root, "sgd_update_interruption" if sgd else "fresh_real_midcycle_interruption", generation=generation, global_step=global_step)
                    raise ValidationInterruption("Fresh-real acceptance exit inside an unfinished cycle")
        sync()
        train_seconds = time.perf_counter() - epoch_start
        if val_loss_masks is not None:
            val_metrics = evaluate(model, val, c, masks=val_masks, loss_masks=val_loss_masks)
        else:
            val_metrics = evaluate(model, val, c, masks=val_masks) if val_masks is not None else evaluate(model, val, c)
        history.append({"epoch": epoch + 1, "learning_rate": lr, "train_loss": total_loss / count,
                        "validation_loss": val_metrics["loss"], "train_seconds": train_seconds,
                        "training_target_tokens_per_second": count / train_seconds})
        if persistent:
            history[-1].update(global_step=global_step, learning_rate_end=optimizer.param_groups[0]["lr"], scored_training_tokens=count)
        if not math.isfinite(val_metrics["loss"]):
            raise RuntimeError("Non-finite validation loss")
        improved = val_metrics["loss"] < best
        if improved:
            best, best_epoch = val_metrics["loss"], epoch + 1
        if (policy != FINAL and improved) or (policy == FINAL and epoch + 1 == c["epochs"]):
            selected_epoch = epoch + 1
            model.save_pretrained(dest / "checkpoint", safe_serialization=True)
        if not persistent and scheduler is not None:
            scheduler.step()
        resume.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "epoch": epoch + 1, "best": best, "selected_epoch": selected_epoch, "history": history,
                    "input_hash": input_hash, "initial_hash": initial, "best_epoch": best_epoch,
                    "training_contract_sha256": contract, "global_step": global_step, "total_steps": total_steps,
                    "rng": capture_rng() if persistent else None,
                    **({"optimizer_kind": "sgd"} if sgd else {})}, resume.with_suffix(".partial"))
        resume.with_suffix(".partial").replace(resume)
        atomic_json(dest / "training_history.json", history)
        event(root, "epoch_complete", generation=generation, **history[-1])
    terminal_hash = weights_hash(model)
    if policy == FINAL:
        # Also repairs an interrupted final save using the verified terminal resume state.
        selected_epoch = c["epochs"]
        model.save_pretrained(dest / "checkpoint", safe_serialization=True)
    atomic_json(dest / "training_history.json", history)
    if persistent:
        state_path = dest / "optimizer_state" / "cycle.pt"
        state_path.parent.mkdir(exist_ok=True)
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict() if scheduler is not None else None, "rng": capture_rng(),
                    "global_step": global_step, "total_steps": total_steps, "model_weights_sha256": terminal_hash,
                    **({"optimizer_kind": "sgd"} if sgd else {})}, state_path.with_suffix(".partial"))
        state_path.with_suffix(".partial").replace(state_path)
    diagnostic_seal = observer.seal(terminal_hash) if observer is not None else None
    del observer
    del model, optimizer, scheduler
    gc.collect()
    if device() == "cuda":
        torch.cuda.empty_cache()
    selected = load_model(c, dest / "checkpoint")
    final_hash = weights_hash(selected)
    if policy == FINAL and final_hash != terminal_hash:
        raise RuntimeError("Reloaded final checkpoint differs from terminal weights")
    if initial == final_hash:
        raise RuntimeError("Training did not change any model weights")
    done = {"generation": generation, "initial_weights_sha256": initial, "selected_weights_sha256": final_hash,
            "input_tokens_sha256": input_hash, "training_blocks": len(blocks), "selected_epoch": selected_epoch,
            "best_validation_loss": best, "best_validation_epoch": best_epoch,
            "selected_validation_loss": history[selected_epoch - 1]["validation_loss"],
            "executed_epochs": len(history), "terminal_weights_sha256": terminal_hash,
            "training_contract_sha256": contract, "seconds": time.perf_counter() - start,
            "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated() if device() == "cuda" else 0,
            "checkpoint_files": checkpoint_manifest(dest / "checkpoint")}
    done.update(initialization=c.get("initialization", "fresh_base"), optimizer_initialization="fresh_each_cycle",
                checkpoint_selection=policy)
    if persistent:
        done.update(optimizer_initialization="carried_across_cycles" if optimizer_checkpoint else "fresh_pretrained_start",
                    optimizer_parent_sha256=parent_state_hash, global_step=global_step, total_optimizer_steps=total_steps,
                    state_files={"optimizer_state/cycle.pt": sha256(state_path)})
    if lineage is not None:
        done["lineage"] = lineage
    if diagnostic_seal is not None:
        done["update_diagnostics"] = diagnostic_seal
    if sgd:
        done.update(optimizer="sgd", learning_rate_policy="constant", momentum=0, weight_decay=0)
    atomic_json(dest / "training.json", done)
    # Retain one resume state until the entire generation is committed; the backup excludes it.
    del selected
    gc.collect()
    return done


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


class ValidationInterruption(Exception):
    pass


def decoding_settings(c):
    from .decoding import decoding_settings as resolve
    return resolve(c)


@contextlib.contextmanager
def generation_rng(c, generation, offset, provenance):
    """Sampling is reproducible per batch and cannot consume the training stream."""
    if c.get("generation_decoder") != "sampling":
        yield
        return
    from .decoding import sampling_seed
    state = capture_rng()
    try:
        torch.manual_seed(sampling_seed(c, generation, offset, provenance))
        yield
    finally:
        restore_rng(state)


@torch.inference_mode()
def generate(root, generation, model, prompts, c, parent_hash, inject=False):
    dest = Path(root) / f"synthetic_for_gen_{generation + 1:02d}"
    dest.mkdir(parents=True, exist_ok=True)
    settings = decoding_settings(c)
    prompt_length = 65 if c.get("workflow") in ("completion_comparison", "completion_arm") else 64
    provenance = {"parent_weights_sha256": parent_hash, "prompt_tokens_sha256": hashlib.sha256(np.asarray(prompts).tobytes()).hexdigest(),
                  "prompt_count": len(prompts), "new_tokens": settings["max_new_tokens"], "num_beams": settings["num_beams"],
                  "repetition_penalty": settings["repetition_penalty"]}
    if c.get("generation_decoder") is not None:
        from .decoding import SAMPLING_POLICY
        if c["generation_shard"] % c["generation_batch"]:
            raise ValueError("Explicit decoding requires generation_shard divisible by generation_batch")
        provenance.update(version=2, decoder=c["generation_decoder"], decoding=settings,
                          seed=c["seed"], generation=generation, purpose=c.get("generation_purpose", "training"),
                          sampling_policy=SAMPLING_POLICY if settings["do_sample"] else None,
                          generation_batch=c["generation_batch"], generation_shard=c["generation_shard"])
    if np.asarray(prompts).ndim != 2 or np.asarray(prompts).shape[1] != prompt_length or not len(prompts):
        raise ValueError("Generation requires nonempty 64-token prompts")
    done = read_json(dest / "manifest.json")
    if done:
        if done["provenance"] != provenance:
            raise RuntimeError("Synthetic dataset provenance changed")
        verify_files(dest, done["files"])
        return np.load(dest / "tokens.npy", mmap_mode="r"), done
    model.eval()
    generated, shards = [], {}
    sync()
    start_time = time.perf_counter()
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
                with generation_rng(c, generation, offset, provenance), autocast(c):
                    outputs = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **settings)
                continuation = outputs[:, prompt_length:].cpu().numpy()
                if outputs.shape[1] != prompt_length + 64 or continuation.shape[1] != 64:
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
    elapsed = time.perf_counter() - start_time
    done = {"provenance": provenance, "continuation_tokens_sha256": hashlib.sha256(values.astype(np.int32).tobytes()).hexdigest(),
            "files": {"tokens.npy": sha256(dest / "tokens.npy"), **shards},
            "seconds": elapsed, "reused_continuations": reused,
            "new_tokens_per_second": (len(prompts) - reused) * 64 / max(elapsed, 1e-9)}
    atomic_json(dest / "manifest.json", done)
    return values, done
