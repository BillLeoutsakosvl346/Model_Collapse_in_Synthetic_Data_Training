"""Bounded real-only learning-rate selection using the shared SGD engine."""
from __future__ import annotations

import copy
import gc
import hashlib
import inspect
import math
from pathlib import Path

import numpy as np

from ..common import atomic_json, digest, now, read_json, sha256, verify_files
from ..completion_data import assemble, load_pairs
from .sgd_profile import CALIBRATION

POLICY = "bounded_real_only_192_updates_v1"


class CalibrationBlocked(RuntimeError):
    pass


def _source_hash(path):
    # Match deployment identity: Windows checkout line endings are not a new algorithm.
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sample_indices(seed, cycle, population=10000):
    if population < CALIBRATION["sample_pairs"]:
        raise ValueError("Calibration source has insufficient examples")
    return np.random.default_rng(np.random.SeedSequence(
        [seed, cycle, CALIBRATION["selection_seed_domain"]])).permutation(population)[:CALIBRATION["sample_pairs"]]


def eligible(result):
    fields = ("training_nll", "baseline_training_nll", "validation_nll", "baseline_validation_nll")
    if not all(isinstance(result.get(k), (int, float)) and math.isfinite(result[k]) for k in fields):
        return False
    return (result.get("weights_changed") is True
            and result["baseline_training_nll"] - result["training_nll"] > CALIBRATION["minimum_train_nll_decrease"]
            and result["validation_nll"] - result["baseline_validation_nll"] <= math.log(CALIBRATION["validation_perplexity_ratio"]))


def rank_candidates(rows):
    """Choose lowest held-out NLL; near ties prefer the smaller rate."""
    remaining = [r for r in rows if r.get("eligible") is True]
    ranked = []
    while remaining:
        minimum = min(r["validation_nll"] for r in remaining)
        near = [r for r in remaining if r["validation_nll"] <= minimum + CALIBRATION["tie_nll"]]
        chosen = min(near, key=lambda r: r["learning_rate"])
        ranked.append(chosen)
        remaining = [r for r in remaining if r is not chosen]
    return ranked


def calibration_contract(c):
    from ..paths import PROJECT
    return digest({"policy": POLICY, "settings": CALIBRATION,
                   "model": c["model"], "model_revision": c["model_revision"],
                   "bank_manifest_sha256": c["prepared_bank"]["manifest_sha256"],
                   "seed": c["seed"], "precision": "fp32", "microbatch": 32, "effective_batch": 128,
                   "optimizer": "sgd", "momentum": 0, "weight_decay": 0, "scheduler": None,
                   "implementation": _source_hash(__file__),
                   "engine_implementation": _source_hash(PROJECT / "model_collapse/science/engine.py"),
                   "optimization_policy_implementation": _source_hash(PROJECT / "model_collapse/workflow/optimization.py")})


def _freeze(path, value):
    previous = read_json(path)
    if previous is not None and previous != value:
        raise RuntimeError("Calibration identity changed: " + Path(path).name)
    if previous is None:
        atomic_json(path, value)


def _arrays(bank, split, rows=None):
    prompt, target, _ = load_pairs(bank, split)
    if rows is not None:
        prompt, target = prompt[rows], target[rows]
    return prompt, target


def calibrate(root, c, bank, *, progress=None):
    """Return a frozen receipt; no test split or synthetic targets are read.

    Candidate tensors are temporary recovery state, deliberately hidden from normal
    artifact backup. Tables, selections, checkpoints' hashes and the final receipt
    remain in the visible calibration directory.
    """
    import torch
    from ..science import engine
    from ..science.gradients import stratified_indices
    root, bank = Path(root), Path(bank)
    output = root / "calibration"
    output.mkdir(parents=True, exist_ok=True)
    work = output / ".work"
    work.mkdir(exist_ok=True)
    contract = calibration_contract(c)
    _freeze(output / "contract.json", {"contract_sha256": contract, "policy": POLICY, "settings": CALIBRATION})
    resources = read_json(root / "resources.json", {})

    def guard(**details):
        if resources:
            from ..supervisor import deadline
            if now() >= deadline(resources) - 90:
                raise CalibrationBlocked("Calibration exhausted the combined setup deadline")
        if progress:
            progress({"stage": "learning_rate_calibration", **details})

    completed = read_json(output / "selection.json")
    if completed:
        if completed.get("contract_sha256") != contract or not completed.get("complete"):
            raise RuntimeError("Calibration receipt is incomplete or incompatible")
        verify_files(output, completed["files"])
        return completed

    guard(phase="preparing")
    manifest = read_json(bank / "manifest.json")
    if sha256(bank / "manifest.json") != c["prepared_bank"]["manifest_sha256"]:
        raise RuntimeError("Calibration bank manifest differs")
    bos = manifest["bos_token_id"]
    validation_prompt, validation_target = _arrays(bank, "validation")
    windows = np.load(bank / "validation/windows.npy", allow_pickle=False)
    probe = stratified_indices(windows, CALIBRATION["validation_probe_pairs"], CALIBRATION["validation_seed"])
    validation = assemble(validation_prompt, validation_target, bos)
    validation_probe = tuple(a[probe] for a in validation)
    selections, chunks = {}, {}
    for cycle in (1, 2):
        relative = f"cycle_{cycle:02d}"
        for name in ("prompts.npy", "targets.npy", "windows.npy"):
            verify_files(bank / relative, {name: manifest["collections"][relative]["files"][name]})
        prompt, target = _arrays(bank, relative)
        indices = sample_indices(c["seed"], cycle, len(prompt))
        selection = {"source_collection": relative, "source_manifest_sha256": sha256(bank / relative / "manifest.json"),
                     "indices": indices.tolist(), "seed_sequence": [c["seed"], cycle, CALIBRATION["selection_seed_domain"]]}
        _freeze(output / f"selection_cycle_{cycle:02d}.json", selection)
        selections[cycle] = selection
        chunks[cycle] = assemble(prompt[indices], target[indices], bos)
    _freeze(output / "validation_probe.json", {"indices": probe.tolist(), "selection_seed": CALIBRATION["validation_seed"],
                "documents": int(len(np.unique(windows[probe, 0]))), "source": "validation", "test_used": False})

    baseline_path = output / "baseline.json"
    baseline = read_json(baseline_path)
    original_checkpoint = work / "pretrained/checkpoint"
    if baseline is None:
        model = engine.load_model(c)
        baseline = {"weights_sha256": engine.weights_hash(model), "model_revision": c["model_revision"],
                    "validation_probe": engine.evaluate(model, validation_probe[0], c, masks=validation_probe[1], loss_masks=validation_probe[2]),
                    "validation_full": engine.evaluate(model, validation[0], c, masks=validation[1], loss_masks=validation[2]),
                    "training_probes": {}}
        n = CALIBRATION["training_probe_pairs"]
        baseline["training_probes"]["1"] = engine.evaluate(model, chunks[1][0][:n], c,
            masks=chunks[1][1][:n], loss_masks=chunks[1][2][:n])
        original_checkpoint.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(original_checkpoint, safe_serialization=True)
        from ..common import checkpoint_manifest
        baseline["checkpoint_files"] = checkpoint_manifest(original_checkpoint)
        atomic_json(baseline_path, baseline)
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    else:
        verify_files(original_checkpoint, baseline["checkpoint_files"])

    budget_path = output / "update_reservations.json"
    reservations = read_json(budget_path, {})
    physical_path = output / "physical_updates.json"
    physical = read_json(physical_path, {"recorded_updates": 0, "events": []})
    candidate_index_path = output / "candidate_index.json"
    candidate_index = read_json(candidate_index_path, {})
    result_paths = []

    def run_candidate(rate, phase):
        cycle = 1 if phase == "screen" else 2
        identifier = "lr_" + format(rate, ".0e").replace("-", "m").replace("+", "p")
        candidate = work / identifier
        result_path = output / f"{identifier}_{phase}.json"
        result_paths.append(result_path)
        previous = read_json(result_path)
        if previous is not None:
            if (previous.get("contract_sha256") != contract or previous.get("learning_rate") != rate
                    or previous.get("phase") != phase):
                raise RuntimeError("Candidate calibration result changed contract")
            if candidate_index.get(result_path.name) != sha256(result_path):
                raise RuntimeError("Candidate calibration result is corrupt or unsealed")
            return previous
        guard(phase=phase, learning_rate=rate)
        stage = candidate / f"seed_42/gen_{cycle:02d}"
        slot = identifier + "_" + phase
        if slot in reservations:
            # An unfinished epoch is never silently repeated beyond the 192-step cap.
            finished = (stage / "training.json").exists()
            if not finished:
                resume = stage / "resume/epoch.pt"
                finished = resume.exists() and torch.load(resume, map_location="cpu", weights_only=False).get("epoch") == 1
            if not finished:
                raise CalibrationBlocked("Interrupted calibration phase cannot be replayed within its reserved update cap")
        else:
            if sum(reservations.values()) + 32 > CALIBRATION["max_updates"]:
                raise CalibrationBlocked("Calibration reached its 192-update ceiling")
            reservations[slot] = 32
            atomic_json(budget_path, reservations)
        cfg = copy.deepcopy(c)
        cfg.update(workflow="completion_arm", arm_id="real", data_kind="real", synthetic_fraction=0.,
                   purpose="sgd_calibration", learning_rate=rate, min_learning_rate=rate,
                   calibration_required=True, calibration_reference=None, train_pairs_per_cycle=4096,
                   generations=2, update_diagnostics={"enabled": False})
        if cycle == 1:
            checkpoint, optimizer_state, expected = original_checkpoint, None, baseline["weights_sha256"]
        else:
            previous_stage = candidate / "seed_42/gen_01"
            checkpoint, optimizer_state = previous_stage / "checkpoint", previous_stage / "optimizer_state/cycle.pt"
            parent_training = read_json(previous_stage / "training.json")
            verify_files(checkpoint, parent_training["checkpoint_files"])
            verify_files(previous_stage, parent_training["state_files"])
            expected = parent_training["selected_weights_sha256"]
        inputs = chunks[cycle]
        extra = {}
        if "progress_callback" in inspect.signature(engine.train).parameters:
            def updated(step, **details):
                physical["recorded_updates"] += 1
                physical["events"].append({"candidate": identifier, "phase": phase, "global_step": step})
                atomic_json(physical_path, physical)
                if physical["recorded_updates"] > CALIBRATION["max_updates"]:
                    raise CalibrationBlocked("Calibration exceeded its physical update ceiling")
                guard(**{**details, "phase": phase, "learning_rate": rate, "update": step})
            extra["progress_callback"] = updated
        try:
            training = engine.train(candidate, cycle, inputs[0], validation_probe[0], cfg, expected,
                initial_checkpoint=checkpoint, optimizer_checkpoint=optimizer_state, total_steps=64,
                masks=inputs[1], loss_masks=inputs[2], val_masks=validation_probe[1], val_loss_masks=validation_probe[2],
                lineage={"purpose": "real_only_lr_calibration", "source_collection": selections[cycle]["source_collection"],
                         "selection_sha256": sha256(output / f"selection_cycle_{cycle:02d}.json"),
                         "source_bank_sha256": c["prepared_bank"]["manifest_sha256"]}, **extra)
            training = training or read_json(stage / "training.json")
            state = torch.load(stage / "optimizer_state/cycle.pt", map_location="cpu", weights_only=False)
            history = read_json(stage / "training_history.json")
            if (state["optimizer"]["state"] or state["scheduler"] is not None
                    or state["global_step"] != cycle * 32
                    or any(group["lr"] != rate for group in state["optimizer"]["param_groups"])
                    or any(row["learning_rate"] != rate or row["learning_rate_end"] != rate for row in history)):
                raise RuntimeError("Calibration did not use stateless, constant-rate SGD")
            model = engine.load_model(cfg, stage / "checkpoint")
            n = CALIBRATION["training_probe_pairs"]
            train_score = engine.evaluate(model, chunks[1][0][:n], cfg, masks=chunks[1][1][:n], loss_masks=chunks[1][2][:n])
            heldout = validation_probe if cycle == 1 else validation
            validation_score = engine.evaluate(model, heldout[0], cfg, masks=heldout[1], loss_masks=heldout[2])
            result = {"contract_sha256": contract, "learning_rate": rate, "phase": phase, "cycle": cycle,
                      "global_step": state["global_step"], "weights_sha256": training["selected_weights_sha256"],
                      "initializer_weights_sha256": expected,
                      "weights_changed": training["selected_weights_sha256"] != expected,
                      "training_nll": train_score["loss"],
                      "baseline_training_nll": baseline["training_probes"]["1"]["loss"],
                      "training_probe_source": "fixed_chunk_1_selection_first_512",
                      "validation_nll": validation_score["loss"],
                      "baseline_validation_nll": baseline["validation_probe" if cycle == 1 else "validation_full"]["loss"],
                      "validation_scope": "1024_pair_probe" if cycle == 1 else "complete_validation",
                      "training": training, "training_probe": train_score, "validation": validation_score,
                      "test_used": False}
            result["eligible"] = eligible(result)
            del model, state
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except (FloatingPointError, OverflowError) as error:
            result = {"contract_sha256": contract, "learning_rate": rate, "phase": phase,
                      "eligible": False, "failure": str(error), "failure_kind": "numerical_instability", "test_used": False}
        except RuntimeError as error:
            if not (str(error).lower().startswith("non-finite") or str(error).startswith("Training did not change")):
                raise
            result = {"contract_sha256": contract, "learning_rate": rate, "phase": phase,
                      "eligible": False, "failure": str(error), "failure_kind": "numerical_instability_or_no_learning", "test_used": False}
        atomic_json(result_path, result)
        candidate_index[result_path.name] = sha256(result_path)
        atomic_json(candidate_index_path, candidate_index)
        guard(phase=phase + "_completed", learning_rate=rate)
        return result

    screens = [run_candidate(rate, "screen") for rate in CALIBRATION["rates"]]
    finalists = rank_candidates(screens)[:2]
    confirmations = [run_candidate(item["learning_rate"], "confirm") for item in finalists]
    ranked = rank_candidates(confirmations)
    if not ranked:
        atomic_json(output / "blocked.json", {"contract_sha256": contract,
            "reason": "No candidate passed the bounded real-only calibration criteria", "screen": screens,
            "confirm": confirmations, "reserved_update_upper_bound": sum(reservations.values()), "test_used": False})
        raise CalibrationBlocked("No candidate passed the bounded real-only calibration; full runs remain unready")
    receipt = {"complete": True, "selection_policy": POLICY, "contract_sha256": contract,
               "selected_learning_rate": ranked[0]["learning_rate"], "tested_rates": CALIBRATION["rates"],
               "screen": screens, "confirm": confirmations, "reserved_update_upper_bound": sum(reservations.values()),
               "recorded_physical_updates": physical["recorded_updates"],
               "max_updates": CALIBRATION["max_updates"], "test_used": False,
               "production_initializer": "original_pretrained_M0_not_a_calibration_checkpoint",
               "pretrained_weights_sha256": baseline["weights_sha256"],
               "boundary_winner": ranked[0]["learning_rate"] in (min(CALIBRATION["rates"]), max(CALIBRATION["rates"])),
               "interpretation": "Best tested rate under this short real-only screen; not a proven optimum or ten-cycle stability guarantee.",
               "files": {p.name: sha256(p) for p in [output / "contract.json", baseline_path, budget_path, candidate_index_path,
                         output / "selection_cycle_01.json", output / "selection_cycle_02.json",
                         output / "validation_probe.json", *([physical_path] if physical_path.exists() else []), *result_paths]}, "time": now()}
    atomic_json(output / "selection.json", receipt)
    guard(phase="selected", learning_rate=receipt["selected_learning_rate"])
    return receipt
