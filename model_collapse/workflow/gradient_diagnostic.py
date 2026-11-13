"""A bounded checkpoint diagnostic; never launches recursive training."""
from __future__ import annotations

import gc
import math
import os
from pathlib import Path

import numpy as np

from ..common import atomic_json, digest, event, now, read_json, sha256

ALLOCATION = "smollm2_gradient_diagnostic"


def validate_config(c):
    if (c.get("workflow") != "gradient_diagnostic" or c.get("arms") != ["mixed99", "synthetic"]
            or c.get("cycles") != [1, 5, 10] or c.get("probe_pairs") != 1024
            or c.get("common_learning_rate") != 3e-6 or c.get("precision") != "fp32"):
        raise ValueError("Unsupported gradient diagnostic protocol")
    if c.get("source_parent", {}).get("run_id") != "20261111T210854Z-full-completion64-mixtures":
        raise ValueError("Gradient diagnostic source comparison differs")
    expected_seconds = 5400 if c.get("phase") == "layer_norms" else 3600
    if c.get("microbatch") != 8 or c.get("max_compute_seconds") != expected_seconds:
        raise ValueError("Gradient diagnostic batch/deadline differs")
    if c.get("required_gpu") not in ("NVIDIA RTX PRO 4000 Blackwell", "NVIDIA GeForce RTX 5090", "NVIDIA A40"):
        raise ValueError("Diagnostic GPU must be explicitly authorized and accepted")
    ref = c["source_parent"]
    if set(ref.get("revisions", {})) != {"model", "dataset"} or any(len(s) != 40 for s in ref["revisions"].values()):
        raise ValueError("Diagnostic requires pinned source commits")
    if len(ref.get("artifact_index_sha256", "")) != 64 or len(c.get("source_config_sha256", "")) != 64:
        raise ValueError("Diagnostic requires source hashes")
    return c


def _accept(model, case, validation_arrays, root, resources):
    import torch
    from ..science.engine import weights_hash
    from ..science.gradients import mean_gradient, combine, relative_error, apply_gradient, sequence_losses
    real = np.setdiff1d(np.arange(len(case["tokens"])), case["synthetic_rows"])
    rows = np.r_[real[:8], case["synthetic_rows"][:8]]
    arrays = tuple(case[n][rows] for n in ("tokens", "attention", "scoring"))
    before_hash = weights_hash(model)
    saved = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
    gr, _ = mean_gradient(model, *arrays, rows=np.arange(8))
    gs, _ = mean_gradient(model, *arrays, rows=np.arange(8, 16))
    direct, _ = mean_gradient(model, *arrays)
    error = relative_error(direct, combine(gr, gs, .5))
    miniature_validation = tuple(a[:16] for a in validation_arrays)
    before, counts = sequence_losses(model, *miniature_validation)
    apply_gradient(model, direct, None, 3e-6)
    changed = weights_hash(model) != before_hash
    after, counts2 = sequence_losses(model, *miniature_validation)
    model.load_state_dict(saved)
    checks = {"finite_conditional_loss": bool(np.isfinite(before).all() and np.isfinite(after).all()),
              "exactly_64_scored_targets": bool(np.all(counts == 64) and np.array_equal(counts, counts2)),
              "mixture_gradient_identity": error < 1e-4,
              "actual_isolated_weight_update": changed,
              "source_weights_unchanged": weights_hash(model) == before_hash,
              "cuda_fp32": next(model.parameters()).is_cuda and next(model.parameters()).dtype == torch.float32}
    receipt = {"checks": checks, "gradient_relative_error": error, "pairs": 16,
               "validation_pairs": 16, "runtime_sha256": resources["code_sha256"],
               "config_sha256": digest(read_json(root / "config.json")), "time": now(),
               "elapsed_paid_seconds": now() - resources["created_at"]}
    if not all(checks.values()):
        atomic_json(root / "acceptance_failed.json", receipt)
        raise RuntimeError("Gradient GPU acceptance failed")
    if now() >= resources["acceptance_deadline"]:
        raise RuntimeError("Gradient acceptance exceeded its paid allowance")
    atomic_json(root / "acceptance.json", receipt)
    (root / "validation_guard_deadline.json").unlink(missing_ok=True)
    del saved, gr, gs, direct
    gc.collect()


def run(root, validation=False):
    import torch
    from filelock import FileLock
    from ..science import engine
    from ..science.gradients import (mean_gradient, combine, relative_error, geometry,
                                    apply_gradient, sequence_losses, stratified_indices, paired_change)
    from ..artifacts.gradient_sources import restore_sources
    from ..completion_data import assemble
    from ..hub import backup
    from ..supervisor import deadline
    root = Path(root)
    c, resources = read_json(root / "config.json"), read_json(root / "resources.json")
    validate_config(c)
    if c.get("phase") == "layer_norms":
        from .gradient_layers import run_layers
        return run_layers(root, c, resources)
    with FileLock(str(root / "worker.lock"), timeout=0):
        try:
            atomic_json(root / "status.json", {"state": "running", "stage": "restoring", "time": now()})
            cases, bank, scientific = restore_sources(root, c)
            import importlib.metadata
            versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "datasets")}
            if versions != scientific["required_environment"]:
                raise RuntimeError("Diagnostic packages differ from the saved experiment environment")
            torch.set_num_threads(4)
            atomic_json(root / "environment.json", {"packages": versions, "gpu": torch.cuda.get_device_name(),
                        "diagnostic_precision": "fp32", "dropout": False, "tf32": False})
            vp, vt, windows = (bank.array("validation/" + n) for n in ("prompts.npy", "targets.npy", "windows.npy"))
            val = assemble(vp, vt, scientific["bos_token_id"])
            probe = stratified_indices(windows, c["probe_pairs"])
            np.save(root / "validation_probe_indices.npy", probe, allow_pickle=False)
            atomic_json(root / "probe_manifest.json", {"indices_sha256": sha256(root / "validation_probe_indices.npy"),
                        "selection_seed": 20261113, "pairs": len(probe), "documents": len(np.unique(windows[probe, 0])),
                        "source_validation_pairs": len(vp), "scope": "validation_only_never_used_for_updates"})
            probe_arrays = tuple(a[probe] for a in val)
            checked_case_sources = []
            for case in cases:
                arm, cycle = case["arm"], case["cycle"]
                folder = root / "cases" / f"{arm}-cycle{cycle:02d}"
                folder.mkdir(parents=True, exist_ok=True)
                def progress(done=0, total=0):
                    if now() > deadline(resources) - 90:
                        raise RuntimeError("Diagnostic deadline reached; preserving partial results")
                    if done == 0 or done == total or done % 256 == 0:
                        atomic_json(root / "status.json", {"state": "running", "arm": arm, "cycle": cycle,
                                    "stage": "diagnostic", "processed": done, "total": total, "time": now()})
                progress()
                model = engine.load_model({**scientific, "precision": "fp32"}, case["checkpoint"])
                if engine.weights_hash(model) != case["weights"]:
                    raise RuntimeError("Cold-restored diagnostic tensors differ")
                state = torch.load(case["optimizer"], map_location="cpu", weights_only=False) if case["optimizer"] else None
                if state and (state["model_weights_sha256"] != case["weights"] or state["global_step"] != (cycle - 1) * 79
                              or state["total_steps"] != 790):
                    raise RuntimeError("Cold-restored diagnostic optimizer differs")
                checked_case_sources.append({"arm": arm, "cycle": cycle, "weights_sha256": case["weights"],
                                            "optimizer_sha256": sha256(case["optimizer"]) if state else None})
                if not (root / "acceptance.json").exists():
                    _accept(model, case, probe_arrays, root, resources)
                    backup(root)
                saved_weights = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}
                original_lr = state["optimizer"]["param_groups"][0]["lr"] if state else scientific["learning_rate"]
                arrays = tuple(case[n] for n in ("tokens", "attention", "scoring"))
                syn = case["synthetic_rows"]
                real = np.setdiff1d(np.arange(len(arrays[0])), syn)
                gs, sm = mean_gradient(model, *arrays, rows=syn, microbatch=c["microbatch"], progress=progress)
                real_arrays = assemble(case["prompts"], case["real"], scientific["bos_token_id"])
                matched, matched_meta = mean_gradient(model, *real_arrays, rows=syn, microbatch=c["microbatch"], progress=progress)
                if len(real):
                    gr, rm = mean_gradient(model, *arrays, rows=real, microbatch=c["microbatch"], progress=progress)
                    mixed = combine(gr, gs, .01)
                    direct, dm = mean_gradient(model, *arrays, microbatch=c["microbatch"], progress=progress)
                    error = relative_error(direct, mixed)
                    if error >= 1e-4:
                        raise RuntimeError("Full mixture gradient does not reconstruct")
                    del direct
                else:
                    gr, rm, mixed, error = matched, matched_meta, None, None
                geom = geometry(gr, gs)
                summary = {"arm": arm, "cycle": cycle, "checkpoint_cycle": cycle - 1, "weights_sha256": case["weights"],
                           "optimizer_sha256": sha256(case["optimizer"]) if state else None,
                           "original_learning_rate": original_lr, "real_source": rm, "synthetic_source": sm,
                           "matched_real_source": matched_meta, "real_vs_synthetic": geom,
                           "matched_real_vs_synthetic": geometry(matched, gs), "mixture_relative_error": error,
                           "synthetic_gradient_geometry": geometry(gs),
                           "weighted_real_norm": .01 * geom["norm"] if mixed is not None else None,
                           "weighted_synthetic_norm": .99 * geom["other_norm"] if mixed is not None else geom["other_norm"],
                           "mixed_norm": geometry(mixed)["norm"] if mixed is not None else None,
                           "scope": "fp32_dropout_off_complete_source_mean_gradients"}
                atomic_json(folder / "gradients.json", summary)
                before, counts = sequence_losses(model, *probe_arrays, microbatch=c["microbatch"])
                np.savez(folder / "probe_before.npz", nll_sum=before, counts=counts, documents=windows[probe, 0])
                full_before, full_counts = sequence_losses(model, *val, microbatch=c["microbatch"])
                np.savez(folder / "validation_before.npz", nll_sum=full_before, counts=full_counts)
                interventions = []
                actions = [("optimizer_history", None), ("real_only", gr), ("synthetic_only", gs)]
                if mixed is not None:
                    actions.append(("mixed", mixed))
                for rate_name, lr in (("historical", original_lr), ("common", c["common_learning_rate"])):
                    for name, gradient in actions:
                        progress()
                        model.load_state_dict(saved_weights)
                        optimizer = apply_gradient(model, gradient, state, lr, weight_decay=scientific["weight_decay"])
                        after, after_counts = sequence_losses(model, *probe_arrays, microbatch=c["microbatch"])
                        if not np.array_equal(after_counts, counts):
                            raise RuntimeError("Diagnostic validation target counts changed")
                        item = {"arm": arm, "cycle": cycle, "action": name, "learning_rate_policy": rate_name,
                                "learning_rate": lr, **paired_change(before, after, counts, windows[probe, 0]),
                                "weights_after_sha256": engine.weights_hash(model)}
                        np.savez(folder / f"{rate_name}-{name}-probe.npz", nll_sum=after, counts=after_counts)
                        if rate_name == "historical" and name in ("mixed", "optimizer_history"):
                            full_after, measured_counts = sequence_losses(model, *val, microbatch=c["microbatch"])
                            if not np.array_equal(measured_counts, full_counts):
                                raise RuntimeError("Complete validation counts changed")
                            item["full_validation"] = paired_change(full_before, full_after, full_counts, windows[:, 0])
                            np.savez(folder / f"{name}-validation.npz", nll_sum=full_after, counts=full_counts)
                        interventions.append(item)
                        atomic_json(folder / "interventions.json", interventions)
                        del optimizer
                for rate_name in ("historical", "common"):
                    history = np.load(folder / f"{rate_name}-optimizer_history-probe.npz")["nll_sum"]
                    for item in interventions:
                        if item["learning_rate_policy"] != rate_name:
                            continue
                        after = np.load(folder / f"{rate_name}-{item['action']}-probe.npz")["nll_sum"]
                        item["relative_to_optimizer_history"] = paired_change(history, after, counts, windows[probe, 0])
                atomic_json(folder / "interventions.json", interventions)
                model.load_state_dict(saved_weights)
                if engine.weights_hash(model) != case["weights"]:
                    raise RuntimeError("Diagnostic source weights changed")
                atomic_json(folder / "completion.json", {"complete": True, "weights_sha256": case["weights"],
                            "config_sha256": digest(c), "time": now()})
                del model, state, saved_weights, gr, gs, matched, mixed, actions, gradient
                gc.collect()
                torch.cuda.empty_cache()
                backup(root)
            atomic_json(root / "source_tensor_checks.json", checked_case_sources)
            from ..analysis.gradients import report
            report(root, for_backup=True)
            atomic_json(root / "completion.json", {"complete": True, "cases": len(cases), "time": now()})
            atomic_json(root / "status.json", {"state": "complete", "time": now()})
            backup(root)
        except BaseException as error:
            atomic_json(root / "status.json", {"state": "failed", "error": str(error), "time": now()})
            try:
                backup(root)
            except Exception:
                pass
            raise


def validate_local(root):
    root = Path(root)
    c = read_json(root / "config.json")
    validate_config(c)
    acceptance = read_json(root / "acceptance.json", {})
    if not acceptance or not all(acceptance.get("checks", {}).values()):
        raise RuntimeError("Diagnostic GPU acceptance missing")
    if acceptance["config_sha256"] != digest(c):
        raise RuntimeError("Diagnostic acceptance configuration differs")
    resources = read_json(root / "resources.json", {})
    if acceptance["runtime_sha256"] != resources.get("code_sha256"):
        raise RuntimeError("Diagnostic acceptance runtime differs")
    index = read_json(root / "artifact_index.json")
    if index:
        from ..hub import verify_recovered_files
        verify_recovered_files(root, index["files"])
    if c.get("phase") == "layer_norms":
        files = list((root / "layers").glob("*.json"))
        return {"complete": len(files)==6 and bool(read_json(root/"completion.json",{}).get("complete")),
                "completed_cases":len(files),"acceptance":acceptance}
    completed = list((root / "cases").glob("*/completion.json"))
    return {"complete": len(completed) == 6 and bool(read_json(root / "completion.json", {}).get("complete")),
            "completed_cases": len(completed), "acceptance": acceptance}
