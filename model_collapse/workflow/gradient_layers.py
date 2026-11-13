"""Read-only recovery of synthetic per-parameter norms omitted by the first pass."""
from __future__ import annotations

import gc
import copy
from pathlib import Path

import numpy as np
from filelock import FileLock

from ..common import atomic_json, digest, now, read_json


def run_layers(root, c, resources):
    import torch
    from ..artifacts.gradient_sources import Snapshot, restore_sources
    from ..completion_data import assemble
    from ..science import engine
    from ..science.gradients import (mean_gradient, geometry, sequence_losses, apply_gradient,
                                    combine, parameters)
    from ..hub import backup
    from ..supervisor import deadline
    from .gradient_diagnostic import _accept
    root = Path(root)
    with FileLock(str(root / "worker.lock"), timeout=0):
        try:
            atomic_json(root / "status.json", {"state": "running", "stage": "restoring_layer_sources", "time": now()})
            reference = Snapshot(c["main_measurements"], root / ".reference")
            prior_config = reference.json("config.json")
            if prior_config["source_parent"] != c["source_parent"] or not reference.json("completion.json").get("complete"):
                raise RuntimeError("Layer pass requires the completed same-source diagnostic")
            cases, bank, scientific = restore_sources(root, c, include_weights=False)
            torch.set_num_threads(4)
            import importlib.metadata
            versions={name:importlib.metadata.version(name) for name in ("torch","transformers","datasets")}
            if versions!=scientific["required_environment"]:
                raise RuntimeError("Layer diagnostic environment differs")
            atomic_json(root/"environment.json",{"packages":versions,"gpu":torch.cuda.get_device_name(),"precision":"fp32","dropout":False})
            probe=reference.array("validation_probe_indices.npy")
            val = assemble(bank.array("validation/prompts.npy")[probe], bank.array("validation/targets.npy")[probe], scientific["bos_token_id"])
            parent = Snapshot(c["source_parent"], root / ".sources/parent")
            receipts = []
            for case in cases:
                arm, cycle = case["arm"], case["cycle"]
                prefix = f"seed_42/gen_{cycle-1:02d}"
                checkpoint = (parent if cycle == 1 else case["child"]).checkpoint(prefix)
                model = engine.load_model(scientific, checkpoint)
                if engine.weights_hash(model) != case["weights"]:
                    raise RuntimeError("Layer source tensor hash differs")
                if not (root / "acceptance.json").exists():
                    _accept(model, case, val, root, resources)
                def progress(done=0, total=0):
                    if now() >= deadline(resources) - 90:
                        raise RuntimeError("Layer supplement exhausted its combined compute allowance")
                    if done == total or done % 256 == 0:
                        atomic_json(root / "status.json", {"state": "running", "stage": "layer_gradients", "arm": arm,
                                    "cycle": cycle, "processed": done, "total": total, "time": now()})
                gradient, measured = mean_gradient(model, case["tokens"], case["attention"], case["scoring"],
                    rows=case["synthetic_rows"], microbatch=c["microbatch"], progress=progress)
                stats = geometry(gradient)
                previous = reference.json(f"cases/{arm}-cycle{cycle:02d}/gradients.json")
                error = abs(stats["norm"] - previous["real_vs_synthetic"]["other_norm"]) / max(stats["norm"], 1e-12)
                if error > 1e-5 or abs(measured["loss"] - previous["synthetic_source"]["loss"]) > 1e-6:
                    raise RuntimeError("Repeated synthetic gradient differs from the primary measurement")
                if engine.weights_hash(model) != case["weights"]:
                    raise RuntimeError("Read-only layer pass changed weights")
                # Exploratory follow-up to the observed positive real/synthetic alignment:
                # a held-out gradient is measured, but is NEVER applied to weights.
                validation_gradient, validation_meta = mean_gradient(model,*val,microbatch=c["microbatch"],progress=progress)
                source_updates=reference.json(f"cases/{arm}-cycle{cycle:02d}/interventions.json")
                state=None
                if cycle>1:
                    state_path=case["child"].get(f"seed_42/gen_{cycle-1:02d}/optimizer_state/cycle.pt")
                    state=torch.load(state_path,map_location="cpu",weights_only=False)
                    if state["model_weights_sha256"]!=case["weights"] or state["global_step"]!=(cycle-1)*79:
                        raise RuntimeError("Directional follow-up optimizer differs")
                historical_lr=state["optimizer"]["param_groups"][0]["lr"] if state else scientific["learning_rate"]
                starts={n:p.detach().cpu().clone() for n,p in model.state_dict().items()}
                before,counts=sequence_losses(model,*val,microbatch=c["microbatch"])
                directions=[]
                actions=[("synthetic_only",gradient,state)]
                real_gradient=None
                if arm=="mixed99":
                    rows=np.setdiff1d(np.arange(len(case["tokens"])),case["synthetic_rows"])
                    real_gradient,_=mean_gradient(model,case["tokens"],case["attention"],case["scoring"],rows=rows,microbatch=c["microbatch"])
                    actions.append(("mixed",combine(real_gradient,gradient,.01),state))
                without_momentum=None
                if state is not None:
                    without_momentum=copy.deepcopy(state)
                    for parameter_state in without_momentum["optimizer"]["state"].values():
                        parameter_state["exp_avg"].zero_()
                    actions.append(("synthetic_without_momentum",gradient,without_momentum))
                for action,update_gradient,optimizer_parent in actions:
                    model.load_state_dict(starts)
                    optimizer=apply_gradient(model,update_gradient,optimizer_parent,historical_lr,weight_decay=scientific["weight_decay"])
                    dot,norm2=0.,0.
                    for name,param in parameters(model):
                        displacement=(param.detach().cpu()-starts[name]).double()
                        dot+=float((displacement*validation_gradient[name].double()).sum())
                        norm2+=float(displacement.square().sum())
                    after,after_counts=sequence_losses(model,*val,microbatch=c["microbatch"])
                    if not np.array_equal(counts,after_counts):raise RuntimeError("Directional evaluation counts changed")
                    observed=float((after-before).sum()/counts.sum())
                    original=next((r for r in source_updates if r["action"]==action and r["learning_rate_policy"]=="historical"),None)
                    if original and abs(observed-original["delta_nll"])>1e-6:
                        raise RuntimeError("Repeated AdamW probe effect differs from primary measurement")
                    arrays=root/"directional_arrays";arrays.mkdir(exist_ok=True)
                    np.savez(arrays/f"{arm}-cycle{cycle:02d}-{action}.npz",before_nll_sum=before,after_nll_sum=after,counts=counts)
                    directions.append({"action":action,"historical_learning_rate":historical_lr,
                        "update_l2_norm":norm2**.5,"first_order_validation_delta_nll":dot,
                        "observed_validation_delta_nll":observed,"nonlinear_residual_nll":observed-dot,
                        "primary_delta_absolute_error":abs(observed-original["delta_nll"]) if original else None,
                        "primary_update_tensor_hash_matches":engine.weights_hash(model)==original["weights_after_sha256"] if original else None,
                        "optimizer_first_moment": "zeroed_only" if action=="synthetic_without_momentum" else "saved",
                        "optimizer_second_moment_and_step":"unchanged_from_parent"})
                    del optimizer
                model.load_state_dict(starts)
                if engine.weights_hash(model)!=case["weights"]:raise RuntimeError("Directional follow-up changed source weights")
                receipt = {"arm": arm, "cycle": cycle, "weights_sha256": case["weights"], "source": measured,
                           "synthetic_geometry": stats, "global_norm_relative_error": error,
                           "real_parameter_norms": previous["real_vs_synthetic"]["parameters"],
                           "matched_real_parameter_norms": previous["matched_real_vs_synthetic"]["parameters"],
                           "primary_reference": c["main_measurements"], "source_artifacts_read_only": True,
                           "temporary_shadow_updates_only":True,
                           "exploratory_validation_gradient":validation_meta,
                           "synthetic_vs_validation_gradient":{k:v for k,v in geometry(gradient,validation_gradient).items() if k!="parameters"},
                           "directional_interventions":directions,"validation_gradient_applied_to_weights":False,
                           "exploratory_scope":"Added after primary middle-checkpoint readout to distinguish raw gradient geometry, AdamW step direction and nonlinear loss changes."}
                atomic_json(root / "layers" / f"{arm}-cycle{cycle:02d}.json", receipt)
                receipts.append({k:v for k,v in receipt.items() if k not in ("synthetic_geometry","real_parameter_norms","matched_real_parameter_norms")})
                del model, gradient,validation_gradient,state,starts,actions,update_gradient,real_gradient,without_momentum,optimizer_parent
                gc.collect(); torch.cuda.empty_cache()
                backup(root)
            atomic_json(root / "layer_summary.json", receipts)
            atomic_json(root / "completion.json", {"complete": True, "phase": "layer_norms", "cases": len(cases), "time": now()})
            atomic_json(root / "status.json", {"state": "complete", "time": now()})
            backup(root)
        except BaseException as error:
            atomic_json(root / "status.json", {"state": "failed", "error": str(error), "time": now()})
            try: backup(root)
            except Exception: pass
            raise
