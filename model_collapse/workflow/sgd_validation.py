"""Bounded acceptance of plain SGD and actual-batch source interventions."""
from __future__ import annotations

import gc
import json
import math
from pathlib import Path

import numpy as np

from ..common import STATE, atomic_json, read_json, sha256, verify_files


def audit_state(state, training, c, expected_step):
    """SGD retains resume bookkeeping, never Adam moments or a scheduler."""
    optimizer = state.get("optimizer", {})
    if (state.get("scheduler") is not None or optimizer.get("state") != {}
            or state.get("global_step") != expected_step
            or state.get("model_weights_sha256") != training["selected_weights_sha256"]
            or not state.get("rng")):
        raise RuntimeError("SGD state contains history, wrong weights or missing resume state")
    groups = optimizer.get("param_groups", [])
    if not groups:
        raise RuntimeError("SGD optimizer parameter groups are missing")
    for group in groups:
        expected = {"lr": c["learning_rate"], "momentum": 0., "dampening": 0.,
                    "weight_decay": 0., "nesterov": False, "maximize": False,
                    "foreach": False, "fused": False}
        if any(group.get(key) != value for key, value in expected.items()):
            raise RuntimeError("SGD arithmetic or constant learning rate differs")
    return {"global_step": expected_step, "optimizer_history_entries": 0,
            "scheduler": None, "constant_learning_rate": c["learning_rate"]}


def audit_greedy_generation(manifest, expected_parent, expected_rows):
    provenance = manifest["provenance"]
    settings = provenance.get("decoding", {})
    if (provenance.get("parent_weights_sha256") != expected_parent
            or provenance.get("prompt_count") != expected_rows
            or provenance.get("new_tokens") != 64
            or provenance.get("num_beams") != 1 or provenance.get("decoder") != "greedy"
            or provenance.get("repetition_penalty") != 1.
            or settings.get("do_sample") is not False
            or settings.get("min_new_tokens") != 64 or settings.get("max_new_tokens") != 64):
        raise RuntimeError("SGD generated source or greedy decoder differs")


def audit_records(records, labels, c, cycle):
    """Check source identity, token weighting and every logical intervention."""
    labels = np.asarray(labels)
    count = len(labels)
    steps = math.ceil(count / c["effective_batch"])
    expected_steps = list(range((cycle - 1) * steps + 1, cycle * steps + 1))
    if [row.get("step") for row in records] != expected_steps:
        raise RuntimeError("SGD diagnostic updates are missing or duplicated")
    seen, real_updates, mixed_microbatches = [], 0, 0
    maximum_error = 0.
    for offset, row in enumerate(records):
        indices = np.asarray(row["batch_indices"], dtype=np.int64)
        expected_count = min(c["effective_batch"], count - offset * c["effective_batch"])
        if (len(indices) != expected_count or len(set(indices.tolist())) != len(indices)
                or np.any(indices < 0) or np.any(indices >= count)):
            raise RuntimeError("SGD batch indices or final tail differ")
        seen.extend(indices.tolist())
        real = int(np.sum(labels[indices] == 0)); synthetic = len(indices) - real
        if (row.get("cycle") != cycle or row.get("learning_rate") != c["learning_rate"]
                or row.get("real_pairs") != real or row.get("synthetic_pairs") != synthetic
                or row.get("real_scored_targets") != real * 64
                or row.get("synthetic_scored_targets") != synthetic * 64
                or row.get("total_scored_targets") != len(indices) * 64):
            raise RuntimeError("SGD source counts, targets or rate differ")
        real_updates += int(real > 0)
        for start in range(0, len(indices), c["microbatch"]):
            values = labels[indices[start:start + c["microbatch"]]]
            mixed_microbatches += int(np.any(values == 0) and np.any(values == 1))
        gradient = row["gradient"]
        error = gradient["reconstruction_relative_error"]
        if not math.isfinite(error) or error > 1e-5:
            raise RuntimeError("SGD source gradients fail direct reconstruction")
        maximum_error = max(maximum_error, error)
        for key in ("real_norm", "synthetic_norm", "total_norm"):
            if not math.isfinite(gradient[key]) or gradient[key] < 0:
                raise RuntimeError("Non-finite SGD gradient norm")
        cosine = gradient["real_synthetic_cosine"]
        if cosine is not None and (not math.isfinite(cosine) or not -1 <= cosine <= 1):
            raise RuntimeError("Invalid gradient alignment")
        if not gradient.get("layers"):
            raise RuntimeError("Per-layer gradient diagnostics are missing")
        for key in ("real_norm", "synthetic_norm", "total_norm"):
            values = [layer[key] for layer in gradient["layers"].values()]
            if any(not math.isfinite(value) or value < 0 for value in values):
                raise RuntimeError("Invalid per-layer gradient norm")
            reconstructed = math.sqrt(math.fsum(value * value for value in values))
            if abs(reconstructed - gradient[key]) > 1e-8 * max(1., reconstructed, gradient[key]):
                raise RuntimeError("Per-layer norms do not reconstruct the global gradient norm")
        if any(row.get(key) is not True for key in ("mixed_shadow_exact", "mode_preserved", "rng_preserved")):
            raise RuntimeError("SGD shadow equality or diagnostic isolation failed")
        if not math.isfinite(row["update_l2_norm"]):
            raise RuntimeError("Non-finite SGD update")
        states = row["states"]
        if set(states) != {"no_update", "real", "synthetic", "mixed"}:
            raise RuntimeError("SGD logical shadow states are incomplete")
        for state in states.values():
            if (not math.isfinite(state["loss"]) or not math.isfinite(state["perplexity"])
                    or abs(math.log(state["perplexity"]) - state["loss"]) > 1e-10):
                raise RuntimeError("Invalid token-weighted shadow NLL/perplexity")
        losses = {key: value["loss"] for key, value in states.items()}
        expected = {"real_only_delta_nll": losses["real"] - losses["no_update"],
                    "synthetic_only_delta_nll": losses["synthetic"] - losses["no_update"],
                    "mixed_delta_nll": losses["mixed"] - losses["no_update"],
                    "real_conditional_delta_nll": losses["mixed"] - losses["synthetic"],
                    "synthetic_conditional_delta_nll": losses["mixed"] - losses["real"]}
        expected["interaction_nll"] = expected["mixed_delta_nll"] - expected["real_only_delta_nll"] - expected["synthetic_only_delta_nll"]
        if any(abs(row["effects"][key] - value) > 1e-10 for key, value in expected.items()):
            raise RuntimeError("SGD effect does not equal its recorded counterfactual")
        for source, present in (("real", real), ("synthetic", synthetic)):
            if not present and (gradient[source + "_norm"] != 0 or row[source + "_source_mean_nll"] is not None
                                or losses[source] != losses["no_update"] or cosine is not None):
                raise RuntimeError("An absent SGD source is not an exact no-op")
    expected_order = np.random.default_rng(c["seed"] + cycle).permutation(count).tolist()
    if sorted(seen) != list(range(count)) or seen != expected_order:
        raise RuntimeError("SGD epoch row coverage or seeded order differs")
    return {"updates": steps, "final_batch_pairs": records[-1]["real_pairs"] + records[-1]["synthetic_pairs"],
            "real_present_updates": real_updates, "mixed_microbatches": mixed_microbatches,
            "real_pairs": int(np.sum(labels == 0)), "synthetic_pairs": int(np.sum(labels == 1)),
            "maximum_reconstruction_relative_error": maximum_error}


def _records(stage, *, expected_probe_identity=None, expected_steps=None):
    from ..science.sgd_effects import verify_seal
    stage = Path(stage)
    training = read_json(stage / "training.json")
    seal = verify_seal(stage, expected_weights=training["selected_weights_sha256"],
                       expected_probe_identity=expected_probe_identity, expected_steps=expected_steps)
    records_path = seal.get("records_path")
    if records_path is None:
        # Manifest paths are relative to the stage, never inferred from an unsealed attempt.
        records_path = next(name for name in seal["files"] if name.endswith("updates.jsonl"))
    path = stage / records_path
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line], seal


def audit_replay(stage, selected_records):
    """The one injected exit must replay, rather than duplicate, its first update."""
    stage = Path(stage); selected = read_json(stage / "diagnostics_manifest.json")["attempt_id"]
    abandoned = []
    for path in (stage / "update_diagnostics/attempts").glob("*/updates.jsonl"):
        if path.parent.name != selected:
            abandoned.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    if len(abandoned) != 1 or abandoned[0].get("step") != 5 or abandoned[0].get("cycle") != 2:
        raise RuntimeError("SGD acceptance must preserve exactly one abandoned first-update attempt")
    before, replayed = abandoned[0], selected_records[0]
    keys = ("step", "cycle", "learning_rate", "batch_indices", "real_pairs", "synthetic_pairs", "total_scored_targets")
    if any(before.get(key) != replayed.get(key) for key in keys):
        raise RuntimeError("SGD replay changed its source batch or update identity")
    if any(abs(before["states"][name]["loss"] - replayed["states"][name]["loss"]) > 1e-6
           for name in ("no_update", "real", "synthetic", "mixed")):
        raise RuntimeError("Replayed SGD probe effects differ from the abandoned attempt")
    return {"abandoned_recorded_updates": 1, "replayed_first_step": 5,
            "accepted_attempt": selected, "abandoned_attempts_excluded": True}


def validate_remote(root):
    import torch
    from .. import engine
    from ..completion_data import assemble, load_pairs
    from ..completion_mixing import select_rows
    from .completion import bank_path, child_roots, verify_completed_arm, sgd_update_probe
    from .completion_restore import verify_terminal_restore
    from .profiles import protocol_fingerprint
    from . import sgd_profile

    root = Path(root); c = read_json(root / "config.json"); resources = read_json(root / "resources.json")
    sgd_profile.validate(c); sgd_profile.require_calibration(c)
    if c["kind"] != "smoke" or c["generations"] != 2 or c["train_pairs_per_cycle"] != 400:
        raise RuntimeError("SGD acceptance requires its bounded two-cycle smoke")
    children = child_roots(root, c)
    if list(children) != [arm["id"] for arm in sgd_profile.specs(c)]:
        raise RuntimeError("SGD acceptance arm order differs")
    baseline = read_json(root / "seed_42/gen_00/metrics.json")
    original = engine.load_model(c)
    if (any(p.dtype != torch.float32 for p in original.parameters())
            or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
        raise RuntimeError("SGD acceptance requires FP32 tensors and TF32 disabled")
    if engine.weights_hash(original) != baseline["weights_sha256"]:
        raise RuntimeError("SGD baseline differs from original pretrained weights")
    first_c = read_json(next(iter(children.values())) / "config.json")
    p, t, _ = load_pairs(bank_path(next(iter(children.values())), first_c), "test", 1)
    ids, attention, scoring = assemble(p, t, first_c["bos_token_id"])
    original.eval()
    with torch.inference_mode():
        logits = original(input_ids=engine.tensor(ids), attention_mask=engine.tensor(attention), use_cache=False).logits
        actual = engine.shifted_nll(logits, engine.tensor(ids), engine.tensor(scoring)).sum().item() / 64
        direct = float((-torch.log_softmax(logits[0, 64:128].float(), dim=-1)[
            torch.arange(64, device=logits.device), engine.tensor(t[0])]).mean())
    if abs(actual - direct) > 1e-5:
        raise RuntimeError("SGD native direct conditional NLL differs")
    del original, logits; gc.collect(); torch.cuda.empty_cache()
    audits, restores, all_records, replay = {}, {}, [], None
    for name, child in children.items():
        arm = read_json(child / "config.json"); bank = bank_path(child, arm)
        update_probe = sgd_update_probe(bank, arm)
        verify_completed_arm(child, arm)
        verify_files(child / "seed_42/gen_01", read_json(child / "cycle1_snapshot.json")["files"])
        previous = baseline["weights_sha256"]; optimizer_hash = None; summaries = []
        for cycle in (1, 2):
            stage = child / f"seed_42/gen_{cycle:02d}"
            metrics = read_json(stage / "metrics.json"); training = metrics["training"]
            history = read_json(stage / "training_history.json")
            expected_step = cycle * 4
            if (len(history) != 1 or history[0].get("global_step") != expected_step
                    or history[0].get("scored_training_tokens") != 25600
                    or history[0].get("learning_rate_end") != arm["learning_rate"]
                    or not all(math.isfinite(history[0][key]) for key in ("train_loss", "validation_loss"))):
                raise RuntimeError("SGD smoke did not execute one constant-rate epoch")
            prompt, real, _ = load_pairs(bank, f"cycle_{cycle:02d}", 400)
            fraction = arm["synthetic_fraction"]
            selection, n_real = select_rows(400, fraction, arm["seed"], cycle)
            indices = selection["synthetic_indices"] if 0 < fraction < 1 else np.arange(400) if fraction == 1 else np.array([], dtype=np.int64)
            expected = real.copy()
            if len(indices):
                folder = child / f"synthetic_for_gen_{cycle:02d}"
                generated = np.load(folder / "tokens.npy", allow_pickle=False)
                generation = read_json(folder / "manifest.json")
                audit_greedy_generation(generation, previous, len(indices))
                if generated.shape != (len(indices), 64):
                    raise RuntimeError("SGD generated continuation shape differs")
                expected[indices] = generated
            arrays = assemble(prompt, expected, arm["bos_token_id"])
            for filename, values in zip(("tokens.npy", "attention.npy", "loss_mask.npy"), arrays):
                if not np.array_equal(np.load(stage / "training_input" / filename, allow_pickle=False), values):
                    raise RuntimeError("SGD training input does not reconstruct from its sources")
            if not np.all(arrays[2][:, :65] == 0) or not np.all(arrays[2][:, 65:] == 1):
                raise RuntimeError("SGD prompt labels or scored continuation boundary differ")
            lineage = training["lineage"]
            if (training["initial_weights_sha256"] != previous or lineage["initializer_weights_sha256"] != previous
                    or training["optimizer_parent_sha256"] != optimizer_hash
                    or lineage["generator_weights_sha256"] != (previous if len(indices) else None)
                    or training["global_step"] != expected_step or training["selected_weights_sha256"] == previous
                    or training["selected_epoch"] != 1 or training["executed_epochs"] != 1):
                raise RuntimeError("SGD initializer/generator/state handoff differs")
            state = torch.load(stage / "optimizer_state/cycle.pt", map_location="cpu", weights_only=False)
            state_audit = audit_state(state, training, arm, expected_step)
            records, seal = _records(stage, expected_probe_identity=update_probe["identity"],
                                      expected_steps=range(expected_step - 3, expected_step + 1))
            record_audit = audit_records(records, selection["source_labels"], arm, cycle)
            probe_folder = stage / "probe" / f"synthetic_for_gen_{cycle + 1:02d}"
            audit_greedy_generation(read_json(probe_folder / "manifest.json"),
                                     training["selected_weights_sha256"], c["probe_count"])
            if name == sgd_profile.interruption_arm(c) and cycle == 2:
                replay = audit_replay(stage, records)
            if not all(math.isfinite(metrics[split]["loss"]) and metrics[split]["scored_tokens"] == 8192
                       for split in ("validation", "test")):
                raise RuntimeError("SGD smoke held-out loss/counts differ")
            summaries.append({"cycle": cycle, "initializer": previous, "selected": training["selected_weights_sha256"],
                              "generator": lineage["generator_weights_sha256"], "state": state_audit,
                              "updates": record_audit, "diagnostic_manifest_sha256": sha256(stage / "diagnostics_manifest.json")})
            all_records.extend(records)
            previous = training["selected_weights_sha256"]; optimizer_hash = sha256(stage / "optimizer_state/cycle.pt")
            del state
        audits[name] = summaries
        restores[name] = verify_terminal_restore(root, child, arm)
    interrupted = children[sgd_profile.interruption_arm(c)]
    events = []
    for path in (interrupted / "events.jsonl", root / "events.jsonl"):
        if path.exists(): events.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    interruption = [row for row in events if row.get("event") in ("sgd_update_interruption", "sgd_training_interruption", "fresh_real_midcycle_interruption")]
    if (len(interruption) != 1 or interruption[0].get("generation") != 2
            or interruption[0].get("global_step") != 5):
        raise RuntimeError("Exactly one deliberate SGD training interruption must be recorded")
    checks = {key: True for key in sgd_profile.checks(c) - {"detached_progress_verified", "independent_watchdog_and_provider_stop_verified"}}
    expected_updates = len(children) * c["generations"] * math.ceil(c["train_pairs_per_cycle"] / c["effective_batch"])
    if len(all_records) != expected_updates:
        raise RuntimeError("SGD acceptance logical update count differs")
    result = {"workflow": "completion_comparison", "acceptance_scope": sgd_profile.acceptance_scope(c),
              "source_code_sha256": resources["code_sha256"], "protocol_fingerprint": protocol_fingerprint(c),
              "checks": checks, "optimizer_updates": expected_updates, "additional_replay_updates": replay["abandoned_recorded_updates"],
              "training_replay": replay,
              "stages": audits, "terminal_restores": restores, "direct_nll_absolute_error": abs(actual - direct),
              "coverage_limit": "Two reduced fresh chunks per arm and one injected training replay. No full-data execution, partial-generation interruption matrix or every historical-stage restore is claimed."}
    atomic_json(root / "remote_validation.json", result)
    benchmark = read_json(root / "benchmark.json", {})
    benchmark["measured_stage_bytes"] = max(sum(p.stat().st_size for p in (child / "seed_42/gen_02").rglob("*") if p.is_file()) for child in children.values())
    benchmark["diagnostic_probe_pairs"] = 128
    atomic_json(root / "benchmark.json", benchmark)
    return result


def validate_local(root):
    from ..remote import local_resources, source_archive, Runpod
    from ..mixing import certificate_name
    from .completion import verify_recovery, benchmark_filename
    from .profiles import require_acceptance
    from . import sgd_profile
    root = Path(root); verify_recovery(root)
    c = read_json(root / "config.json"); sgd_profile.require_calibration(c)
    resources = local_resources(root.name); result = read_json(root / "remote_validation.json")
    guard = read_json(root / "watchdog_exit.json", {}); pod = Runpod().pod(resources["pod_id"])
    if pod["status"] != "EXITED" or guard.get("pod_id") != resources["pod_id"] or guard.get("reason") != "complete":
        raise RuntimeError("SGD ordinary watchdog/provider shutdown is unverified")
    detached = read_json(root / "detached_verification.json", {})
    if detached.get("pod_id") != resources["pod_id"] or detached.get("progress_after_disconnect") is not True:
        raise RuntimeError("SGD progress across closed SSH connections is unverified")
    runtime = source_archive()[1]
    result["checks"].update(detached_progress_verified=True, independent_watchdog_and_provider_stop_verified=True)
    require_acceptance(c, result, runtime)
    result.update(validated_run_id=root.name, provider_pod_state=pod["status"], gpu=resources["gpu"],
                  validation_storage_backend=resources.get("storage_backend", "network"), full_working_storage_backend="network")
    atomic_json(root / "validation_results.json", result)
    atomic_json(STATE / certificate_name(c, runtime), result)
    atomic_json(STATE / benchmark_filename(c), read_json(root / "benchmark.json"))
    return result
