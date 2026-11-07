"""Actual masked-training, lineage, recovery and provider acceptance checks."""
from __future__ import annotations

import gc
import json
import math
from pathlib import Path

import numpy as np

from ..common import STATE, atomic_json, read_json, sha256, verify_files
from ..completion_data import assemble, load_pairs, restore_bank
from .completion import acceptance_checks, bank_path, child_roots, verify_completed_arm, benchmark_filename, smoke_pair_count
from ..mixing import mixture_counts


def validate_remote(root):
    import torch
    from .. import engine
    from ..hub import fetch
    from .profiles import protocol_fingerprint
    root = Path(root)
    c, resources = read_json(root / "config.json"), read_json(root / "resources.json")
    count = smoke_pair_count(c)
    if c["kind"] != "smoke" or c["generations"] != 3 or c["train_pairs_per_cycle"] != count:
        raise RuntimeError("Completion GPU validation requires the declared three-cycle smoke")
    bank = bank_path(root, c)
    bank_manifest = restore_bank(c["prepared_bank"], bank)
    c["bos_token_id"] = bank_manifest["bos_token_id"]
    baseline = read_json(root / "seed_42/gen_00/metrics.json")
    model = engine.load_model(c)
    if engine.weights_hash(model) != baseline["weights_sha256"]:
        raise RuntimeError("Shared baseline differs from original pretrained weights")
    del model
    gc.collect()
    audits, restores, selections = {}, {}, {}
    examples = {}
    for name, child in child_roots(root).items():
        arm = read_json(child / "config.json")
        snapshot = verify_completed_arm(child, arm)
        before = read_json(child / "cycle1_snapshot.json")
        verify_files(child / "seed_42/gen_01", before["files"])
        summaries = []
        for cycle in range(1, 4):
            stage = child / f"seed_42/gen_{cycle:02d}"
            training = read_json(stage / "training.json")
            ids = np.load(stage / "training_input/tokens.npy")
            attention = np.load(stage / "training_input/attention.npy")
            scoring = np.load(stage / "training_input/loss_mask.npy")
            prompt, target, _ = load_pairs(bank, f"cycle_{cycle:02d}", count)
            if (ids.shape != (count, 129) or not np.all(attention == 1) or
                    not np.all(scoring[:, :65] == 0) or not np.all(scoring[:, 65:] == 1) or
                    not np.array_equal(ids[:, 1:65], prompt) or int(scoring[:, 1:].sum()) != count * 64):
                raise RuntimeError("GPU training did not use visible prompts and continuation-only targets")
            fraction = arm.get("synthetic_fraction", float(arm["data_kind"] == "synthetic"))
            real_count, synthetic_count = mixture_counts(count, fraction)
            expected_target = target.copy()
            if c.get("completion_arms") is not None:
                from ..completion_mixing import select_rows
                expected_selection, _ = select_rows(count, fraction, c["seed"], cycle)
                selections.setdefault(name, {})[cycle] = expected_selection
                for key, expected in expected_selection.items():
                    actual = np.load(stage / "mixture" / (key + ".npy"))
                    if not np.array_equal(actual, expected):
                        raise RuntimeError("GPU mixture selection differs from its deterministic reconstruction")
                if synthetic_count:
                    generated = np.load(child / f"synthetic_for_gen_{cycle:02d}/tokens.npy")
                    expected_target[expected_selection["synthetic_indices"]] = generated
            elif synthetic_count:
                expected_target = np.load(child / f"synthetic_for_gen_{cycle:02d}/tokens.npy")
            if not np.array_equal(ids[:, 65:], expected_target):
                raise RuntimeError("Training continuation differs from its declared source")
            history = read_json(stage / "training_history.json")
            if len(history) != 1 or training["selected_epoch"] != 1 or history[0]["scored_training_tokens"] != count * 64:
                raise RuntimeError("Smoke stage did not execute exactly one scored epoch")
            if training["initial_weights_sha256"] == training["selected_weights_sha256"]:
                raise RuntimeError("GPU smoke stage did not update weights")
            state = torch.load(stage / "optimizer_state/cycle.pt", map_location="cpu", weights_only=False)
            step = cycle * math.ceil(count / c["effective_batch"])
            if (state["global_step"] != step or state["scheduler"]["last_epoch"] != step or
                    state["model_weights_sha256"] != training["selected_weights_sha256"] or
                    any(int(v["step"]) != step for v in state["optimizer"]["state"].values())):
                raise RuntimeError("Optimizer state did not continue across all cycles")
            if cycle == 3 and not math.isclose(state["optimizer"]["param_groups"][0]["lr"], c["min_learning_rate"], rel_tol=1e-9):
                raise RuntimeError("Global cosine schedule did not reach its terminal learning rate")
            summaries.append({"cycle": cycle, "weights_sha256": training["selected_weights_sha256"],
                              "initializer_sha256": training["initial_weights_sha256"], "optimizer_step": step,
                              "scored_training_tokens": count * 64,
                              "real_pairs": real_count, "synthetic_pairs": synthetic_count})
            del state
        events = [json.loads(line)["event"] for line in (child / "events.jsonl").read_text().splitlines()]
        if events.count("fresh_real_midcycle_interruption") != 1:
            raise RuntimeError("Training recovery injection was not exercised")
        if arm.get("synthetic_fraction", float(arm["data_kind"] == "synthetic")) > 0:
            source = read_json(child / "synthetic_for_gen_02/manifest.json")
            reused = min(c["generation_shard"], mixture_counts(count, fraction)[1])
            if source["reused_continuations"] != reused or events.count("validation_generation_interruption") != 1:
                raise RuntimeError("Partial synthetic generation did not resume from its verified shard")
        restored = root.parent.parent / "completion_acceptance_restore" / child.name
        recovery = fetch(child.name, resources["repos"], restored, revisions=snapshot["revisions"])
        if recovery["artifact_index_sha256"] != snapshot["artifact_index_sha256"]:
            raise RuntimeError("Restored child backup index differs")
        if c.get("completion_arms") is not None:
            for cycle in range(1, 4):
                restored_stage = restored / f"seed_42/gen_{cycle:02d}"
                if sha256(restored_stage / "stage_completion.json") != sha256(child / f"seed_42/gen_{cycle:02d}/stage_completion.json"):
                    raise RuntimeError("Restored completed-stage seal differs")
                verify_files(restored_stage, read_json(restored_stage / "stage_completion.json")["files"])
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(c["model"], revision=c["model_revision"])
            first = child / "seed_42/gen_01"
            prompt, real, _ = load_pairs(bank, "cycle_01", count)
            trained = np.load(first / "training_input/tokens.npy")[:, 65:]
            indices = selections[name][1]["synthetic_indices"][:3]
            examples[name] = [{"row_index": int(i), "prompt": tokenizer.decode(prompt[i]),
                               "real_continuation": tokenizer.decode(real[i]),
                               "generated_training_continuation": tokenizer.decode(trained[i])} for i in indices]
        final = restored / "seed_42/gen_03"
        model = engine.load_model(c, final / "checkpoint")
        if engine.weights_hash(model) != summaries[-1]["weights_sha256"]:
            raise RuntimeError("Restored terminal checkpoint tensor hash differs")
        prompt, target, _ = load_pairs(bank, "test", 64)
        ids, attention, scoring = assemble(prompt, target, c["bos_token_id"])
        measured = engine.evaluate(model, ids, c, masks=attention, loss_masks=scoring)
        original = read_json(final / "metrics.json")["test"]
        error = abs(measured["loss"] - original["loss"])
        if error > 1e-6 or measured["scored_tokens"] != original["scored_tokens"]:
            raise RuntimeError("Restored masked held-out evaluation differs")
        x, mask = engine.tensor(ids[:2]), engine.tensor(attention[:2])
        with torch.inference_mode(), engine.autocast(c):
            logits = model(input_ids=x, attention_mask=mask, use_cache=False).logits
        direct = -torch.log_softmax(logits[:, 64:128].float(), -1).gather(-1, x[:, 65:, None]).squeeze(-1)
        reference = engine.shifted_nll(logits, x, engine.tensor(scoring[:2]))[:, 64:]
        direct_error = float((direct - reference).abs().max())
        if direct_error > 2e-6:
            raise RuntimeError("Direct conditional NLL calculation differs")
        optimizer = torch.load(final / "optimizer_state/cycle.pt", map_location="cpu", weights_only=False)
        if optimizer["model_weights_sha256"] != engine.weights_hash(model):
            raise RuntimeError("Restored optimization state belongs to different weights")
        audits[name] = {"stages": summaries, "restored_loss_error": error, "direct_nll_error": direct_error,
                        "restored_optimizer_sha256": sha256(final / "optimizer_state/cycle.pt")}
        restores[name] = recovery
        del model, logits, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    events = [json.loads(line)["event"] for line in (root / "events.jsonl").read_text().splitlines()]
    if events.count("completion_handoff_interruption") != 1 or events.count("validation_worker_resumed") != 1:
        raise RuntimeError("Coordinator did not recover exactly once across the arm handoff")
    if selections:
        left = next(iter(selections.values()))
        for cycle in range(1, 4):
            comparisons = [value[cycle] for value in list(selections.values())[1:]]
            comparisons.extend(select_rows(count, fraction, c["seed"], cycle)[0] for fraction in (.2, .5))
            for b in comparisons:
                a = left[cycle]
                smaller, larger = sorted((a["synthetic_indices"], b["synthetic_indices"]), key=len)
                if not np.array_equal(a["permutation"], b["permutation"]) or not set(smaller).issubset(larger):
                    raise RuntimeError("Mixed ratios do not use nested synthetic selections")
    checks = {key: True for key in acceptance_checks(c) - {"detached_progress_verified", "independent_budget_guard_and_provider_stop_verified"}}
    atomic_json(root / "remote_validation.json", {"workflow": "completion_comparison", "checks": checks,
                "source_code_sha256": resources["code_sha256"], "protocol_fingerprint": protocol_fingerprint(read_json(root / "config.json")),
                "arm_audits": audits, "restorations": restores,
                "unique_training_stages": 3 * len(audits), "executed_epochs": 3 * len(audits)})
    if examples:
        atomic_json(root / "mixture_examples.json", examples)


def validate_local(root):
    from ..remote import local_resources, source_archive, Runpod
    from ..mixing import certificate_name
    from .completion import verify_recovery
    from .profiles import protocol_fingerprint, require_acceptance
    root = Path(root)
    verify_recovery(root)
    c, r = read_json(root / "config.json"), local_resources(root.name)
    result = read_json(root / "remote_validation.json")
    guard = read_json(root / "watchdog_exit.json", {})
    pod = Runpod().pod(r["pod_id"])
    if pod["status"] != "EXITED" or guard.get("pod_id") != r["pod_id"] or guard.get("reason") != "validation_budget_guard":
        raise RuntimeError("Independent completion watchdog/provider shutdown is unverified")
    detached = read_json(root / "detached_verification.json", {})
    if detached.get("pod_id") != r["pod_id"] or detached.get("progress_after_disconnect") is not True:
        raise RuntimeError("Detached completion progress was not verified")
    runtime = source_archive()[1]
    result["checks"].update(detached_progress_verified=True, independent_budget_guard_and_provider_stop_verified=True)
    require_acceptance(c, result, runtime)
    result.update(validated_run_id=root.name, provider_pod_state=pod["status"], gpu=r["gpu"])
    atomic_json(root / "validation_results.json", result)
    atomic_json(STATE / certificate_name(c, runtime), result)
    atomic_json(STATE / benchmark_filename(c), read_json(root / "benchmark.json"))
    return result
