"""Matched real and self-generated completion training under one provider owner."""
from __future__ import annotations

import copy
import gc
import hashlib
import math
import re
import shutil
import time
from pathlib import Path

from . import smollm

import numpy as np
from filelock import FileLock

from ..common import STATE, atomic_json, checkpoint_manifest, digest, event, now, read_json, sha256, verify_files
from ..completion_data import assemble, bank_contract, load_pairs, restore_bank
from ..mixing import mixture_counts, synthetic_fraction

ALLOCATION = "fineweb_completion64"
ACCEPTANCE = "fineweb_completion64_acceptance"
ORDER = ("real", "synthetic")
MIXTURE_ALLOCATION = "fineweb_completion64_mixtures"
REPEATED_ALLOCATION = "fineweb_completion64_repeated_chunk"
DECODING_ALLOCATION = "fineweb_completion64_decoding"
CHECKS = {"six_terminal_epochs", "matched_initial_weights_and_prompts", "continuation_only_loss",
          "all_handoffs_and_optimizer_states", "generation_shard_recovery", "training_recovery",
          "completed_stages_unchanged", "coordinator_recovery", "both_private_restores",
          "restored_evaluation", "direct_masked_nll", "detached_progress_verified",
          "independent_budget_guard_and_provider_stop_verified"}


class BackupPending(RuntimeError):
    pass


def native_profile(c):
    from . import sgd_profile
    return sgd_profile if sgd_profile.enabled(c) else smollm if smollm.enabled(c) else None


def native_model(c):
    return native_profile(c) is not None


def source_collection(c, cycle):
    """Resolve scientific data identity independently of the training epoch."""
    policy = c.get("training_chunk_policy", "fresh_per_cycle")
    if policy not in ("fresh_per_cycle", "repeat_fixed"):
        raise ValueError("Unknown completion training_chunk_policy")
    if type(cycle) is not int or not 1 <= cycle <= c.get("training_cycles", c.get("full_cycles", 10)):
        raise ValueError("Invalid completion training cycle")
    fixed = c.get("fixed_chunk")
    if policy == "repeat_fixed":
        if type(fixed) is not int or fixed != 1:
            raise ValueError("Repeated-chunk control requires fixed_chunk: 1")
        cycle = fixed
    elif fixed is not None:
        raise ValueError("fixed_chunk is valid only for repeat_fixed")
    return f"cycle_{cycle:02d}"


def arm_specs(c):
    if native_model(c):
        return native_profile(c).specs(c)
    decoders = c.get("decoding_arms")
    if decoders is not None:
        expected = [{"id": "greedy", "decoder": "greedy"}, {"id": "sampling", "decoder": "sampling"}]
        if decoders != expected or c.get("completion_arms") is not None or c.get("endpoint_arms") is not None:
            raise ValueError("Decoding comparison requires greedy then sampling, without mixture/endpoint definitions")
        return [{**item, "synthetic_fraction": 1.0} for item in decoders]
    definitions = c.get("completion_arms")
    if definitions is None:
        endpoints = c.get("endpoint_arms", list(ORDER))
        if (not isinstance(endpoints, list) or not endpoints or len(endpoints) > 2 or
                any(name not in ORDER for name in endpoints) or len(set(endpoints)) != len(endpoints)):
            raise ValueError("Endpoint arms must be a unique nonempty real/synthetic subset")
        return [{"id": name, "synthetic_fraction": float(name == "synthetic")} for name in endpoints]
    if c.get("endpoint_arms") is not None:
        raise ValueError("Endpoint arms and mixture arms are mutually exclusive")
    if not isinstance(definitions, list) or len(definitions) not in (1, 2):
        raise ValueError("Completion comparison requires one or two ordered arm definitions")
    names = set()
    for item in definitions:
        if not isinstance(item, dict) or set(item) != {"id", "synthetic_fraction"}:
            raise ValueError("Completion arms require an id and synthetic_fraction")
        name = item["id"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", name) or name in names:
            raise ValueError("Completion arm identifiers must be unique safe names")
        names.add(name)
        synthetic_fraction(item)
    return definitions


def allocation_id(c):
    if native_model(c):
        if c.get("optimizer") == "sgd":
            return native_profile(c).allocation_id(c)
        return native_profile(c).SETUP_ALLOCATION if native_profile(c).__name__.endswith("sgd_profile") and c["kind"] == "smoke" else native_profile(c).ALLOCATION
    if c.get("decoding_arms") is not None:
        return DECODING_ALLOCATION
    if c.get("training_chunk_policy") == "repeat_fixed":
        if c.get("endpoint_arms") is not None:
            return REPEATED_ALLOCATION + "_" + "_".join(x["id"] for x in arm_specs(c)) + "_" + str(c.get("training_cycles", c["full_cycles"])) + "epochs"
        return REPEATED_ALLOCATION
    if c.get("completion_arms") is not None and len(arm_specs(c)) == 1:
        return MIXTURE_ALLOCATION + "_" + arm_specs(c)[0]["id"]
    return MIXTURE_ALLOCATION if c.get("completion_arms") is not None else ALLOCATION


def acceptance_campaign(c):
    return allocation_id(c) + "_acceptance"


def benchmark_filename(c):
    if native_model(c):
        if c.get("optimizer") == "sgd":
            return "benchmark_sgd_greedy_mixtures.json" if native_profile(c).mixtures(c) else "benchmark_sgd_greedy_effects.json"
        return "benchmark_sgd_greedy_effects.json" if c.get("optimizer") == "sgd" else "benchmark_smollm2_completion64.json"
    if c.get("decoding_arms") is not None:
        return "benchmark_completion_decoding.json"
    if c.get("training_chunk_policy") == "repeat_fixed":
        if c.get("endpoint_arms") is not None:
            return "benchmark_" + allocation_id(c) + ".json"
        return "benchmark_completion_repeated_chunk.json"
    if c.get("completion_arms") is not None and len(arm_specs(c)) == 1:
        return "benchmark_completion_" + arm_specs(c)[0]["id"] + ".json"
    return "benchmark_completion_mixtures.json" if c.get("completion_arms") is not None else "benchmark_completion64.json"


def acceptance_checks(c):
    if native_model(c):
        if c.get("optimizer") == "sgd":
            return native_profile(c).checks(c)
        return native_profile(c).CHECKS
    extra = {"exact_mixture_reconstruction", "nested_selections", "sealed_private_stage_restores"}
    checks = CHECKS | extra if c.get("completion_arms") is not None else CHECKS
    if c.get("decoding_arms") is not None:
        checks |= {"explicit_training_decoders", "fixed_five_beam_probe", "sampling_rng_isolation",
                   "sampling_recovery_byte_identity", "historical_beam_parity", "sealed_private_stage_restores"}
    if c.get("training_chunk_policy") == "repeat_fixed":
        checks = checks | {"fixed_chunk_inputs", "fixed_real_anchor"}
        if any(item["synthetic_fraction"] for item in arm_specs(c)):
            checks = checks | {"recursive_same_prompt_targets"}
    if c["learning_rate"] == c["min_learning_rate"]:
        checks = checks | {"constant_learning_rate_all_updates"}
    if not any(item["synthetic_fraction"] for item in arm_specs(c)):
        checks = checks - {"generation_shard_recovery"}
    if len(arm_specs(c)) == 1:
        return (checks - {"six_terminal_epochs", "both_private_restores"}) | {"three_terminal_epochs", "private_restore"}
    return checks


def smoke_pair_count(c):
    if native_model(c):
        return 400 if c.get("optimizer") == "sgd" else 128
    """Keep historical scopes, with explicit larger smoke sets for sparse mixtures."""
    default = 320 if c.get("completion_arms") is not None else 256
    count = c.get("smoke_train_pairs", default)
    if type(count) is not int or not 256 <= count <= 10000:
        raise ValueError("Completion smoke_train_pairs must be an integer from 256 to 10000")
    return count


def required_backup_bytes(c):
    if native_model(c):
        return native_profile(c).required_backup_bytes(c) if c.get("optimizer") == "sgd" else (6_000_000_000 if c["kind"] == "smoke" else 131_000_000_000)
    if len(arm_specs(c)) == 2:
        return 40_000_000_000
    # The verified OPT stages use <1.53 GB each, including model, optimizer,
    # inputs and generated pools. Namespace admission also keeps 10 GB free.
    # Smoke admission covers its three stages plus the later ten-stage run.
    stages = c.get("training_cycles", c["full_cycles"]) + (c["generations"] if c["kind"] == "smoke" else 0)
    return stages * 1_530_000_000


def arm_config(c, name):
    if native_model(c):
        spec = next(x for x in arm_specs(c) if x["id"] == name)
        f = spec["synthetic_fraction"]
        return {**c, "workflow": "completion_arm", "arm_id": name, "synthetic_fraction": f,
                "data_kind": "real" if f == 0 else "synthetic" if f == 1 else "mixed"}
    arm = {**c, "workflow": "completion_arm", "data_kind": name}
    if c.get("decoding_arms") is not None:
        from ..science.decoding import decoder_config
        spec = next(x for x in arm_specs(c) if x["id"] == name)
        arm = decoder_config({**arm, "data_kind": "synthetic", "arm_id": name}, spec["decoder"])
    elif c.get("completion_arms") is not None:
        spec = next(x for x in arm_specs(c) if x["id"] == name)
        arm.update(data_kind="mixed", arm_id=name, synthetic_fraction=spec["synthetic_fraction"])
    return arm


def validate_config(c):
    if native_model(c):
        return native_profile(c).validate(c)
    from .configuration import validate_training_numbers
    validate_training_numbers(c)
    specs = arm_specs(c)
    source_collection(c, 1)
    if c.get("decoding_arms") is not None:
        from ..science.decoding import decoding_settings
        decoding_settings(c)
        if c.get("training_chunk_policy", "fresh_per_cycle") != "fresh_per_cycle":
            raise ValueError("Decoding comparison requires ten fresh chunks")
        if c["generation_shard"] % c["generation_batch"]:
            raise ValueError("Decoding shard size must be divisible by its batch size")
        if c["workflow"] == "completion_comparison" and c.get("generation_decoder") is not None:
            raise ValueError("The parent retains historical five-beam probe decoding")
    cycles = c.get("training_cycles", c["full_cycles"])
    if (type(cycles) is not int or not 1 <= cycles <= 20 or
            (cycles != c["full_cycles"] and c.get("training_chunk_policy") != "repeat_fixed")):
        raise ValueError("Only repeated-chunk controls may extend training beyond the frozen bank")
    if c.get("endpoint_arms") is not None and c.get("training_chunk_policy") != "repeat_fixed":
        raise ValueError("Explicit endpoint subsets currently require repeat_fixed")
    if c.get("training_chunk_policy") == "repeat_fixed" and c.get("completion_arms") is not None:
        raise ValueError("Repeated-chunk control requires the historical real/synthetic endpoint arms")
    if c.get("backup_store"):
        from ..cloud.s3_artifacts import validate_store
        validate_store(c["backup_store"])
    if c.get("completion_arms") is not None:
        from ..completion_mixing import POLICY
        if c.get("mixture_selection_policy") != POLICY:
            raise ValueError("Unknown completion mixture selection policy")
    if (c.get("workflow") not in ("completion_comparison", "completion_arm") or
            c.get("model") != "facebook/opt-125m" or c.get("dataset") != "HuggingFaceFW/fineweb-edu" or
            c.get("dataset_config") != "sample-10BT" or c.get("epochs") != 1 or
            c.get("checkpoint_selection") != "final_epoch" or c.get("initialization") != "continued" or
            c.get("optimizer_policy") != "carry_across_cycles"):
        raise ValueError("Completion comparison requires pretrained OPT, continued optimization and one terminal epoch")
    for key in ("model_revision", "dataset_revision"):
        if len(c.get(key, "")) != 40 or any(x not in "0123456789abcdef" for x in c[key]):
            raise ValueError("Completion sources require pinned revisions")
    expected = {"full_cycles": 10, "full_train_pairs": 10000, "max_sequence_length": 129,
                "block_size": 64, "new_tokens": 64, "network_volume_gb": 60}
    if c.get("decoding_arms") is None:
        expected["num_beams"] = 5
    if any(c.get(k) != v for k, v in expected.items()) or c.get("repetition_penalty") != 1.0:
        raise ValueError("Completion comparison has an incompatible length, decoding or bank policy")
    profile = (c.get("generations"), c.get("train_pairs_per_cycle"), c.get("evaluation_pairs"), c.get("probe_count"))
    smoke_pairs = smoke_pair_count(c)
    if profile != ({"full": (cycles, 10000, None, 256), "smoke": (3, smoke_pairs, 64, 64)}.get(c.get("kind"))):
        raise ValueError("Completion comparison scope differs from its full/smoke definition")
    for key in ("microbatch", "effective_batch", "generation_batch", "generation_shard", "eval_batch", "shuffle_buffer"):
        if type(c.get(key)) is not int or c[key] <= 0:
            raise ValueError("Invalid completion batch size")
    if c["effective_batch"] % c["microbatch"] or c.get("precision") not in ("bf16", "fp32"):
        raise ValueError("Invalid completion accumulation or precision")
    if not 0 < c.get("min_learning_rate", 0) <= c.get("learning_rate", 0) or c.get("weight_decay", -1) < 0:
        raise ValueError("Invalid completion optimizer settings")
    if c.get("storage_backend") not in ("network", "pod_persistent") or not c.get("dataset_files"):
        raise ValueError("Completion comparison requires pinned sample shards and persistent storage")
    if c.get("storage_backend") == "pod_persistent" and not (
            c.get("training_chunk_policy") == "repeat_fixed" and len(specs) == 1):
        raise ValueError("Pod-persistent completion storage requires one repeated endpoint")
    if c["workflow"] == "completion_arm":
        if c.get("decoding_arms") is not None:
            spec = next((x for x in specs if x["id"] == c.get("arm_id")), None)
            if not spec or c.get("data_kind") != "synthetic" or c.get("generation_decoder") != spec["decoder"]:
                raise ValueError("Training decoder arm differs from its parent definition")
        elif c.get("completion_arms") is None:
            if c.get("data_kind") not in ORDER:
                raise ValueError("Unknown completion arm")
        elif (c.get("data_kind") != "mixed" or not any(
                x["id"] == c.get("arm_id") and x["synthetic_fraction"] == c.get("synthetic_fraction") for x in specs)):
            raise ValueError("Mixed completion arm differs from its parent definition")


def resolve_inputs(c):
    if native_model(c):
        return native_profile(c).resolve_inputs(c)
    c = copy.deepcopy(c)
    ref = c.get("prepared_bank") or read_json(STATE / "completion64_bank.json")
    if not ref or ref.get("contract") != bank_contract(c):
        raise RuntimeError("Prepare and publish the matching full completion bank before provisioning")
    if any(len(v) != 40 for v in ref["revisions"].values()):
        raise RuntimeError("Completion bank backup commits must be immutable")
    path = Path(STATE).parent / "data/fineweb_completion64/manifest.json"
    if not path.exists() or sha256(path) != ref["manifest_sha256"]:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(ref["repos"]["dataset"], ref["run_id"] + "/manifest.json", repo_type="dataset", revision=ref["revisions"]["dataset"]))
    if sha256(path) != ref["manifest_sha256"]:
        raise RuntimeError("Prepared completion bank manifest checksum differs")
    manifest = read_json(path)
    if (manifest["contract"] != ref["contract"] or any(
            manifest["collections"].get(f"cycle_{i:02d}", {}).get("pairs") != 10000 for i in range(1, 11)) or
            manifest["collections"]["validation"]["pairs"] != 7322 or manifest["collections"]["test"]["pairs"] != 7151):
        raise RuntimeError("Full completion launch requires the complete full-data bank")
    c["prepared_bank"] = ref
    return c


def bank_path(root, c):
    return Path(root).parent.parent / "completion_banks" / c["prepared_bank"]["run_id"]


def child_roots(root, c=None):
    root = Path(root)
    c = c if c is not None else read_json(root / "config.json", {})
    return {item["id"]: root.with_name(root.name + "-" + item["id"]) for item in arm_specs(c)}


def _write_fixed(path, value):
    previous = read_json(path)
    if previous is not None and previous != value:
        raise RuntimeError("Completed comparison identity changed: " + Path(path).name)
    if previous is None:
        atomic_json(path, value)


def _visible_prompt(prompt, bos):
    return np.concatenate([np.full((len(prompt), 1), bos, dtype=np.int32), prompt], axis=1)


def sgd_update_probe(bank, c):
    prompt, target, _ = load_pairs(bank, "validation")
    windows = np.load(Path(bank) / "validation/windows.npy", allow_pickle=False)
    indices = np.load(Path(bank) / "probe_indices.npy")[:128]
    if indices.shape != (128,) or len(np.unique(indices)) != 128:
        raise RuntimeError("SGD requires exactly 128 distinct frozen validation probe rows")
    tokens, attention, scoring = assemble(prompt[indices], target[indices], c["bos_token_id"])
    identity = digest({"bank": c["prepared_bank"]["manifest_sha256"],
                       "indices": indices.tolist(), "tokens": hashlib.sha256(tokens.tobytes()).hexdigest(),
                       "loss_mask": hashlib.sha256(scoring.tobytes()).hexdigest(), "precision": "fp32"})
    return {"tokens": tokens, "attention": attention, "scoring": scoring,
            "documents": windows[indices, 0], "identity": identity}


def _primary(model, bank, split, c, export=None):
    from .. import engine
    prompt, target, _ = load_pairs(bank, split, c["evaluation_pairs"])
    tokens, attention, scoring = assemble(prompt, target, c["bos_token_id"])
    value = engine.evaluate(model, tokens, c, export, masks=attention, loss_masks=scoring)
    if export:
        per_example = read_json(export)["sequence_mean_nll"]
        value.update(median_example_nll=float(np.median(per_example)), p95_example_nll=float(np.percentile(per_example, 95)))
    return value


def measure(stage, model, bank, c, weights, cycle):
    from .. import engine
    from ..science.metrics import diversity, probe_diagnostics
    stage = Path(stage)
    stage.mkdir(parents=True, exist_ok=True)
    result = {"generation": cycle, "weights_sha256": weights,
              "validation": _primary(model, bank, "validation", c),
              "test": _primary(model, bank, "test", c, stage / "test_sequences.json"), "secondary": {}}
    if c.get("training_chunk_policy") == "repeat_fixed":
        prompt, target, _ = load_pairs(bank, source_collection(c, 1), c["train_pairs_per_cycle"])
        ids, attention, scoring = assemble(prompt, target, c["bos_token_id"])
        result["fixed_real_anchor"] = engine.evaluate(model, ids, c, masks=attention, loss_masks=scoring)
        result["fixed_real_anchor"].update(source_collection=source_collection(c, 1),
            chunk_manifest_sha256=sha256(bank / source_collection(c, 1) / "manifest.json"))
    for corpus in c["secondary_corpora"]:
        tokens = np.load(bank / "secondary" / corpus / "tokens.npy", mmap_mode="r")
        masks = np.load(bank / "secondary" / corpus / "masks.npy", mmap_mode="r")
        limit = (1 if native_model(c) else 64) if c["kind"] == "smoke" else None
        result["secondary"][corpus] = engine.evaluate(model, tokens[:limit], c, masks=masks[:limit])
    validation, _, _ = load_pairs(bank, "validation")
    indices = np.load(bank / "probe_indices.npy")[:c["probe_count"]]
    prompts = _visible_prompt(validation[indices], c["bos_token_id"])
    probe_c = c
    if c.get("decoding_arms") is not None:
        from ..science.decoding import decoder_config
        probe_c = {**decoder_config(c, "beam"), "generation_purpose": "probe"}
        result["probe_decoding"] = engine.decoding_settings(probe_c)
    values, provenance = engine.generate(stage / "probe", cycle, model, prompts, probe_c, weights)
    diagnostics = smollm.probe_diagnostics(values, c["newline_token_ids"]) if native_model(c) else probe_diagnostics(values)
    result["probe"] = {**diversity(values), **diagnostics,
                       "continuation_tokens_sha256": provenance["continuation_tokens_sha256"]}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(c["model"], revision=c["model_revision"])
    atomic_json(stage / "probe_examples.json", [{"probe_index": i, "prompt": tokenizer.decode(prompts[i, 1:]),
                "generated_continuation": tokenizer.decode(values[i])} for i in range(min(3, len(values)))])
    return result


def benchmark(root, c, bank):
    if c.get("optimizer") == "sgd":
        from .sgd_runtime import benchmark as sgd_benchmark
        return sgd_benchmark(root, c, bank)
    import torch
    from .. import engine
    path = Path(root) / "benchmark.json"
    if path.exists():
        return read_json(path)
    prompts, targets, _ = load_pairs(bank, "cycle_01", max(c["effective_batch"], c["generation_batch"]))
    tokens, attention, scoring = assemble(prompts, targets, c["bos_token_id"])
    model = engine.load_model(c)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"])
    train_seconds = []
    for step in range(2 if native_model(c) else 5):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        engine.sync()
        start = time.perf_counter()
        for offset in range(0, c["effective_batch"], c["microbatch"]):
            rows = np.arange(offset, offset + c["microbatch"]) % len(tokens)
            ids, mask, loss_mask = engine.tensor(tokens[rows]), engine.tensor(attention[rows]), engine.tensor(scoring[rows])
            with engine.autocast(c):
                loss = model(input_ids=ids, attention_mask=mask, labels=ids.masked_fill(~loss_mask.bool(), -100), use_cache=False).loss
            (loss * len(rows) / c["effective_batch"]).backward()
        if not native_model(c):
            optimizer.step()
        engine.sync()
        train_seconds.append(time.perf_counter() - start)
    model.eval()
    generation_seconds = []
    decoder_rates = {}
    modes = ["beam"] if c.get("decoding_arms") is None else ["beam", "greedy", "sampling"]
    from ..science.decoding import decoder_config
    with torch.inference_mode():
        for mode in modes:
            config = decoder_config(c, mode) if c.get("decoding_arms") is not None else c
            durations = []
            for step in range(2 if native_model(c) else 3):
                ids = engine.tensor(_visible_prompt(prompts[:c["generation_batch"]], c["bos_token_id"]))
                engine.sync()
                start = time.perf_counter()
                rng = engine.generation_rng(config, 0, step, {"parent_weights_sha256": "benchmark", "prompt_tokens_sha256": "benchmark"})
                with rng, engine.autocast(config):
                    model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **engine.decoding_settings(config))
                engine.sync()
                durations.append(time.perf_counter() - start)
            decoder_rates[mode] = {"new_tokens_per_second": c["generation_batch"] * 64 / float(np.median(durations[1:])),
                                   "generation_batch_seconds": durations}
            if mode == "beam":
                generation_seconds = durations
    primary = _primary(model, bank, "test", c)
    secondary_tokens = np.load(bank / "secondary/fineweb_edu/tokens.npy", mmap_mode="r")[:64]
    secondary_masks = np.load(bank / "secondary/fineweb_edu/masks.npy", mmap_mode="r")[:64]
    secondary = engine.evaluate(model, secondary_tokens, c, masks=secondary_masks)
    result = {"training_target_tokens_per_second": c["effective_batch"] * 64 / float(np.median(train_seconds[1:] if native_model(c) else train_seconds[2:])),
              "generation_new_tokens_per_second": c["generation_batch"] * 64 / float(np.median(generation_seconds[1:])),
              "primary_target_tokens_per_second": primary["scored_tokens"] / primary["seconds"],
              "secondary_target_tokens_per_second": secondary["scored_tokens"] / secondary["seconds"],
              "training_step_seconds": train_seconds, "generation_batch_seconds": generation_seconds,
              "microbatch": c["microbatch"], "effective_batch": c["effective_batch"], "generation_batch": c["generation_batch"],
              "full_primary_targets": {s: read_json(bank / s / "manifest.json")["target_tokens"] for s in ("validation", "test")},
              "full_secondary_targets": sum(int(np.load(bank / "secondary" / name / "masks.npy", mmap_mode="r")[:, 1:].sum()) for name in c["secondary_corpora"]),
              "benchmark_updates_discarded": 0 if native_model(c) else 5, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"}
    if c.get("decoding_arms") is not None:
        result["decoder_rates"] = decoder_rates
    atomic_json(path, result)
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def forecast(b, c, hourly):
    if c.get("optimizer") == "sgd":
        from .sgd_runtime import forecast as sgd_forecast
        return sgd_forecast(b, c, hourly)
    cycles = c.get("training_cycles", c["full_cycles"])
    targets = 10000 * 64 * cycles
    specs = arm_specs(c)
    stages = len(specs) * cycles
    probe_sets = stages + 1
    generated_targets = sum(mixture_counts(10000, x["synthetic_fraction"])[1] * 64 * cycles for x in specs)
    primary = sum(b["full_primary_targets"].values()) * probe_sets
    secondary = b["full_secondary_targets"] * probe_sets
    seconds = (len(specs) * targets / b["training_target_tokens_per_second"] + generated_targets / b["generation_new_tokens_per_second"] +
               probe_sets * 256 * 64 / b["generation_new_tokens_per_second"] + primary / b["primary_target_tokens_per_second"] +
               secondary / b["secondary_target_tokens_per_second"] + len(specs) * targets / b["primary_target_tokens_per_second"])
    if c.get("decoding_arms") is not None:
        seconds -= generated_targets / b["generation_new_tokens_per_second"]
        seconds += sum(10000 * 64 * cycles / b["decoder_rates"][x["decoder"]]["new_tokens_per_second"] for x in specs)
    if c.get("training_chunk_policy") == "repeat_fixed":
        # Fixed real diagnostics at baseline/each stage and both cold terminal restores.
        seconds += probe_sets * 640000 / b["primary_target_tokens_per_second"] + 900
    # Include checkpoint persistence, immutable private backups and ordinary setup.
    hours = (seconds * 1.5 + stages * 120 + 600) / 3600
    return {"estimated_hours": hours, "estimated_compute_and_running_storage_usd": hours * hourly,
            "unique_training_stages": stages, "epochs": stages,
            "full_generation_stages": sum(x["synthetic_fraction"] == 1 for x in specs) * cycles,
            "generation_stages": sum(x["synthetic_fraction"] > 0 for x in specs) * cycles,
            "probe_sets": probe_sets, "training_target_exposures": len(specs) * targets, "forecast_margin": 1.5,
            "generated_training_continuations": generated_targets // 64,
            "generated_training_tokens": generated_targets}


def _targets(root, cycle, prompt, c, parent_checkpoint, parent_hash, validation, *, create=True):
    from .. import engine
    if c["data_kind"] == "real":
        return None, None
    folder = Path(root) / f"synthetic_for_gen_{cycle:02d}"
    model = None
    if not (folder / "manifest.json").exists():
        if not create:
            raise RuntimeError("Completed synthetic source manifest is missing")
        model = engine.load_model(c, parent_checkpoint)
    try:
        values, manifest = engine.generate(root, cycle - 1, model, _visible_prompt(prompt, c["bos_token_id"]), c,
                                           parent_hash, inject=validation and cycle == 2 and not native_model(c))
    finally:
        del model
        gc.collect()
    return values, sha256(folder / "manifest.json")


def freeze_input(stage, prompt, target, c, lineage, *, create=True):
    from ..data import save_array
    folder = Path(stage) / "training_input"
    meta = {"lineage": lineage, "prompt_content_sha256": hashlib.sha256(np.asarray(prompt, dtype=np.int32).tobytes()).hexdigest(),
            "target_content_sha256": hashlib.sha256(np.asarray(target, dtype=np.int32).tobytes()).hexdigest(),
            "pairs": len(prompt), "scored_targets": len(prompt) * 64, "objective": "conditional_continuation_64"}
    previous = read_json(folder / "manifest.json")
    if previous:
        if {k: previous[k] for k in meta} != meta:
            raise RuntimeError("Training prompt, target or generator provenance changed")
        verify_files(folder, previous["files"])
    else:
        if not create:
            raise RuntimeError("Completed training input manifest is missing")
        tokens, attention, scoring = assemble(prompt, target, c["bos_token_id"])
        for name, values in (("tokens.npy", tokens), ("attention.npy", attention), ("loss_mask.npy", scoring)):
            save_array(folder / name, values)
        meta["files"] = {name: sha256(folder / name) for name in ("tokens.npy", "attention.npy", "loss_mask.npy")}
        atomic_json(folder / "manifest.json", meta)
    return tuple(np.load(folder / name, mmap_mode="r") for name in ("tokens.npy", "attention.npy", "loss_mask.npy"))


def training_input(root, cycle, prompt, real, c, checkpoint, previous, optimizer_hash, validation=False, *, create=True):
    stage = Path(root) / f"seed_42/gen_{cycle:02d}"
    selected_prompt = prompt
    selection = description = None
    if c.get("completion_arms") is not None and not (native_model(c) and c["data_kind"] in ("real", "synthetic")):
        from ..completion_mixing import freeze_selection
        selection, description = freeze_selection(stage, prompt, real, c["synthetic_fraction"],
                                                  c["seed"], cycle, previous, create=create)
        selected_prompt = prompt[selection["synthetic_indices"]]
    generated, generation_hash = (None, None)
    if len(selected_prompt):
        generated, generation_hash = _targets(root, cycle, selected_prompt, c, checkpoint, previous,
                                              validation, create=create)
    lineage = {"initializer_weights_sha256": previous, "optimizer_parent_sha256": optimizer_hash,
               "generator_weights_sha256": previous if generated is not None else None,
               "generation_manifest_sha256": generation_hash, "bank_manifest_sha256": c["prepared_bank"]["manifest_sha256"],
               "chunk_manifest_sha256": sha256(bank_path(root, c) / source_collection(c, cycle) / "manifest.json"),
               "data_kind": c["data_kind"]}
    if c.get("training_chunk_policy") == "repeat_fixed":
        lineage.update(training_chunk_policy="repeat_fixed", source_collection=source_collection(c, cycle),
                       training_epoch=cycle)
    if c.get("decoding_arms") is not None:
        from ..science.decoding import SAMPLING_POLICY, decoding_settings
        lineage.update(training_decoder=c["generation_decoder"], decoding=decoding_settings(c),
                       sampling_policy=SAMPLING_POLICY if c["generation_decoder"] == "sampling" else None)
    target = real if generated is None else generated
    if selection is not None:
        from ..completion_mixing import freeze_targets
        target, mixture_hash = freeze_targets(stage, real, generated, selection, description, generation_hash, create=create)
        lineage.update(mixture_manifest_sha256=mixture_hash, synthetic_fraction=c["synthetic_fraction"],
                       real_pairs=description["real_pairs"], synthetic_pairs=description["synthetic_pairs"])
    arrays = freeze_input(stage, prompt, target, c, lineage, create=create)
    return arrays, lineage


def verify_stage(stage, c, lineage, *, update_probe_identity=None):
    from .selection import training_contract, verify_selection
    stage = Path(stage)
    seal = read_json(stage / "stage_completion.json")
    verify_files(stage, seal["files"])
    training = read_json(stage / "training.json")
    verify_selection(training, read_json(stage / "training_history.json"), c)
    if training["lineage"] != lineage or training["initial_weights_sha256"] != lineage["initializer_weights_sha256"]:
        raise RuntimeError("Completed completion stage lineage differs")
    if c.get("optimizer") == "sgd":
        from ..science.sgd_effects import verify_seal
        recorded = training["update_diagnostics"]
        binding = {"training_contract_sha256": training_contract(c),
                   "input_tokens_sha256": training["input_tokens_sha256"],
                   "initial_weights_sha256": lineage["initializer_weights_sha256"],
                   "optimizer_parent_sha256": lineage["optimizer_parent_sha256"],
                   "lineage_sha256": digest(lineage),
                   "total_optimizer_steps": training["total_optimizer_steps"]}
        seal = verify_seal(stage, expected_weights=training["selected_weights_sha256"],
                           expected_probe_identity=update_probe_identity or recorded["probe_identity"],
                           expected_steps=math.ceil(c["train_pairs_per_cycle"] / c["effective_batch"]),
                           expected_binding=binding)
        if seal != recorded:
            raise RuntimeError("Completed stage diagnostic seal differs from its training record")
    return read_json(stage / "metrics.json")


def run_arm(root, validation=False):
    from .. import engine
    from ..hub import backup
    from ..supervisor import deadline
    root = Path(root)
    with FileLock(str(root / "worker.lock"), timeout=0):
        c, resources = read_json(root / "config.json"), read_json(root / "resources.json")
        validate_config(c)
        parent = root.parent / resources["owner_completion_id"]
        bank = bank_path(root, c)
        restore_bank(c["prepared_bank"], bank)
        _write_fixed(root / "manifest.json", {"config_hash": digest(c), "bank_sha256": c["prepared_bank"]["manifest_sha256"],
                                             "code_sha256": resources["code_sha256"]})
        baseline = read_json(parent / "seed_42/gen_00/metrics.json")
        _write_fixed(root / "seed_42/gen_00/metrics.json", baseline)
        checkpoint, optimizer_state = parent / "seed_42/gen_00/checkpoint", None
        previous = baseline["weights_sha256"]
        val_prompt, val_target, _ = load_pairs(bank, "validation", c["evaluation_pairs"])
        val, val_attention, val_scoring = assemble(val_prompt, val_target, c["bos_token_id"])
        total_steps = c["generations"] * math.ceil(c["train_pairs_per_cycle"] / c["effective_batch"])
        update_probe = sgd_update_probe(bank, c) if c.get("optimizer") == "sgd" else None
        for cycle in range(1, c["generations"] + 1):
            stage = root / f"seed_42/gen_{cycle:02d}"
            prompt, real, _ = load_pairs(bank, source_collection(c, cycle), c["train_pairs_per_cycle"])
            if c["kind"] == "full" and not (stage / "stage_completion.json").exists():
                b = read_json(parent / "benchmark.json")
                seconds = (len(prompt) * 64 / b["training_target_tokens_per_second"] +
                           len(prompt) * 64 / b["primary_target_tokens_per_second"] +
                           sum(b["full_primary_targets"].values()) / b["primary_target_tokens_per_second"] +
                           b["full_secondary_targets"] / b["secondary_target_tokens_per_second"] +
                           c["probe_count"] * 64 / b["generation_new_tokens_per_second"] + 120)
                fraction = c.get("synthetic_fraction", float(c["data_kind"] == "synthetic"))
                generation_rate = (b["decoder_rates"][c["generation_decoder"]]["new_tokens_per_second"]
                                   if c.get("decoding_arms") is not None else b["generation_new_tokens_per_second"])
                seconds += mixture_counts(len(prompt), fraction)[1] * 64 / generation_rate
                if seconds * 1.5 > deadline(resources) - now():
                    atomic_json(root / "status.json", {"state": "budget_paused", "generation": cycle, "time": now()})
                    backup(root)
                    return
            atomic_json(root / "status.json", {"state": "running", "generation": cycle, "stage": "input", "time": now()})
            optimizer_hash = sha256(optimizer_state) if optimizer_state else None
            (tokens, attention, scoring), lineage = training_input(root, cycle, prompt, real, c, checkpoint,
                previous, optimizer_hash, validation, create=not (stage / "stage_completion.json").exists())
            if (stage / "stage_completion.json").exists():
                metrics = verify_stage(stage, c, lineage,
                                       update_probe_identity=update_probe["identity"] if update_probe else None)
            else:
                if shutil.disk_usage(root).free < 3_000_000_000:
                    raise RuntimeError("Insufficient space for completion stage recovery")
                if c["kind"] == "full" and deadline(resources) - now() < 180:
                    atomic_json(root / "status.json", {"state": "budget_paused", "generation": cycle, "time": now()})
                    backup(root)
                    return
                atomic_json(root / "status.json", {"state": "running", "generation": cycle, "stage": "training", "time": now()})
                labels = (np.full(len(tokens), int(c["data_kind"] == "synthetic"), dtype=np.uint8)
                          if c["data_kind"] != "mixed" else np.load(stage / "mixture/source_labels.npy")) if update_probe else None
                def progress(step, **details):
                    if deadline(resources) - now() < 90:
                        raise RuntimeError("SGD training reached its shutdown reserve")
                    atomic_json(root / "status.json", {"state": "running", "stage": "training", "generation": cycle,
                                                       "global_step": step, "time": now()})
                training = engine.train(root, cycle, tokens, val, c, previous, initial_checkpoint=checkpoint, lineage=lineage,
                                        masks=attention, val_masks=val_attention, loss_masks=scoring, val_loss_masks=val_scoring,
                                        optimizer_checkpoint=optimizer_state, total_steps=total_steps,
                                        source_labels=labels, update_probe=update_probe,
                                        progress_callback=progress if update_probe else None,
                                        inject=validation and cycle == 2 and (not native_model(c) or
                                                (c.get("optimizer") == "sgd" and c["arm_id"] == native_profile(c).interruption_arm(c))))
                model = engine.load_model(c, stage / "checkpoint")
                metrics = measure(stage, model, bank, c, training["selected_weights_sha256"], cycle)
                metrics["training"] = training
                metrics["training_evaluation"] = engine.evaluate(model, tokens, c, masks=attention, loss_masks=scoring)
                metrics["input_manifest_sha256"] = sha256(stage / "training_input/manifest.json")
                if c.get("completion_arms") is not None and not (native_model(c) and c["data_kind"] in ("real", "synthetic")):
                    metrics["mixture"] = {k: lineage[k] for k in ("synthetic_fraction", "real_pairs", "synthetic_pairs", "mixture_manifest_sha256")}
                atomic_json(stage / "metrics.json", metrics)
                files = {p.relative_to(stage).as_posix(): sha256(p) for p in stage.rglob("*") if p.is_file()
                         and "resume" not in p.parts and p.name != "stage_completion.json" and not p.name.endswith(".partial")}
                atomic_json(stage / "stage_completion.json", {"files": files, "cycle": cycle})
                (stage / "resume/epoch.pt").unlink(missing_ok=True)
                del model
                gc.collect()
                event(root, "completion_cycle_complete", cycle=cycle, test_perplexity=metrics["test"]["perplexity"])
            if validation and cycle == 1:
                _write_fixed(root / "cycle1_snapshot.json", read_json(stage / "stage_completion.json"))
            previous = metrics["training"]["selected_weights_sha256"]
            checkpoint, optimizer_state = stage / "checkpoint", stage / "optimizer_state/cycle.pt"
            backup(root)
        atomic_json(root / "completion.json", {"complete": True, "cycles": c["generations"], "data_kind": c["data_kind"], "time": now()})
        atomic_json(root / "status.json", {"state": "complete", "time": now()})
        backup(root)


def record_child_snapshot(child):
    from ..hub import backup
    child = Path(child)
    r = read_json(child / "resources.json")
    parent = child.parent / r["owner_completion_id"]
    if child not in child_roots(parent).values() or r["logical_run_id"] != child.name:
        raise RuntimeError("Completion child owner differs")
    with FileLock(str(parent / "backup.lock")):
        snapshots = read_json(parent / "child_snapshots.json", {})
        name = next(k for k, path in child_roots(parent).items() if path == child)
        snapshots[name] = read_json(child / "backup_snapshot.json")
        atomic_json(parent / "child_snapshots.json", snapshots)
        backup(parent)


def verify_completed_arm(child, c):
    """Verify the entire source/model/optimizer chain before a coordinator handoff."""
    child = Path(child)
    completion = read_json(child / "completion.json", {})
    if not completion.get("complete") or completion.get("cycles") != c["generations"]:
        raise RuntimeError("Completion arm has not reached its configured endpoint")
    r = read_json(child / "resources.json")
    parent = child.parent / r["owner_completion_id"]
    bank = bank_path(child, c)
    restore_bank(c["prepared_bank"], bank)
    update_probe_identity = sgd_update_probe(bank, c)["identity"] if c.get("optimizer") == "sgd" else None
    previous = read_json(parent / "seed_42/gen_00/metrics.json")["weights_sha256"]
    optimizer_hash = None
    for cycle in range(1, c["generations"] + 1):
        stage = child / f"seed_42/gen_{cycle:02d}"
        prompt, real, _ = load_pairs(bank, source_collection(c, cycle), c["train_pairs_per_cycle"])
        _, lineage = training_input(child, cycle, prompt, real, c, None, previous, optimizer_hash, create=False)
        metrics = verify_stage(stage, c, lineage, update_probe_identity=update_probe_identity)
        expected_step = cycle * math.ceil(c["train_pairs_per_cycle"] / c["effective_batch"])
        if metrics["training"]["global_step"] != expected_step:
            raise RuntimeError("Completed optimizer step count differs")
        previous = metrics["training"]["selected_weights_sha256"]
        optimizer_hash = sha256(stage / "optimizer_state/cycle.pt")
    snapshot = read_json(child / "backup_snapshot.json")
    index = read_json(child / "artifact_index.json")
    if not snapshot or not index or snapshot["artifact_index_sha256"] != sha256(child / "artifact_index.json"):
        raise BackupPending("Completed child private backup is unsealed")
    if index["files"].get("completion.json", {}).get("sha256") != sha256(child / "completion.json"):
        raise BackupPending("Child backup does not contain its completed endpoint")
    verify_files(child, index["files"])
    return snapshot


def release_coordinator_gpu():
    """Return baseline/benchmark allocator reservations before spawning workers."""
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(root, validation=False):
    from transformers import AutoTokenizer
    from .. import engine
    from ..hub import backup
    from ..pipeline import environment
    from ..supervisor import run_worker
    root = Path(root)
    with FileLock(str(root / "worker.lock"), timeout=0):
        c, r = read_json(root / "config.json"), read_json(root / "resources.json")
        validate_config(c)
        if c.get("optimizer") != "sgd":
            _write_fixed(root / "manifest.json", {"config_hash": digest(c), "code_sha256": r["code_sha256"]})
        atomic_json(root / "status.json", {"state": "running", "phase": "preparation", "time": now()})
        if not (root / "environment.json").exists():
            atomic_json(root / "environment.json", environment())
        bank = bank_path(root, c)
        data = restore_bank(c["prepared_bank"], bank)
        c["bos_token_id"] = data["bos_token_id"]
        if native_model(c):
            c["newline_token_ids"] = data["newline_token_ids"]
        if c.get("optimizer") == "sgd":
            from .sgd_runtime import prepare
            c = prepare(root, c, r, bank)
            _write_fixed(root / "manifest.json", {"config_hash": digest(c), "code_sha256": r["code_sha256"]})
        atomic_json(root / "status.json", {"state": "running", "phase": "baseline", "time": now()})
        if not (root / "seed_42/gen_00/metrics.json").exists():
            if shutil.disk_usage(root).free < ((8_000_000_000 if c["kind"] == "smoke" else 30_000_000_000 if c.get("optimizer") == "sgd" else 150_000_000_000) if native_model(c) else 40_000_000_000):
                raise RuntimeError("Completion comparison requires 40 GB initially free")
            benchmark(root, c, bank)
            model = engine.load_model(c)
            base = root / "seed_42/gen_00"
            model.save_pretrained(base / "checkpoint", safe_serialization=True)
            tokenizer = AutoTokenizer.from_pretrained(c["model"], revision=c["model_revision"])
            tokenizer.save_pretrained(base / "tokenizer")
            baseline = measure(base, model, bank, c, engine.weights_hash(model), 0)
            baseline["checkpoint_files"] = checkpoint_manifest(base / "checkpoint")
            atomic_json(base / "metrics.json", baseline)
            del model
            gc.collect()
        verify_files(root / "seed_42/gen_00/checkpoint", read_json(root / "seed_42/gen_00/metrics.json")["checkpoint_files"])
        # Child processes cannot reuse this coordinator's CUDA allocator cache.
        # Release baseline/benchmark reservations before handing the GPU to them.
        release_coordinator_gpu()
        backup(root)
        roots = child_roots(root)
        _write_fixed(root / "children.json", {name: child.name for name, child in roots.items()})
        for name, child in roots.items():
            arm = arm_config(c, name)
            _write_fixed(child / "config.json", arm)
            if not read_json(child / "completion.json", {}).get("complete"):
                atomic_json(child / "resources.json", {**r, "owner_completion_id": root.name, "logical_run_id": child.name})
                if (root / "requirements-tested.txt").exists() and not (child / "requirements-tested.txt").exists():
                    shutil.copyfile(root / "requirements-tested.txt", child / "requirements-tested.txt")
                atomic_json(root / "status.json", {"state": "running", "phase": name, "active_child": child.name, "time": now()})
                run_worker(child, validation=validation, max_restarts=2)
            if read_json(child / "status.json", {}).get("state") != "complete":
                atomic_json(root / "status.json", {"state": "budget_paused", "active_child": child.name, "time": now()})
                backup(root)
                return
            if not read_json(child / "backup_snapshot.json"):
                backup(child)
            try:
                verify_completed_arm(child, arm)
            except BackupPending:
                backup(child)
                verify_completed_arm(child, arm)
            if not validation and c.get("training_chunk_policy") == "repeat_fixed":
                from .completion_restore import verify_terminal_restore
                verify_terminal_restore(root, child, arm)
            if validation and not native_model(c) and name == next(iter(roots)) and not (root / ".handoff_injected").exists():
                (root / ".handoff_injected").touch()
                event(root, "completion_handoff_interruption")
                raise engine.ValidationInterruption("Injected completion coordinator handoff exit")
        if validation:
            if native_model(c):
                if c.get("optimizer") == "sgd":
                    from .sgd_runtime import finish_benchmark
                    finish_benchmark(root, c)
                    from .sgd_validation import validate_remote
                else:
                    from .smollm_validation import validate_remote
            else:
                from .completion_validation import validate_remote
            validate_remote(root)
        if c.get("optimizer") == "sgd":
            from ..analysis.sgd_effects import report
        else:
            from ..analysis.completion import report
        report(root, for_backup=True)
        atomic_json(root / "completion.json", {"complete": True, "workflow": "completion_comparison", "cycles": c["generations"],
                    "unique_training_stages": c["generations"] * len(roots),
                    "executed_epochs": c["generations"] * len(roots), "time": now()})
        atomic_json(root / "status.json", {"state": "complete", "time": now()})
        try:
            backup(root)
        except BaseException:
            atomic_json(root / "completion.json", {"complete": False, "finalization_pending": True, "time": now()})
            raise


def fetch_comparison(root, resources, metadata_only=False):
    from ..hub import fetch
    root = Path(root)
    receipt = fetch(root.name, resources["repos"], root, metadata_only=metadata_only)
    if read_json(root / "config.json").get("optimizer") == "sgd":
        from ..cloud.sgd_setup import adopt_resolved_config
        adopt_resolved_config(root)
    receipt["children"] = {}
    snapshots = read_json(root / "child_snapshots.json", {})
    for name, ref in snapshots.items():
        child = child_roots(root)[name]
        result = fetch(child.name, resources["repos"], child, metadata_only=metadata_only, revisions=ref["revisions"])
        if result["artifact_index_sha256"] != ref["artifact_index_sha256"]:
            raise RuntimeError("Recovered completion child differs from its pinned snapshot")
        atomic_json(child / "recovery_verified.json", result)
        receipt["children"][name] = result
    atomic_json(root / "recovery_verified.json", receipt)
    return receipt


def verify_recovery(root):
    from ..hub import verify_recovered_files
    root = Path(root)
    receipt = read_json(root / "recovery_verified.json", {})
    if set(receipt.get("children", {})) != set(child_roots(root)):
        raise RuntimeError("Both completion arm restorations are required")
    for name, child in child_roots(root).items():
        result = receipt["children"][name]
        snapshot = read_json(root / "child_snapshots.json")[name]
        if (result["artifact_index_sha256"] != snapshot["artifact_index_sha256"] or result["revisions"] != snapshot["revisions"] or
                result["run_id"] != child.name or snapshot["run_id"] != child.name or result["repos"] != snapshot["repos"]):
            raise RuntimeError("Child recovery snapshot differs")
        index = read_json(child / "artifact_index.json")
        selected = {p: meta for p, meta in index["files"].items() if not result["metadata_only"] or
                    not p.endswith((".npy", ".safetensors", ".pt", "documents.jsonl"))}
        verify_recovered_files(child, selected)
