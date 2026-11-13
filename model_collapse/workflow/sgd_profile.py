"""Opt-in plain-SGD completion protocol; historical profiles stay unchanged."""
from __future__ import annotations

import copy
import math

from ..common import STATE, digest, read_json
from . import smollm

PROFILE = "smollm2_sgd_greedy_effects_v1"
ALLOCATION = "smollm2_sgd_greedy_effects"
SETUP_ALLOCATION = ALLOCATION + "_setup"
ARMS = [{"id": name, "synthetic_fraction": fraction} for name, fraction in
        (("real", 0.), ("mixed99", .99), ("synthetic", 1.))]
DIAGNOSTICS = {"enabled": True, "policy": "same_graph_sgd_components_v1", "probe_pairs": 128,
               "evaluation_precision": "fp32", "numerical_floor": 1e-6}
CALIBRATION = {"rates": [1e-4, 1e-3, 1e-2, 1e-1], "screen_updates": 32, "confirm_updates": 32,
               "sample_pairs": 4096, "validation_probe_pairs": 1024, "training_probe_pairs": 512,
               "selection_seed_domain": 7731, "validation_seed": 20261113, "max_updates": 192,
               "validation_perplexity_ratio": 1.05, "minimum_train_nll_decrease": 1e-5,
               "tie_nll": 1e-5}
CHECKS = {"twenty_four_sgd_updates", "constant_lr_empty_optimizer_state", "fp32_greedy",
          "source_gradient_reconstruction", "shadow_actual_update_equality", "native_masks_and_lineage",
          "both_fresh_chunk_handoffs", "diagnostic_rng_isolation", "training_replay_seals",
          "terminal_private_restores_all_arms", "detached_progress_verified",
          "independent_watchdog_and_provider_stop_verified"}


def enabled(c):
    return c.get("completion_profile") == PROFILE


def specs(c):
    if c.get("completion_arms") != ARMS:
        raise ValueError("SGD comparison requires real, mixed99 and synthetic in that order")
    return copy.deepcopy(ARMS)


def require_calibration(c):
    if c.get("purpose") == "sgd_calibration":
        return
    ref = c.get("calibration_reference") or {}
    if (c.get("calibration_required") is not False or len(ref.get("receipt_sha256", "")) != 64
            or ref.get("selected_learning_rate") != c.get("learning_rate")
            or ref.get("selection_policy") != "bounded_real_only_192_updates_v1"):
        raise RuntimeError("Freeze and verify the selected SGD learning rate before acceptance or full training")


def apply_calibration(c, receipt, receipt_sha256):
    if not receipt.get("complete") or receipt.get("selection_policy") != "bounded_real_only_192_updates_v1":
        raise RuntimeError("Incomplete or incompatible SGD calibration receipt")
    rate = receipt.get("selected_learning_rate")
    if rate not in CALIBRATION["rates"] or len(receipt_sha256) != 64:
        raise RuntimeError("Invalid selected SGD rate or receipt checksum")
    from .sgd_calibration import calibration_contract
    if receipt.get("contract_sha256") != calibration_contract(c):
        raise RuntimeError("Selected SGD rate belongs to different calibration inputs or code")
    result = copy.deepcopy(c)
    result.update(learning_rate=rate, min_learning_rate=rate, calibration_required=False,
                  calibration_reference={"receipt_sha256": receipt_sha256,
                                         "selected_learning_rate": rate,
                                         "selection_policy": receipt["selection_policy"]})
    return result


def validate(c):
    from .configuration import validate_training_numbers
    from ..science.decoding import decoding_settings
    validate_training_numbers(c)
    specs(c)
    decoding_settings(c)
    expected = {"model": smollm.MODEL, "model_revision": smollm.REVISION,
        "dataset": "HuggingFaceFW/fineweb-edu", "dataset_revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "dataset_config": "sample-10BT", "epochs": 1, "checkpoint_selection": "final_epoch",
        "initialization": "continued", "optimizer_policy": "carry_across_cycles", "full_cycles": 10,
        "full_train_pairs": 10000, "max_sequence_length": 129, "block_size": 64, "new_tokens": 64,
        "generation_decoder": "greedy", "num_beams": 1, "do_sample": False, "repetition_penalty": 1.,
        "optimizer": "sgd", "learning_rate_policy": "constant", "scheduler_policy": "none",
        "momentum": 0., "dampening": 0., "nesterov": False, "maximize": False,
        "foreach": False, "fused": False, "weight_decay": 0., "gradient_clipping": None,
        "microbatch": 32, "effective_batch": 128, "generation_batch": 32, "eval_batch": 32,
        "precision": "fp32", "seed": 42, "storage_backend": "network", "backup_capacity_gb": 30,
        "backup_policy": "runpod_s3_pinned_hf_index", "mixture_selection_policy": "nested_seedsequence_6401_v1"}
    if any(c.get(k) != v for k, v in expected.items()):
        raise ValueError("SGD scientific settings differ from the declared simple protocol")
    if c["min_learning_rate"] != c["learning_rate"] or not math.isfinite(c["learning_rate"]):
        raise ValueError("SGD requires one constant positive learning rate")
    if c.get("update_diagnostics") != DIAGNOSTICS or c.get("calibration") != CALIBRATION:
        raise ValueError("SGD diagnostic or bounded calibration policy differs")
    if c.get("workflow") not in ("completion_comparison", "completion_arm"):
        raise ValueError("SGD uses the shared completion coordinator")
    target = {"full": (10, 10000, None, 256, 60), "smoke": (2, 400, 128, 32, 30)}.get(c.get("kind"))
    if target != tuple(c.get(k) for k in ("generations", "train_pairs_per_cycle", "evaluation_pairs", "probe_count", "network_volume_gb")):
        raise ValueError("SGD full/smoke scope differs")
    if (c.get("generation_shard") != (32 if c["kind"] == "smoke" else 256)
            or c.get("decoding_arms") or c.get("endpoint_arms")
            or c.get("training_chunk_policy", "fresh_per_cycle") != "fresh_per_cycle"):
        raise ValueError("SGD requires fresh chunks and explicit greedy generation")
    if c["kind"] == "smoke" and (c.get("validation_max_seconds") != 1800 or c.get("max_session_usd") != .4):
        raise ValueError("SGD setup requires its combined 30-minute/$0.40 limits")
    if type(c.get("calibration_required")) is not bool:
        raise ValueError("SGD configuration must declare calibration readiness")
    if not c["calibration_required"]:
        require_calibration(c)
    if c["workflow"] == "completion_arm":
        spec = next((v for v in ARMS if v["id"] == c.get("arm_id")), None)
        kind = "real" if spec and spec["synthetic_fraction"] == 0 else "synthetic" if spec and spec["synthetic_fraction"] == 1 else "mixed"
        if not spec or c.get("data_kind") != kind or c.get("synthetic_fraction") != spec["synthetic_fraction"]:
            raise ValueError("SGD child differs from its declared arm")


def training_identity(c):
    """Extra fields only for the opt-in contract; no historical hash changes."""
    return {k: copy.deepcopy(c.get(k)) for k in ("completion_profile", "optimizer", "learning_rate_policy",
        "scheduler_policy", "momentum", "dampening", "nesterov", "maximize", "foreach", "fused",
        "gradient_clipping", "generation_decoder", "num_beams", "do_sample", "update_diagnostics",
        "calibration_reference", "purpose")}


def resolve_inputs(c):
    # The frozen builder's contract does not contain the training/optimizer profile.
    return smollm.resolve_inputs(c)


def resolve_backup(c, api):
    result = copy.deepcopy(c)
    smoke = c.get("kind") == "smoke"
    store = c.get("backup_store") or read_json(STATE / ("smollm2_backup_store.json" if smoke else "sgd_effects_backup_store.json"))
    if not store:
        raise RuntimeError("SGD requires a verified private backup store; setup does not provision full-run storage")
    store = copy.deepcopy(store)
    suffix = "-sgd-greedy-effects-setup"
    if smoke and not store["prefix"].endswith(suffix):
        original = store["prefix"]
        base = original if len(original) + len(suffix) <= 100 else original[:100 - len(suffix) - 9] + "-" + digest(original)[:8]
        store["prefix"] = base + suffix
    from ..cloud.s3_artifacts import validate_store
    validate_store(store)
    volume = api.request("GET", "/network-volumes/" + store["bucket"])
    if (not smoke and volume["size"] < 30) or volume["dataCenter"] != store["region"]:
        raise RuntimeError("SGD private backup volume requires at least 30 GB in its recorded region")
    result["backup_store"] = store
    return result


def required_backup_bytes(c):
    return 5_000_000_000 if c["kind"] == "smoke" else 20_000_000_000


def retention_forecast(c, backup_capacity_bytes, reserve_usd):
    return smollm.retention_forecast(c, backup_capacity_bytes, reserve_usd)


def probe_diagnostics(values, ids):
    return smollm.probe_diagnostics(values, ids)
