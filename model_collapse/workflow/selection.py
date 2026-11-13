"""Checkpoint selection is part of the scientific training contract."""
from __future__ import annotations

from ..common import digest
from .optimization import carries_optimizer

BEST = "lowest_fixed_real_validation_loss"
FINAL = "final_epoch"


def selection_policy(c):
    value = c.get("checkpoint_selection", BEST)
    if value not in (BEST, FINAL):
        raise ValueError("Unsupported checkpoint_selection")
    return value


def training_contract(c):
    keys = ("epochs", "seed", "learning_rate", "min_learning_rate", "weight_decay",
            "microbatch", "effective_batch", "precision")
    settings = {**{k: c[k] for k in keys}, "checkpoint_selection": selection_policy(c)}
    if c.get("synthetic_fraction", 1.0) != 1.0:
        settings["synthetic_fraction"] = c["synthetic_fraction"]
    if carries_optimizer(c) and c.get("workflow") == "fresh_real":
        from ..fresh_data import data_contract
        settings.update(workflow="fresh_real", data_contract=data_contract(c),
                        optimizer_policy="carry_across_cycles", scheduler_policy="global_cosine_per_step")
    if carries_optimizer(c) and c.get("workflow") == "completion_arm":
        settings.update(workflow="completion_arm", objective="conditional_continuation_64",
                        bank=c["prepared_bank"]["manifest_sha256"], data_kind=c["data_kind"],
                        optimizer_policy="carry_across_cycles", scheduler_policy="global_cosine_per_step")
        if c.get("training_chunk_policy") == "repeat_fixed":
            settings.update(training_chunk_policy="repeat_fixed", fixed_chunk=c["fixed_chunk"])
        if c.get("decoding_arms") is not None:
            from ..science.decoding import decoding_settings, SAMPLING_POLICY
            settings.update(training_decoder=c["generation_decoder"], decoding=decoding_settings(c),
                            generation_batch=c["generation_batch"], sampling_policy=SAMPLING_POLICY)
    if c.get("optimizer") == "sgd":
        from ..science.decoding import decoding_settings
        settings.update(optimizer="sgd", momentum=c["momentum"], foreach=False, fused=False,
                        scheduler_policy="none", learning_rate_policy=c["learning_rate_policy"],
                        training_decoder=decoding_settings(c), purpose=c.get("purpose", "production"),
                        diagnostics=c.get("update_diagnostics"), calibration_reference=c.get("calibration_reference"))
    return digest(settings)


def verify_selection(record, history, c):
    policy = selection_policy(c)
    if record.get("checkpoint_selection", BEST) != policy or len(history) != c["epochs"]:
        raise RuntimeError("Checkpoint selection policy or epoch count changed")
    if record.get("training_contract_sha256", training_contract(c)) != training_contract(c):
        raise RuntimeError("Training settings changed")
    if [row["epoch"] for row in history] != list(range(1, c["epochs"] + 1)):
        raise RuntimeError("Training history has missing or duplicated epochs")
    if policy == FINAL and (record.get("selected_epoch") != c["epochs"] or
                            record.get("terminal_weights_sha256") != record.get("selected_weights_sha256")):
        raise RuntimeError("Final-epoch checkpoint does not match terminal training weights")
