"""Checkpoint selection is part of the scientific training contract."""
from __future__ import annotations

from ..common import digest

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
    return digest({**{k: c[k] for k in keys}, "checkpoint_selection": selection_policy(c)})


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
