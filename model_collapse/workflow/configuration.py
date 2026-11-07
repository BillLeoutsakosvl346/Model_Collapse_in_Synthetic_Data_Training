"""Shared numeric preflight checks, without imports of training/cloud clients."""
import math


def validate_training_numbers(c):
    # Recorded protocols currently use seed 42 and some stage paths encode it.
    # Enforce that contract until all paths can support a separately named seed.
    if type(c.get("seed")) is not int or c["seed"] != 42:
        raise ValueError("These recorded protocols require integer seed 42")
    for key in ("learning_rate", "min_learning_rate", "weight_decay"):
        value = c.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"{key} must be a finite nonnegative number")
    if not 0 < c["min_learning_rate"] <= c["learning_rate"]:
        raise ValueError("Invalid learning-rate schedule")
    for key in ("epochs", "microbatch", "effective_batch", "eval_batch", "generation_batch"):
        if key in c and (type(c[key]) is not int or c[key] <= 0):
            raise ValueError(f"{key} must be a positive integer")
    if "generations" in c and (type(c["generations"]) is not int or c["generations"] < 0):
        raise ValueError("generations must be a nonnegative integer")
