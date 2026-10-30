"""Offline throughput estimates, independent of training and provider clients."""
from __future__ import annotations


def forecast(bench, generations, hourly_usd, *, continued=False, initialization=None, shared_through=1, first_new_cycle=None,
             epochs=5, imported_d1=False):
    policy = initialization or ("continued" if continued else "fresh_base")
    if policy not in ("fresh_base", "continued", "reset_s0"):
        raise ValueError(f"Unknown initialization policy: {policy}")
    if first_new_cycle is not None:
        shared_through = first_new_cycle - 1
    if policy == "reset_s0":
        training_stages = max(0, generations - shared_through)
        full_generation_stages = max(0, generations - shared_through - 1)
    else:
        training_stages = generations + (policy == "fresh_base")
        full_generation_stages = generations
    n = bench["full_split_blocks"]["train"]
    if imported_d1 and policy == "continued":
        full_generation_stages = max(0, full_generation_stages - 1)
    train_seconds = training_stages * epochs * n * 63 / bench["training_target_tokens_per_second"]
    gen_seconds = (full_generation_stages * n + 256) * 64 / bench["generation_new_tokens_per_second"]
    evaluation_stages = (training_stages + 1) if policy == "reset_s0" else (generations + 2)
    evaluation_seconds = evaluation_stages * (6 * bench["full_split_blocks"]["validation"] + bench["full_split_blocks"]["test"]) * 63 / bench["training_target_tokens_per_second"]
    checkpoint_stages = training_stages if policy == "reset_s0" else generations + 1
    seconds = (train_seconds + gen_seconds + evaluation_seconds) * 1.5 + 300 + 180 * checkpoint_stages
    return {"synthetic_generations": generations, "initialization": policy,
            "estimated_hours": seconds / 3600,
            "estimated_compute_and_running_storage_usd": seconds / 3600 * hourly_usd,
            "new_training_stages": training_stages,
            "new_full_generation_stages": full_generation_stages,
            "method": "Warmed real training/five-beam generation; 1.5x compute margin plus setup/checkpoint/upload allowance. Estimate, not a measured full experiment. Imported shared-stage work excluded."}


def forecast_pair(bench, generations, hourly_usd):
    arms = {"continued": forecast(bench, generations, hourly_usd, initialization="continued", imported_d1=True),
            "reset": forecast(bench, generations, hourly_usd, initialization="reset_s0")}
    # Includes a separate bounded allowance for source verification, imports and paired backup.
    hours = sum(a["estimated_hours"] for a in arms.values()) + .25
    return {"arms": arms, "estimated_hours": hours,
            "estimated_compute_and_running_storage_usd": hours * hourly_usd,
            "new_training_stages": 2 * generations - 1, "epochs_per_stage": 5,
            "executed_epochs": 5 * (2 * generations - 1),
            "method": "Sequential matched arms, shared S1/D2 and imported D1; margins include both arms and handoff."}
