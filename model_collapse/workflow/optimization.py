"""Explicit existing optimization policies, independent of coordinator plumbing."""


def carries_optimizer(c):
    return c.get("workflow") in ("fresh_real", "completion_arm")


def is_plain_sgd(c):
    return c.get("optimizer", "adamw") == "sgd"


def optimizer_and_scheduler(model, c, total_steps, persistent):
    """Keep historical defaults; opt-in SGD has no adaptive/history state."""
    import torch
    if is_plain_sgd(c):
        if (c.get("momentum") != 0 or c.get("weight_decay") != 0 or
                c.get("learning_rate_policy") != "constant" or c.get("precision") != "fp32"):
            raise ValueError("Plain SGD requires FP32, constant rate, zero momentum and decay")
        return torch.optim.SGD(model.parameters(), lr=c["learning_rate"], momentum=0,
                               weight_decay=0, dampening=0, nesterov=False,
                               foreach=False, fused=False), None
    if c.get("optimizer", "adamw") != "adamw":
        raise ValueError("Unknown optimizer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"], betas=(.9, .999),
                                  eps=1e-8, weight_decay=c["weight_decay"])
    return optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps if persistent else c["epochs"], eta_min=c["min_learning_rate"])


def restore_optimizer(optimizer, scheduler, state, c):
    if is_plain_sgd(c):
        if state.get("optimizer_kind") != "sgd" or state.get("scheduler") is not None or state["optimizer"]["state"]:
            raise RuntimeError("SGD cannot reuse adaptive optimizer or scheduler history")
        if any(group["lr"] != c["learning_rate"] or group.get("momentum", 0) or group.get("weight_decay", 0)
               for group in state["optimizer"]["param_groups"]):
            raise RuntimeError("SGD resume settings changed")
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
