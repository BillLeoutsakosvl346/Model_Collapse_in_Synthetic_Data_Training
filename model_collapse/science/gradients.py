"""Read-only loss geometry and isolated AdamW interventions.

These aggregate-gradient steps are diagnostics, not minibatch training replays.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch

from .engine import shifted_nll


def parameters(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def mean_gradient(model, tokens, attention, scoring, *, rows=None, microbatch=8, progress=None):
    """Mean over scored tokens, with no dropout, autocast or parameter updates."""
    model.eval()
    selected = np.arange(len(tokens)) if rows is None else np.asarray(rows, dtype=np.int64)
    if not len(selected) or len(set(selected.tolist())) != len(selected):
        raise ValueError("Gradient source must contain unique nonempty rows")
    if np.any(selected < 0) or np.any(selected >= len(tokens)):
        raise ValueError("Gradient row outside source")
    if attention.shape != tokens.shape or scoring.shape != tokens.shape:
        raise ValueError("Gradient masks do not match tokens")
    if np.any((scoring != 0) & (scoring != 1)) or np.any(scoring > attention):
        raise ValueError("Invalid gradient loss mask")
    count = int(scoring[selected, 1:].sum())
    if count <= 0:
        raise ValueError("No scored gradient targets")
    device = next(model.parameters()).device
    named = parameters(model)
    model.zero_grad(set_to_none=True)
    total = 0.0
    for offset in range(0, len(selected), microbatch):
        ix = selected[offset:offset + microbatch]
        ids = torch.as_tensor(np.array(tokens[ix], copy=True), dtype=torch.long, device=device)
        att = torch.as_tensor(np.array(attention[ix], copy=True), dtype=torch.long, device=device)
        mask = torch.as_tensor(np.array(scoring[ix], copy=True), dtype=torch.long, device=device)
        logits = model(input_ids=ids, attention_mask=att, use_cache=False).logits
        loss = shifted_nll(logits, ids, mask).sum()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite diagnostic loss")
        (loss / count).backward()
        total += float(loss.detach())
        if progress:
            progress(offset + len(ix), len(selected))
    result = {}
    for name, p in named:
        value = torch.zeros_like(p, device="cpu") if p.grad is None else p.grad.detach().cpu().clone()
        if not torch.isfinite(value).all():
            raise RuntimeError("Non-finite diagnostic gradient")
        result[name] = value
    model.zero_grad(set_to_none=True)
    return result, {"loss": total / count, "scored_tokens": count, "pairs": len(selected)}


def combine(left, right, left_weight):
    if set(left) != set(right) or not 0 <= left_weight <= 1:
        raise ValueError("Incompatible gradient combination")
    return {n: left_weight * left[n] + (1 - left_weight) * right[n] for n in left}


def geometry(left, right=None):
    def norm(values):
        return math.sqrt(sum(float(v.double().square().sum()) for v in values.values()))
    ln = norm(left)
    result = {"norm": ln, "parameters": {n: float(v.double().norm()) for n, v in left.items()}}
    if right is not None:
        if set(left) != set(right):
            raise ValueError("Incompatible gradient names")
        rn = norm(right)
        dot = sum(float((left[n].double() * right[n].double()).sum()) for n in left)
        result.update(other_norm=rn, dot=dot,
                      cosine=None if min(ln, rn) <= 1e-12 else max(-1., min(1., dot / (ln * rn))))
    return result


def relative_error(left, right):
    diff = {n: left[n] - right[n] for n in left}
    return geometry(diff)["norm"] / max(geometry(left)["norm"], geometry(right)["norm"], 1e-12)


def optimizer_for(model, saved=None, lr=5e-6, weight_decay=.01):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9, .999), eps=1e-8,
                                  weight_decay=weight_decay)
    if saved is not None:
        optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return optimizer


def apply_gradient(model, gradient, saved, lr, *, weight_decay=.01):
    """One isolated step; explicit zeros preserve AdamW momentum/decay control."""
    optimizer = optimizer_for(model, saved, lr, weight_decay)
    for name, p in parameters(model):
        p.grad = torch.zeros_like(p) if gradient is None else gradient[name].to(p.device).clone()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return optimizer


@torch.inference_mode()
def sequence_losses(model, tokens, attention, scoring, microbatch=8):
    model.eval()
    device = next(model.parameters()).device
    sums, counts = [], []
    for offset in range(0, len(tokens), microbatch):
        ix = slice(offset, offset + microbatch)
        ids = torch.as_tensor(np.array(tokens[ix], copy=True), dtype=torch.long, device=device)
        att = torch.as_tensor(np.array(attention[ix], copy=True), dtype=torch.long, device=device)
        mask = torch.as_tensor(np.array(scoring[ix], copy=True), dtype=torch.long, device=device)
        logits = model(input_ids=ids, attention_mask=att, use_cache=False).logits
        sums.extend(shifted_nll(logits, ids, mask).double().sum(1).cpu().tolist())
        counts.extend(mask[:, 1:].sum(1).cpu().tolist())
    if not sums or not np.isfinite(sums).all() or min(counts) <= 0:
        raise RuntimeError("Invalid diagnostic evaluation")
    return np.asarray(sums), np.asarray(counts)


def stratified_indices(windows, count, seed=20261113):
    """Shuffle document IDs, then round-robin windows so long documents do not dominate."""
    if count > len(windows) or count < 1:
        raise ValueError("Invalid validation probe size")
    rng = np.random.default_rng(seed)
    groups = {int(d): rng.permutation(np.flatnonzero(windows[:, 0] == d)).tolist()
              for d in np.unique(windows[:, 0])}
    order = rng.permutation(list(groups)).tolist()
    chosen = []
    while len(chosen) < count:
        for doc in order:
            if groups[doc]:
                chosen.append(groups[doc].pop())
                if len(chosen) == count:
                    break
    return np.asarray(chosen, dtype=np.int64)


def paired_change(before, after, counts, documents, repetitions=1000, seed=42):
    delta = np.asarray(after) - np.asarray(before)
    docs = np.unique(documents)
    numerator = np.asarray([delta[documents == d].sum() for d in docs])
    denominator = np.asarray([counts[documents == d].sum() for d in docs])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(docs), size=(repetitions, len(docs)))
    changes = numerator[sampled].sum(1) / denominator[sampled].sum(1)
    loss0, loss1 = float(np.sum(before) / np.sum(counts)), float(np.sum(after) / np.sum(counts))
    return {"before_nll": loss0, "after_nll": loss1, "delta_nll": loss1 - loss0,
            "before_perplexity": math.exp(loss0), "after_perplexity": math.exp(loss1),
            "delta_nll_document_bootstrap_95": np.quantile(changes, [.025, .975]).tolist(),
            "documents": len(docs), "scored_tokens": int(np.sum(counts))}
