"""Bounded cold restoration of terminal stages before repeated-control shutdown."""
from __future__ import annotations

import gc
import shutil
from pathlib import Path

from ..common import atomic_json, read_json, sha256, verify_files


def verify_terminal_restore(parent, child, c):
    from .. import engine
    from ..hub import fetch
    from ..artifacts.hub import download_snapshot_file
    from ..completion_data import assemble, load_pairs
    from .completion import bank_path
    import torch

    parent, child = Path(parent), Path(child)
    snapshot = read_json(child / "backup_snapshot.json")
    if not snapshot:
        raise RuntimeError("Terminal restore requires a pinned private snapshot")
    audit_path = parent / "terminal_restores" / (child.name + ".json")
    existing = read_json(audit_path)
    if existing:
        if existing["snapshot"] != snapshot:
            raise RuntimeError("Terminal restoration source changed")
        restored = Path(existing["restored_path"])
        verify_files(restored, read_json(restored / "stage_completion.json")["files"])
        return existing
    # Restore into a separate workspace, leaving live stages and ownership untouched.
    out = parent.parent.parent / "completion_terminal_restore" / child.name
    receipt = fetch(child.name, snapshot["repos"], out, metadata_only=True, revisions=snapshot["revisions"])
    if receipt["artifact_index_sha256"] != snapshot["artifact_index_sha256"]:
        raise RuntimeError("Terminal restoration index differs from the sealed snapshot")
    index = read_json(out / "artifact_index.json")
    prefix = f"seed_42/gen_{c['generations']:02d}/"
    selected = {p: m for p, m in index["files"].items() if p.startswith(prefix)}
    if not selected or prefix + "stage_completion.json" not in selected:
        raise RuntimeError("Terminal stage is absent from the private index")
    for relative, meta in selected.items():
        src = download_snapshot_file(child.name, snapshot["repos"], relative, meta, snapshot["revisions"])
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        verify_files(out, {relative: meta})
    stage = out / prefix
    verify_files(stage, read_json(stage / "stage_completion.json")["files"])
    training = read_json(stage / "training.json")
    model = engine.load_model(c, stage / "checkpoint")
    if engine.weights_hash(model) != training["selected_weights_sha256"]:
        raise RuntimeError("Restored terminal tensors differ")
    state = torch.load(stage / "optimizer_state/cycle.pt", map_location="cpu", weights_only=False)
    if state["model_weights_sha256"] != training["selected_weights_sha256"] or state["global_step"] != training["global_step"]:
        raise RuntimeError("Restored optimizer does not belong to the terminal model")
    prompt, target, _ = load_pairs(bank_path(child, c), "test", c["evaluation_pairs"])
    tokens, attention, scoring = assemble(prompt, target, c["bos_token_id"])
    measured = engine.evaluate(model, tokens, c, masks=attention, loss_masks=scoring)
    original = read_json(stage / "metrics.json")["test"]
    error = abs(measured["loss"] - original["loss"])
    if error > 1e-6 or measured["scored_tokens"] != original["scored_tokens"]:
        raise RuntimeError("Cold-restored terminal evaluation differs")
    result = {"snapshot": snapshot, "scope": "complete_terminal_stage_only",
              "restored_path": str(stage), "verified_files": len(selected),
              "verified_bytes": sum(m["bytes"] for m in selected.values()),
              "optimizer_sha256": sha256(stage / "optimizer_state/cycle.pt"),
              "weights_sha256": training["selected_weights_sha256"],
              "test": measured, "test_loss_absolute_error": error}
    atomic_json(audit_path, result)
    del model, state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
