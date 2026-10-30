"""Bind GPU acceptance to the effective protocol, separately from runtime identity."""
from __future__ import annotations

import copy

from ..common import digest
from .selection import BEST, selection_policy


def protocol_fingerprint(c):
    def arm(value):
        value = copy.deepcopy(value)
        effective_policy = selection_policy(value)
        # Preserve the historical one-epoch acceptance / five-epoch best-validation protocol.
        # Final-epoch acceptance always retains its actual epoch count.
        if effective_policy == BEST and value.get("epochs") in (1, 5):
            value["epochs"] = 5
        # These are explicit smoke scope / operational differences, not model settings.
        for key in ("kind", "generations", "train_blocks", "validation_blocks", "test_blocks",
                    "generation_shard", "experiment_id", "max_session_usd"):
            value.pop(key, None)
        value["checkpoint_selection"] = effective_policy
        if "branch_source" in value:
            value["branch_source"] = {"first_new_cycle": value["branch_source"]["first_new_cycle"]}
        return value
    if c.get("workflow") == "paired":
        return digest({"workflow": "paired", "arms": {k: arm(v) for k, v in c["arms"].items()}})
    return digest(arm(c))


PAIR_CHECKS = {"both_child_audits", "four_five_epoch_training_stages", "epoch_five_checkpoints",
               "exact_shared_prefix", "handoff_restart", "continued_evidence_unchanged",
               "single_owner_and_deadline", "both_private_restores", "detached_progress_verified",
               "independent_budget_guard_and_provider_stop_verified"}


def require_acceptance(c, certificate, runtime):
    if not certificate or certificate.get("source_code_sha256") != runtime:
        raise RuntimeError("Validate the complete smoke test for this source version before a full-data launch")
    if certificate.get("protocol_fingerprint") != protocol_fingerprint(c):
        raise RuntimeError("GPU acceptance does not match the scientific protocol")
    if c.get("workflow") == "paired":
        checks = certificate.get("checks", {})
        if certificate.get("workflow") != "paired" or any(checks.get(k) is not True for k in PAIR_CHECKS):
            raise RuntimeError("GPU acceptance must verify the complete paired workflow")
