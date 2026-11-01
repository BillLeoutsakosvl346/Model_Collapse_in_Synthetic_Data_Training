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
    if c.get("workflow") == "mixed_campaign":
        return digest({"workflow": "mixed_campaign", "jobs": [
            {"name": name, "protocol": arm(c["jobs"][name])}
            for name in ("reset50", "continued20", "reset20")]})
    return digest(arm(c))


PAIR_CHECKS = {"both_child_audits", "four_five_epoch_training_stages", "epoch_five_checkpoints",
               "exact_shared_prefix", "handoff_restart", "continued_evidence_unchanged",
               "single_owner_and_deadline", "both_private_restores", "detached_progress_verified",
               "independent_budget_guard_and_provider_stop_verified"}

OPERATIONAL_CHECKS = {"detached_progress_verified", "independent_budget_guard_and_provider_stop_verified"}

MIXED_COMMON_CHECKS = {
    "exact_mixture_reconstruction", "epoch_five_checkpoints", "weights_changed_each_training_cycle",
    "fixed_real_prompts_and_evaluation", "continuation_only_synthetic_source_blocks",
    "finite_token_weighted_perplexity", "private_hub_download_hashes_verified",
    "restored_checkpoint_tensor_hash_and_evaluation", "direct_shifted_nll_calculation",
    "two_real_worker_exits_and_resume", "partial_generation_shard_reused", *OPERATIONAL_CHECKS,
}
MIXED_CONTINUED_CHECKS = {
    "pinned_previous_real_g0_start", "both_weights_and_data_handoffs",
    "completed_cycle_1_unchanged_after_resume",
}
MIXED_RESET_CHECKS = {
    "exact_pinned_shared_prefix_import", "shared_evidence_scope_preserved",
    "fixed_s0_weights_and_separate_generator_handoffs",
    "completed_first_new_cycle_unchanged_after_resume", "full_shared_mixture_reconstruction",
}
# Keep the historical public name for continued-mixture callers and tests.
MIXED_CHECKS = MIXED_COMMON_CHECKS | MIXED_CONTINUED_CHECKS

CAMPAIGN_CHECKS = {
    "three_child_audits", "six_five_epoch_training_stages", "epoch_five_checkpoints",
    "exact_shared_prefix_and_cycle2_mixture", "two_handoff_restarts", "completed_evidence_unchanged",
    "single_owner_and_deadline", "three_private_restores", *OPERATIONAL_CHECKS,
}


def mixture_checks(c, *, include_operational=True):
    mode = c.get("initialization")
    if mode not in ("continued", "reset_s0"):
        raise RuntimeError("Mixed-data acceptance requires continued or fixed-S0 initialization")
    checks = MIXED_COMMON_CHECKS | (MIXED_RESET_CHECKS if mode == "reset_s0" else MIXED_CONTINUED_CHECKS)
    return checks if include_operational else checks - OPERATIONAL_CHECKS


def _require_mixed_children(c, certificate, runtime, children):
    from ..mixing import mixture_enabled
    records = certificate.get("child_certificates", {})
    for name, child in children.items():
        if not mixture_enabled(child):
            continue
        result = records.get(name, {})
        if (result.get("source_code_sha256") != runtime or
                result.get("protocol_fingerprint") != protocol_fingerprint(child) or
                any(result.get("checks", {}).get(k) is not True
                    for k in mixture_checks(child, include_operational=False))):
            raise RuntimeError(f"GPU acceptance must verify the mixed-data child protocol: {name}")


def _require_shared_mixture(certificate):
    proof = certificate.get("shared20_mixture", {})
    for kind in ("tokens", "selection"):
        values = [proof.get(f"{arm}_{kind}_sha256") for arm in ("continued", "reset")]
        if (not all(isinstance(v, str) and len(v) == 64 and
                    all(ch in "0123456789abcdef" for ch in v) for v in values) or
                values[0] != values[1]):
            raise RuntimeError("GPU acceptance must verify the exact shared cycle-2 mixture")


def require_acceptance(c, certificate, runtime):
    if not certificate or certificate.get("source_code_sha256") != runtime:
        raise RuntimeError("Validate the complete smoke test for this source version before a full-data launch")
    if certificate.get("protocol_fingerprint") != protocol_fingerprint(c):
        raise RuntimeError("GPU acceptance does not match the scientific protocol")
    if c.get("workflow") == "paired":
        checks = certificate.get("checks", {})
        if certificate.get("workflow") != "paired" or any(checks.get(k) is not True for k in PAIR_CHECKS):
            raise RuntimeError("GPU acceptance must verify the complete paired workflow")
        _require_mixed_children(c, certificate, runtime, c["arms"])
        from ..mixing import mixture_enabled
        if any(mixture_enabled(child) for child in c["arms"].values()):
            proof = certificate.get("shared_cycle2_mixture", {})
            _require_shared_mixture({"shared20_mixture": proof})
    elif c.get("workflow") == "mixed_campaign":
        checks = certificate.get("checks", {})
        if (certificate.get("workflow") != "mixed_campaign" or
                certificate.get("unique_training_stages") != 6 or certificate.get("executed_epochs") != 30 or
                any(checks.get(k) is not True for k in CAMPAIGN_CHECKS)):
            raise RuntimeError("GPU acceptance must verify the complete mixed-data campaign")
        _require_mixed_children(c, certificate, runtime, c["jobs"])
        _require_shared_mixture(certificate)
    else:
        from ..mixing import mixture_enabled
        if mixture_enabled(c) and any(certificate.get("checks", {}).get(k) is not True for k in mixture_checks(c)):
            raise RuntimeError("GPU acceptance must verify mixed-data training and recovery")
