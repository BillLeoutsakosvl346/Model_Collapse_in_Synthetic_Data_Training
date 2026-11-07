"""Admission limited to an explicitly authorized set of independent jobs."""
from __future__ import annotations

def scope(c, state):
    name = c.get("parallel_campaign")
    authorization = state.get("parallel_campaigns", {}).get(name, {})
    from ..workflow.completion import allocation_id
    if (not authorization or c.get("workflow") != "completion_comparison" or
            allocation_id(c) not in authorization.get("allocations", {}) or
            authorization.get("max_active_pods") != 3 or
            c.get("backup_store") != authorization.get("backup_store")):
        raise RuntimeError("Parallel job lacks its exact campaign authorization")
    from ..workflow.profiles import protocol_fingerprint
    if protocol_fingerprint(c) != authorization["allocations"][allocation_id(c)]["protocol"]:
        raise RuntimeError("Parallel job protocol differs from its authorization")
    return authorization


def check_active(c, pods, state):
    from ..remote import local_resources, same_owner
    authorization = scope(c, state)
    from ..workflow.completion import allocation_id
    active = [p for p in pods if p["status"] not in ("EXITED", "TERMINATED")]
    if len(active) >= authorization["max_active_pods"]:
        raise RuntimeError("All authorized parallel GPU slots are occupied")
    active_allocations = set()
    for pod in active:
        entries = [x for x in state["launches"] if x.get("pod_id") == pod["id"] and
                   x.get("parallel_campaign") == c["parallel_campaign"]]
        if len(entries) != 1:
            raise RuntimeError("Active Pod is outside the authorized parallel campaign")
        entry = entries[0]
        owner = local_resources(entry["run_id"])
        mounts = {x["volumeId"] for x in pod.get("mounts", {}).get("network", [])}
        if (not same_owner(owner, entry) or owner.get("allocation") not in authorization["allocations"] or
                owner.get("parallel_campaign") != c["parallel_campaign"] or
                owner.get("kind") != c["kind"] or pod.get("name") != "iib-" + owner["run_id"] or
                owner["volume_id"] not in mounts or
                owner.get("code_sha256") != authorization.get("runtime_sha256")):
            raise RuntimeError("Parallel Pod ownership/runtime association differs")
        active_allocations.add(owner["allocation"])
    if allocation_id(c) in active_allocations:
        raise RuntimeError("This condition already has an active GPU")


def reserved_sibling_allowance(c, state, accrued_function):
    """Reserve the other jobs before the first Pod is created.

    Settled costs are already in global accrued spending. Subtract only each
    sibling's remaining authorization, including storage retention reserves.
    """
    from ..workflow.completion import allocation_id
    authorization = scope(c, state)
    current = allocation_id(c)
    reserved = authorization["backup_storage_reserve_usd"]
    for name, spec in authorization["allocations"].items():
        if name == current:
            continue
        launches = [x for x in state["launches"] if x.get("allocation") == name]
        ids = {x["volume_id"] for x in launches}
        subset = {**state, "launches": launches,
                  "volumes": {k: v for k, v in state.get("volumes", {}).items() if k in ids}}
        reserved += max(0.0, spec["budget_usd"] - accrued_function(subset))
    return reserved
