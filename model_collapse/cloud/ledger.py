"""Pure ledger transitions. File locking, persistence and backup belong to execution."""
from .ownership import same_owner


def apply_update(d, resources, updates, *, volume_hourly):
    updates = dict(updates)
    old = next((x for x in d["launches"] if x["launch_id"] == resources["launch_id"]), None)
    if old is None:
        old = {k: resources[k] for k in ("launch_id", "run_id", "kind", "pod_id", "volume_id", "created_at", "guard_hourly_usd")}
        if resources.get("compute_hourly_usd") is not None:
            old.update(compute_started_at=resources["created_at"], compute_hourly_usd=resources["compute_hourly_usd"])
        for key in ("workflow", "acceptance_campaign", "allocation", "parallel_campaign"):
            if resources.get(key):
                old[key] = resources[key]
        d["launches"].append(old)
    elif not same_owner(old, resources):
        raise RuntimeError("Cost update does not match the registered launch ownership tuple")
    if old.get("compute_started_at") is not None:
        old["compute_hourly_usd"] = max(old["compute_hourly_usd"], resources.get("compute_hourly_usd", 0))
    # A settled compute interval cannot grow when a recovery volume is cleaned up days later.
    if old.get("stopped_at") and updates.get("stopped_at"):
        updates["stopped_at"] = min(old["stopped_at"], updates["stopped_at"])
    if old.get("compute_stopped_at") and updates.get("compute_stopped_at"):
        updates["compute_stopped_at"] = min(old["compute_stopped_at"], updates["compute_stopped_at"])
    if old.get("stopped_at") and "estimated_usd" in old:
        updates.pop("estimated_usd", None)
    old.update(updates)
    if old.get("compute_started_at") is not None and old.get("stopped_at") is not None:
        old.setdefault("compute_stopped_at", old["stopped_at"])
    volume_id = resources.get("volume_id")
    if volume_id:
        # Storage belongs to a volume, not to each Pod launch using that volume.
        volumes = d.setdefault("volumes", {})
        if volume_id not in volumes:
            siblings = [x for x in d["launches"] if x.get("volume_id") == volume_id]
            legacy = [x for x in siblings if "compute_started_at" not in x]
            retained = [x["volume_retained_since"] for x in legacy if x.get("volume_retained_since")]
            volumes[volume_id] = {"run_id": resources["run_id"], "volume_id": volume_id,
                "created_at": min(x["created_at"] for x in siblings),
                # Legacy running estimates already included storage. Do not charge it twice.
                "charge_started_at": min(retained) if retained else (None if legacy else resources.get("volume_created_at", resources["created_at"])),
                "hourly_usd": resources.get("volume_hourly_usd", volume_hourly)}
        volume = volumes[volume_id]
        if volume["run_id"] != resources["run_id"]:
            raise RuntimeError("Volume ledger owner mismatch")
        if updates.get("volume_deleted_at") is not None:
            volume.setdefault("deleted_at", updates["volume_deleted_at"])
        if updates.get("volume_retained_since") and volume.get("charge_started_at") is None:
            volume["charge_started_at"] = updates["volume_retained_since"]
    d["schema_version"] = 2
    return d
