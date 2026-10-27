"""Pure conservative cost estimates; provider observations stay in the ledger."""

def accrued_cost(d, timestamp, kind=None, *, default_volume_hourly=30 * 0.07 / 730):
    total = 0.0
    for x in d["launches"]:
        if kind and x["kind"] != kind:
            continue
        if x.get("compute_started_at") is not None:
            end = x.get("compute_stopped_at", x.get("stopped_at", timestamp))
            estimate = max(0, end - x["compute_started_at"]) / 3600 * x["compute_hourly_usd"]
        else:
            # Preserve original receipts and conservative historical estimates byte-for-byte.
            end = x.get("stopped_at", timestamp)
            estimate = x.get("estimated_usd", max(0, end - x["created_at"]) / 3600 * x["guard_hourly_usd"])
        total += max(estimate, x.get("actual_usd", 0))
    volumes = dict(d.get("volumes", {}))
    # Read old ledgers without mutation, coalescing repeated resumes of one retained volume.
    for x in d["launches"]:
        vid = x.get("volume_id")
        if not vid or vid in d.get("volumes", {}) or not x.get("volume_retained_since"):
            continue
        previous = volumes.get(vid)
        start = x["volume_retained_since"]
        if previous is None or start < previous["charge_started_at"]:
            volumes[vid] = {"run_id": x.get("run_id"), "charge_started_at": start,
                "hourly_usd": x.get("volume_hourly_usd", default_volume_hourly), "deleted_at": x.get("volume_deleted_at")}
    allowed_runs = {x.get("run_id") for x in d["launches"] if not kind or x["kind"] == kind}
    for volume in volumes.values():
        if volume.get("charge_started_at") is None or (kind and volume.get("run_id") not in allowed_runs):
            continue
        end = volume.get("deleted_at") or timestamp
        total += max(0, end - volume["charge_started_at"]) / 3600 * volume["hourly_usd"]
    return total

