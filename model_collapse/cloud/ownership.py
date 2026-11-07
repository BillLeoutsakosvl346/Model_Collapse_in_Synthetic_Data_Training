"""Pure provider states and launch ownership; no API or local-state access."""
TERMINAL_STATES = frozenset(("EXITED", "TERMINATED"))
OWNER_FIELDS = ("run_id", "launch_id", "pod_id", "volume_id")


def terminal_state(state):
    return state in TERMINAL_STATES


def same_owner(left, right):
    """A Pod id alone cannot authorize acting on a run's recovery volume."""
    return all(key in left and key in right and left[key] == right[key] for key in OWNER_FIELDS)


def latest_session(launches, resources, pod_id):
    """Resolve the most recent session for the exact run/volume/Pod association."""
    candidates = [x for x in launches if x.get("pod_id") == pod_id
                  and x.get("run_id") == resources["run_id"]
                  and ("launch_id" not in resources or x.get("volume_id") == resources["volume_id"])]
    if not candidates:
        raise RuntimeError("No owned provider session for this run/volume/Pod")
    return max(candidates, key=lambda x: x.get("created_at", 0))
