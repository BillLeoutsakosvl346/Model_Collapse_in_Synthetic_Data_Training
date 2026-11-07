"""Launch the 80% condition with time to generate its first 8,000 targets."""
from __future__ import annotations

import json
import os


def launch():
    from model_collapse import remote
    from model_collapse.common import PROJECT, config
    from model_collapse.workflow.completion import allocation_id

    path = PROJECT / "comparisons/fineweb_completion64_mixed80/full.yaml"
    settings = config(path)
    if settings.get("completion_arms") != [{"id": "mixed80", "synthetic_fraction": .8}]:
        raise RuntimeError("This launcher requires the recorded 80% completion condition")
    previous = [entry for entry in remote.ledger()["launches"]
                if entry.get("allocation") == allocation_id(settings) and entry.get("kind") == "full"]
    if previous:
        raise RuntimeError("The 80% condition already has a full launch. Use status or resume for " + previous[-1]["run_id"])
    original = remote.verify_detached

    def verify_startup(run_id, *, max_wait_seconds=None):
        # The independent watchdog still bounds the observation to its funded
        # deadline. Generation must finish before a complete training stage exists.
        return original(run_id, max_wait_seconds=max(max_wait_seconds or 0, 900))

    remote.verify_detached = verify_startup
    try:
        return remote.launch(config_path=path, generations=10)
    finally:
        remote.verify_detached = original


def main():
    if os.name == "nt":
        os.environ.setdefault("HF_HUB_CACHE", r"C:\hf")
    print(json.dumps(launch(), indent=2))


if __name__ == "__main__":
    main()
