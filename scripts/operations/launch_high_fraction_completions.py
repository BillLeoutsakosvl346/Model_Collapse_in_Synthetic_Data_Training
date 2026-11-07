"""Launch named high-fraction conditions; 99% is a separately admitted standalone job."""
from __future__ import annotations

import argparse
import json
import os


def launch_one(percent, *, smoke=False, resume_id=None):
    from model_collapse import remote
    from model_collapse.common import PROJECT, config
    from model_collapse.workflow.completion import allocation_id
    if percent not in (85, 90, 95, 99):
        raise ValueError("Only the named high-fraction conditions are supported")
    kind = "smoke" if smoke else "full"
    path = PROJECT / f"comparisons/fineweb_completion64_mixed{percent}/{kind}.yaml"
    settings = config(path)
    if settings["completion_arms"] != [{"id": f"mixed{percent}", "synthetic_fraction": percent / 100}]:
        raise RuntimeError("Condition configuration differs from its named ratio")
    existing = [entry for entry in remote.ledger()["launches"]
                if entry.get("allocation") == allocation_id(settings) and entry.get("kind") == kind]
    if existing and not resume_id:
        raise RuntimeError("This condition already has a launch; inspect status/resume: " + existing[-1]["run_id"])
    if resume_id and (not existing or existing[-1]['run_id'] != resume_id):
        raise RuntimeError('Resume must match the recorded allocation and condition')
    original = remote.verify_detached
    original_preflight = remote.preflight
    original_setup_budget = remote.SETUP_BUDGET

    def colocated_preflight(*args, **kwargs):
        result = original_preflight(*args, **kwargs)
        # Prefer live stock next to the bulk backup store; do not change the
        # model of GPU or manufacture availability when the catalog is empty.
        preferred = settings['backup_store']['region']
        result['gpu_candidates'] = sorted(result.get('gpu_candidates', []),
            key=lambda item: (item['data_center'] != preferred, item['hourly_usd']))
        return result

    def observe(run_id, *, max_wait_seconds=None):
        return original(run_id, max_wait_seconds=max(max_wait_seconds or 0, 1200))

    remote.verify_detached = observe
    remote.preflight = colocated_preflight
    remote.SETUP_BUDGET = remote.ledger().get('setup_budget_usd', original_setup_budget)
    try:
        if resume_id:
            return remote.launch(resume_id=resume_id)
        return remote.launch(config_path=path, generations=10, smoke=smoke)
    finally:
        remote.verify_detached = original
        remote.preflight = original_preflight
        remote.SETUP_BUDGET = original_setup_budget


def main():
    if os.name == "nt":
        os.environ.setdefault("HF_HUB_CACHE", r"C:\hf")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--percent", type=int, choices=(85, 90, 95, 99))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    # Provision serially to keep registry and ledger admission atomic; all
    # workers run detached on separate GPUs and overlap after their startups.
    for percent in ([args.percent] if args.percent else [85, 90, 95]):
        print(json.dumps(launch_one(percent, smoke=args.smoke), indent=2), flush=True)


if __name__ == "__main__":
    main()
