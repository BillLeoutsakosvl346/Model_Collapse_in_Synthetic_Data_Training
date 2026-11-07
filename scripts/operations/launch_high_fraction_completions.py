"""Launch named high-fraction conditions with separate allocation and acceptance."""
from __future__ import annotations

import argparse
import json
import os


PROFILES = {85: ("mixed85", .85), 90: ("mixed90", .90), 95: ("mixed95", .95),
            99: ("mixed99", .99), 99.9: ("mixed99p9", .999)}


def profile(percent):
    if isinstance(percent, bool) or percent not in PROFILES:
        raise ValueError("Only the named high-fraction conditions are supported")
    return PROFILES[percent]


def parse_percent(value):
    number = float(value)
    if number not in PROFILES:
        raise argparse.ArgumentTypeError("Choose 85, 90, 95, 99 or 99.9 percent")
    return int(number) if number.is_integer() else number


def launch_one(percent, *, smoke=False, resume_id=None):
    from model_collapse import remote
    from model_collapse.common import PROJECT, config
    from model_collapse.workflow.completion import allocation_id
    name, fraction = profile(percent)
    kind = "smoke" if smoke else "full"
    path = PROJECT / f"comparisons/fineweb_completion64_{name}/{kind}.yaml"
    settings = config(path)
    if settings["completion_arms"] != [{"id": name, "synthetic_fraction": fraction}]:
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
    parser.add_argument("--percent", type=parse_percent, choices=PROFILES)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    # Provision serially to keep registry and ledger admission atomic; all
    # workers run detached on separate GPUs and overlap after their startups.
    for percent in ([args.percent] if args.percent else [85, 90, 95]):
        print(json.dumps(launch_one(percent, smoke=args.smoke), indent=2), flush=True)


if __name__ == "__main__":
    main()
