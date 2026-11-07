"""Host admission for the explicitly authorized second full-comparison GPU."""
from __future__ import annotations

import argparse
import json
import os


def check_parallel_full_pods(c, pods, ledger_state):
    from model_collapse.remote import local_resources, same_owner
    from model_collapse.workflow.completion import MIXTURE_ALLOCATION, allocation_id
    from model_collapse.workflow.profiles import protocol_fingerprint

    scope = ledger_state.get("allocations", {}).get(MIXTURE_ALLOCATION, {}).get("parallel_full_scope", {})
    if (c.get("workflow") != "completion_comparison" or c.get("kind") != "full" or
            allocation_id(c) != MIXTURE_ALLOCATION or not scope or
            protocol_fingerprint(c) != scope.get("protocol_fingerprint")):
        raise RuntimeError("Full parallel admission requires the recorded matched-mixture authorization")
    active = [p for p in pods if p["status"] not in ("EXITED", "TERMINATED")]
    if not active:
        return
    if len(active) != 1:
        raise RuntimeError("Only one protected comparison may be active before this launch")
    owner = local_resources(scope["protected_run_id"])
    pod = active[0]
    volumes = {item["volumeId"] for item in pod.get("mounts", {}).get("network", [])}
    if (not same_owner(owner, scope["protected_owner"]) or pod["id"] != owner["pod_id"] or
            pod.get("name") != "iib-" + scope["protected_run_id"] or owner["volume_id"] not in volumes or
            owner.get("kind") != "full" or owner.get("workflow") != "completion_comparison"):
        raise RuntimeError("Active resources differ from the authorized protected comparison")


def launch(resume_id=None):
    from model_collapse import remote
    from model_collapse.common import PROJECT, STATE, config, read_json
    from model_collapse.mixing import certificate_candidates
    from model_collapse.workflow.completion import MIXTURE_ALLOCATION
    from model_collapse.workflow.profiles import require_acceptance

    c = read_json(PROJECT / "runs" / resume_id / "config.json") if resume_id else config(
        PROJECT / "comparisons/fineweb_completion64_mixtures/full.yaml")
    state = remote.ledger()
    scope = state.get("allocations", {}).get(MIXTURE_ALLOCATION, {}).get("parallel_full_scope", {})
    runtime = remote.source_archive()[1]
    if runtime != scope.get("runtime_sha256"):
        raise RuntimeError("Parallel launch runtime differs from the accepted authorization")
    certificate = next((read_json(STATE / name) for name in certificate_candidates(c, runtime)
                        if (STATE / name).exists()), None)
    require_acceptance(c, certificate, runtime)
    existing = [item for item in state["launches"] if item.get("allocation") == MIXTURE_ALLOCATION
                and item.get("kind") == "full"]
    if existing and not resume_id:
        raise RuntimeError("Full mixtures already have a recorded launch. Use status or resume for " + existing[-1]["run_id"])
    if resume_id and (not existing or resume_id != existing[-1]["run_id"]):
        raise RuntimeError("Resume must use this allocation's recorded full comparison")
    original = remote.check_active_pods
    remote.check_active_pods = check_parallel_full_pods
    try:
        if resume_id:
            return remote.launch(resume_id=resume_id)
        return remote.launch(config_path=PROJECT / "comparisons/fineweb_completion64_mixtures/full.yaml", generations=10)
    finally:
        remote.check_active_pods = original


def main():
    if os.name == "nt":
        os.environ.setdefault("HF_HUB_CACHE", r"C:\hf")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", metavar="RUN_ID")
    args = parser.parse_args()
    print(json.dumps(launch(args.resume), indent=2))


if __name__ == "__main__":
    main()
