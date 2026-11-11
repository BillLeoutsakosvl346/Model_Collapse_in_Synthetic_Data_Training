"""Dated, recovery-only adapter for the interrupted 2026-11-10 smoke test.

Uses the unchanged guarded launcher and original worker archive. All six training
stages must already be complete. Detached evidence measures restoration work,
and preserves the earlier detached-training receipt instead of inventing epochs.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
RUN = "20261110T230452Z-smoke-completion64-decoding"
RUNTIME = "7de3d994dfe35e75b5aeabb9a41cffbf54a66871456db0883b666a1d8f63fd6d"


def require_restoration_proof(value, c, runtime):
    from model_collapse.workflow.completion import acceptance_checks
    from model_collapse.workflow.profiles import protocol_fingerprint
    operational = {"detached_progress_verified", "independent_budget_guard_and_provider_stop_verified"}
    if (value.get("source_code_sha256") != runtime or
            value.get("protocol_fingerprint") != protocol_fingerprint(c) or
            any(value.get("checks", {}).get(k) is not True for k in acceptance_checks(c) - operational)):
        raise RuntimeError("Restoration proof does not match the executed smoke protocol")


def main():
    from filelock import FileLock
    from model_collapse import remote
    from model_collapse.common import STATE, atomic_json, read_json, sha256
    from model_collapse.cloud.s3_artifacts import client
    from model_collapse.workflow.completion import child_roots
    from model_collapse.supervisor import deadline

    root = PROJECT / "runs" / RUN
    c = read_json(root / "config.json")
    assert c["kind"] == "smoke" and c["generations"] == 3 and c["experiment_id"] == "fineweb_completion64_decoding"
    assert not read_json(root / "completion.json", {}).get("complete")
    assert remote.source_archive()[1] == RUNTIME
    previous = remote.local_resources(RUN)
    original = read_json(root / "detached_verification.json")
    assert original["pod_id"] == previous["pod_id"] and original["progress_after_disconnect"]
    assert original["observations"][-1]["finite_new_training"] is True
    prior = root / "operational_history" / (previous["launch_id"] + "-detached.json")
    atomic_json(prior, original)
    seals = {}
    for name, child in child_roots(root).items():
        assert read_json(child / "completion.json")["complete"]
        for cycle in range(1, 4):
            file = child / f"seed_42/gen_{cycle:02d}/stage_completion.json"
            seals[file.relative_to(root.parent).as_posix()] = sha256(file)
    with FileLock(str(STATE / "ledger.lock")):
        d = remote.ledger()
        assert d["total_budget_usd"] == 75
        d["setup_budget_usd"] = 4.55
        d["allocations"]["fineweb_completion64_decoding"].update(budget_usd=5., validation_cap_usd=.50)
        atomic_json(STATE / "ledger.json", d)
    atomic_json(PROJECT / "docs/validation/2026-11-10_decoding-restoration-authorization.json", {
        "authorized_at": time.time(), "source": "User approved finishing interrupted restoration then full launch",
        "cumulative_ceiling_usd": 75., "task_cap_usd": 5., "validation_cap_usd": .50,
        "setup_ceiling_usd": 4.55, "reserve_usd": 2., "credit_purchase_authorized": False,
        "original_worker_runtime": RUNTIME, "host_adapter_sha256": sha256(__file__),
        "completed_training_stage_hashes": seals, "original_training_detached_sha256": sha256(prior)})

    def restored_progress(run_id, max_wait_seconds=None):
        assert run_id == RUN
        r = remote.local_resources(RUN)
        assert r["code_sha256"] == RUNTIME and r["launch_id"] != previous["launch_id"]
        api = client({"backend": "runpod_s3", "bucket": r["volume_id"], "region": r["data_center"], "prefix": "restore-observer"})
        prefix = "runs/" + RUN + "/"
        def read(name):
            try:
                obj = api.get_object(Bucket=r["volume_id"], Key=prefix + name)
                return json.loads(obj["Body"].read()), obj["LastModified"].timestamp()
            except api.exceptions.NoSuchKey:
                return {}, 0
        first = {name: read(name + ".json")[0] for name in ("watchdog", "supervisor")}
        cutoff = min(deadline(r) - 15, time.time() + 600)
        while time.time() < cutoff:
            proof, modified = read("remote_validation.json")
            if proof and modified >= r["created_at"]:
                require_restoration_proof(proof, c, RUNTIME)
                last = {name: read(name + ".json")[0] for name in ("watchdog", "supervisor")}
                if not all(last[name].get("time", 0) > first[name].get("time", 0) for name in last):
                    time.sleep(2)
                    continue
                for path, checksum in seals.items():
                    key = "runs/" + path
                    import hashlib
                    body = api.get_object(Bucket=r["volume_id"], Key=key)["Body"].read()
                    if hashlib.sha256(body).hexdigest() != checksum:
                        raise RuntimeError("Previously completed training stage changed during restoration")
                receipt = {"run_id": RUN, "pod_id": r["pod_id"], "launch_id": r["launch_id"],
                    "progress_after_disconnect": True, "new_training_executed": False,
                    "scope": "Restoration-only resume: fresh GPU restoration/evaluation completed after startup SSH disconnected; all six earlier training seals unchanged.",
                    "original_training_detached_sha256": sha256(prior), "completed_training_stage_hashes": seals,
                    "observations": [first, last], "restoration_proof_modified_at": modified,
                    "host_adapter_sha256": sha256(__file__)}
                atomic_json(root / "detached_verification.json", receipt)
                api.put_object(Bucket=r["volume_id"], Key=prefix + "detached_verification.json",
                               Body=json.dumps(receipt).encode())
                return receipt
            time.sleep(3)
        raise RuntimeError("Restoration-only progress did not complete within its independent budget deadline")

    remote.verify_detached = restored_progress
    r = remote.launch(resume_id=RUN)
    atomic_json(PROJECT / "docs/validation/2026-11-10_decoding-restoration-launch.json", {
        k: r.get(k) for k in ("run_id", "launch_id", "pod_id", "volume_id", "created_at", "code_sha256", "session_allowance_usd")})
    print(json.dumps({"restoration_verified": True, "new_training_executed": False, "run_id": RUN, "pod_id": r["pod_id"]}), flush=True)


if __name__ == "__main__":
    main()
