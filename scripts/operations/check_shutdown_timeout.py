"""Real, short GPU shutdown probe with a deliberately stalled optional receipt.

Does not train a model. At most $0.10 of the remaining $3 setup allowance is
available; its task-owned Pod is removed in finally. Research artifacts are untouched.
"""
from __future__ import annotations

import datetime as dt
import json
import shlex
import time

from filelock import FileLock

from model_collapse.common import PROJECT, STATE, atomic_json, now
from model_collapse.remote import (TOTAL_BUDGET, SETUP_BUDGET, RESERVE, accrued, command, ledger, register_key, source_archive,
                                  ssh_connect, update_ledger, upload_json)
from model_collapse.runpod_api import Runpod


def main():
    STATE.mkdir(exist_ok=True)
    with FileLock(str(STATE / "launch.lock"), timeout=0):
        probe()


def probe():
    api = Runpod()
    account = api.account()
    if account["isAutoPayEnabled"] or account["enableBackupPayments"]:
        raise RuntimeError("Automatic payments must remain disabled")
    if (accrued(ledger(), "smoke") + .10 > SETUP_BUDGET
            or accrued(ledger()) + .10 + RESERVE > TOTAL_BUDGET
            or account["clientBalance"] < RESERVE + .10):
        raise RuntimeError("Insufficient reserved setup budget")
    if any(p["status"] not in ("EXITED", "TERMINATED") for p in api.request("GET", "/pods")["pods"]):
        raise RuntimeError("Another Pod is active")
    choices = api.candidates()
    if not choices or choices[0]["hourly_usd"] > 1:
        raise RuntimeError("No affordable guard-test GPU available")
    choice = choices[0]
    template = next(t for t in api.request("GET", "/catalog/templates?source=official")["templates"] if t["id"] == "runpod-torch-v280")
    key = register_key(api)
    archive, source = source_archive()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = stamp + "-guard-probe"
    local = PROJECT / "runs" / run_id
    local.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"guard_probe": choice, "maximum_additional_setup_usd": .10}), flush=True)
    created = now()
    pod = api.request("POST", "/pods", {"name": "iib-" + run_id, "cloud": "SECURE", "image": template["image"],
        "disk": 10, "gpu": {"id": choice["gpu"], "count": 1, "minCudaVersion": "12.8"},
        "dataCenterIds": [choice["data_center"]], "startSsh": True, "startJupyter": False, "ports": ["22/tcp"]})
    r = {"run_id": run_id, "launch_id": stamp, "kind": "smoke", "pod_id": pod["id"], "volume_id": None,
         "created_at": created, "guard_hourly_usd": (choice["hourly_usd"] + 10 * .1 / 730) * 1.15,
         "gpu": choice["gpu"], "source_code_sha256": source}
    atomic_json(local / "resources.json", r)
    update_ledger(r, state="provisioning")
    client = None
    proof = None
    try:
        ready_by = created + min(180, .10 / r["guard_hourly_usd"] * 3600 - 20)
        while now() < ready_by:
            pod = api.pod(r["pod_id"])
            if pod["status"] == "RUNNING" and pod["ssh"]["direct"]:
                try:
                    client = ssh_connect(pod, key)
                    break
                except Exception:
                    pass
            time.sleep(2)
        if client is None:
            raise RuntimeError("Guard probe bootstrap timed out")
        command(client, "mkdir -p /tmp/iib-project /tmp/iib-guard-probe")
        sftp = client.open_sftp()
        sftp.put(str(archive), "/tmp/iib-code.tar.gz")
        command(client, "tar -xzf /tmp/iib-code.tar.gz -C /tmp/iib-project")
        # An intentionally expired tiny allowance exercises the real budget branch.
        upload_json(sftp, "/tmp/iib-guard-probe/resources.json", {**r, "session_allowance_usd": .000001})
        with sftp.open("/run/iib-guard-key", "w") as f:
            f.write("export RUNPOD_API_KEY=" + shlex.quote(api.key) + "\n")
        sftp.chmod("/run/iib-guard-key", 0o600)
        program = ("import time,pathlib,json\nimport model_collapse.supervisor as s\n"
                   "def slow(root, resources):\n"
                   " (root/'sender_started.json').write_text(json.dumps({'started':True}))\n"
                   " time.sleep(60)\n"
                   "s._publish_shutdown_intent=slow\ns.watchdog('/tmp/iib-guard-probe')\n")
        with sftp.open("/tmp/iib-guard-probe/probe.py", "w") as f:
            f.write(program)
        start = now()
        background = "setsid python3 /tmp/iib-guard-probe/probe.py >/tmp/iib-guard-probe/probe.log 2>&1 </dev/null &"
        command(client, "cd /tmp/iib-project && . /run/iib-guard-key && export PYTHONPATH=/tmp/iib-project && sh -c " + shlex.quote(background))
        sender_started = False
        intent = None
        for _ in range(10):
            try:
                with sftp.open("/tmp/iib-guard-probe/sender_started.json") as f:
                    sender_started = json.load(f)["started"]
                with sftp.open("/tmp/iib-guard-probe/watchdog_exit.json") as f:
                    intent = json.load(f)
                break
            except OSError:
                time.sleep(.2)
        client.close()
        client = None
        while now() - start < 30:
            p = api.pod(r["pod_id"])
            if p["status"] == "EXITED":
                break
            time.sleep(1)
        else:
            raise RuntimeError("Stalled receipt delayed budget shutdown")
        if not sender_started or not intent or intent["reason"] != "budget_exhausted":
            raise RuntimeError("Guard probe fault was not exercised")
        proof = {"passed": True, "source_code_sha256": source, "forced_receipt_delay_seconds": 60,
                 "receipt_wait_limit_seconds": 5, "seconds_from_guard_start_to_provider_exited": now() - start,
                 "sender_started": sender_started, "provider_state": p["status"], "shutdown_reason": intent["reason"]}
        atomic_json(local / "proof.json", proof)
    finally:
        if client:
            client.close()
        api.stop(r["pod_id"])
        api.request("DELETE", "/pods/" + r["pod_id"])
        if any(p["id"] == r["pod_id"] for p in api.request("GET", "/pods")["pods"]):
            raise RuntimeError("Guard-probe removal not verified")
        update_ledger(r, state="removed", stopped_at=now(), estimated_usd=(now() - created) / 3600 * r["guard_hourly_usd"])
    print(json.dumps({"run_id": run_id, **proof, "pod_removed": True}), flush=True)


if __name__ == "__main__":
    main()
