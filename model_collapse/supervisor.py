"""Pod-side process supervision and an independent stdlib-only budget watchdog."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from .common import atomic_json, event, now, read_json
from .runpod_api import Runpod


def deadline(resources):
    # Deadline never resets on a process restart. Includes the entire provisioning interval.
    return resources["created_at"] + resources["session_allowance_usd"] / resources["guard_hourly_usd"] * 3600


def watchdog(root):
    root = Path(root)
    resources = read_json(root / "resources.json")
    api = Runpod()
    cutoff = deadline(resources)
    while True:
        atomic_json(root / "watchdog.json", {"pid": os.getpid(), "time": now(), "deadline": cutoff})
        request = read_json(root / "stop_request.json")
        expired = now() >= cutoff
        supervisor = read_json(root / "supervisor.json")
        dead = False
        if supervisor:
            try:
                os.kill(supervisor["pid"], 0)
            except ProcessLookupError:
                dead = True
        test_cutoff = read_json(root / "validation_guard_deadline.json")
        tested = test_cutoff and now() >= test_cutoff["deadline"]
        if expired or request or dead or tested:
            reason = "budget_exhausted" if expired else "validation_budget_guard" if tested else "supervisor_died" if dead else request["reason"]
            event(root, "watchdog_shutdown_requested", reason=reason)
            atomic_json(root / "watchdog_exit.json", {"reason": reason, "time": now(), "pod_id": resources["pod_id"]})
            # Save the shutdown intent off-Pod before calling the API; do not block shutdown on backup errors.
            try:
                # Bootstrap started this process before the venv existed. Reuse its now-installed Hub client.
                for site in Path("/workspace/project/.venv/lib").glob("python*/site-packages"):
                    sys.path.insert(0, str(site))
                from huggingface_hub import HfApi, get_token
                HfApi(token=os.environ.get("HF_TOKEN") or get_token()).upload_file(
                    path_or_fileobj=str(root / "watchdog_exit.json"), path_in_repo=f"{root.name}/watchdog_exit.json",
                    repo_id=resources["repos"]["dataset"], repo_type="dataset", commit_message="Record automatic compute shutdown")
            except Exception:
                pass
            # Keep retrying while this host remains alive, even after a transient API outage.
            while True:
                try:
                    api.stop(resources["pod_id"])
                    return
                except Exception:
                    time.sleep(10)
        time.sleep(3)


def supervise(root, validation=False):
    root = Path(root)
    atomic_json(root / "supervisor.json", {"pid": os.getpid(), "time": now()})
    try:
        attempts = 0
        while True:
            args = [sys.executable, "-m", "model_collapse", "worker", "--root", str(root)]
            if validation:
                args += ["--validation"]
            with (root / "pipeline.log").open("a", encoding="utf-8") as log:
                child = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    atomic_json(root / "supervisor.json", {"pid": os.getpid(), "child_pid": child.pid, "time": now()})
                    time.sleep(3)
            if child.returncode == 75 and validation and attempts < 2:
                attempts += 1
                event(root, "validation_worker_resumed", interruption=attempts)
                continue
            if child.returncode:
                raise RuntimeError(f"Worker failed with exit code {child.returncode}; inspect pipeline.log")
            break
        if validation:
            # A deliberately tiny additional cost allowance exercises the real budget-guard shutdown.
            r = read_json(root / "resources.json")
            atomic_json(root / "validation_guard_deadline.json", {"deadline": now() + 8,
                        "additional_allowance_usd": r["guard_hourly_usd"] * 8 / 3600})
            event(root, "validation_budget_guard_armed", additional_seconds=8)
        else:
            state = read_json(root / "status.json", {}).get("state", "complete")
            atomic_json(root / "stop_request.json", {"reason": state, "time": now()})
        # Stay alive until the watchdog stops the Pod. Never let normal supervisor exit look like a crash.
        while True:
            atomic_json(root / "supervisor.json", {"pid": os.getpid(), "state": "awaiting_shutdown", "time": now()})
            time.sleep(3)
    except BaseException as e:
        atomic_json(root / "status.json", {"state": "failed", "error": str(e), "time": now()})
        atomic_json(root / "stop_request.json", {"reason": "failure", "time": now()})
        raise
