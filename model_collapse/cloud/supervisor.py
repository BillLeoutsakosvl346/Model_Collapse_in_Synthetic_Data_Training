"""Pod-side process supervision and an independent stdlib-only budget watchdog."""
from __future__ import annotations

import os
import math
import subprocess
import sys
import time
import threading
from pathlib import Path

from ..common import atomic_json, event, now, read_json
from ..runpod_api import Runpod

INTENT_BACKUP_TIMEOUT_SECONDS = 5


def _best_effort(operation, *args, **kwargs):
    """Local evidence must never prevent an independent provider stop."""
    try:
        operation(*args, **kwargs)
        return True
    except Exception:
        return False


def _publish_shutdown_intent(root, resources):
    try:
        for site in Path("/opt/iib-venv/lib").glob("python*/site-packages"):
            sys.path.insert(0, str(site))
        from huggingface_hub import HfApi, get_token
        HfApi(token=os.environ.get("HF_TOKEN") or get_token()).upload_file(
            path_or_fileobj=str(root / "watchdog_exit.json"), path_in_repo=f"{root.name}/watchdog_exit.json",
            repo_id=resources["repos"]["dataset"], repo_type="dataset", commit_message="Record automatic compute shutdown")
    except Exception:
        pass


def deadline(resources):
    # Deadline never resets on a process restart. Includes the entire provisioning interval.
    return resources["created_at"] + resources["session_allowance_usd"] / resources["guard_hourly_usd"] * 3600


def watchdog(root):
    root = Path(root)
    resources = read_json(root / "resources.json")
    api = Runpod()
    cutoff = deadline(resources)
    while True:
        expired = now() >= cutoff
        control_failed = not _best_effort(atomic_json, root / "watchdog.json",
                                        {"pid": os.getpid(), "time": now(), "deadline": cutoff})
        request, supervisor, test_cutoff = None, None, None
        try:
            request = read_json(root / "stop_request.json")
            supervisor = read_json(root / "supervisor.json")
            test_cutoff = read_json(root / "validation_guard_deadline.json")
            if request is not None and (not isinstance(request, dict) or not isinstance(request.get("reason"), str)):
                raise ValueError("Invalid shutdown request")
            if supervisor is not None and (not isinstance(supervisor, dict) or type(supervisor.get("pid")) is not int or supervisor["pid"] <= 0):
                raise ValueError("Invalid supervisor identity")
            if test_cutoff is not None and (not isinstance(test_cutoff, dict)
                    or type(test_cutoff.get("deadline")) not in (int, float)
                    or not math.isfinite(test_cutoff["deadline"])):
                raise ValueError("Invalid validation deadline")
        except (OSError, ValueError, TypeError):
            control_failed = True
            request, supervisor, test_cutoff = None, None, None
        dead = False
        if supervisor:
            try:
                os.kill(supervisor["pid"], 0)
            except ProcessLookupError:
                dead = True
            except (OSError, OverflowError):
                control_failed = True
        tested = test_cutoff and now() >= test_cutoff["deadline"]
        if expired or request or dead or tested or control_failed:
            reason = "budget_exhausted" if expired else "control_io_failure" if control_failed else "validation_budget_guard" if tested else request["reason"] if request else "supervisor_died"
            _best_effort(event, root, "watchdog_shutdown_requested", reason=reason)
            _best_effort(atomic_json, root / "watchdog_exit.json", {"reason": reason, "time": now(), "pod_id": resources["pod_id"]})
            # The Hub SDK can wait indefinitely. Its optional receipt must never delay stop beyond five seconds.
            try:
                receipt = threading.Thread(target=_publish_shutdown_intent, args=(root, resources), daemon=True)
                receipt.start()
                receipt.join(timeout=INTENT_BACKUP_TIMEOUT_SECONDS)
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


def run_worker(root, validation=False, max_restarts=2, heartbeat_root=None):
    """Restart only deliberate validation exits; never owns provider shutdown."""
    root = Path(root)
    attempts = 0
    while True:
        args = [sys.executable, "-m", "model_collapse", "worker", "--root", str(root)]
        if validation:
            args += ["--validation"]
        with (root / "pipeline.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                if heartbeat_root is not None:
                    atomic_json(Path(heartbeat_root) / "supervisor.json", {"pid": os.getpid(), "child_pid": child.pid, "time": now()})
                time.sleep(3)
        if child.returncode == 75 and validation and attempts < max_restarts:
            attempts += 1
            event(root, "validation_worker_resumed", interruption=attempts)
            continue
        if child.returncode:
            raise RuntimeError(f"Worker failed with exit code {child.returncode}; inspect {root.name}/pipeline.log")
        return


def supervise(root, validation=False):
    root = Path(root)
    atomic_json(root / "supervisor.json", {"pid": os.getpid(), "time": now()})
    try:
        workflow = read_json(root / "config.json", {}).get("workflow")
        run_worker(root, validation=validation, max_restarts=1 if workflow == "paired" else 2, heartbeat_root=root)
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
