from __future__ import annotations

import argparse
import json
import sys


def main():
    p = argparse.ArgumentParser(description="OPT-125M recursive synthetic-data experiment")
    sub = p.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--probe", action="store_true", help="Verify private upload/download checksums")
    launch = sub.add_parser("launch")
    launch.add_argument("--generations", type=int, choices=range(1, 11), default=10)
    sub.add_parser("smoke")
    for name in ("status", "resume", "fetch", "report", "cleanup"):
        s = sub.add_parser(name)
        s.add_argument("run_id")
    for name in ("worker", "supervisor", "watchdog"):
        s = sub.add_parser(name)
        s.add_argument("--root", required=True)
        s.add_argument("--validation", action="store_true")
    a = p.parse_args()
    if a.command == "watchdog":
        from .supervisor import watchdog
        return watchdog(a.root)
    if a.command == "supervisor":
        from .supervisor import supervise
        return supervise(a.root, a.validation)
    if a.command == "worker":
        from .pipeline import run
        from .engine import ValidationInterruption
        try:
            return run(a.root, a.validation)
        except ValidationInterruption as e:
            print(str(e), flush=True)
            raise SystemExit(75)
    from .common import PROJECT, STATE, atomic_json, read_json
    from . import remote
    if a.command == "preflight":
        result = remote.preflight(probe=a.probe)
    elif a.command in ("launch", "smoke"):
        result = remote.launch(generations=getattr(a, "generations", 2), smoke=a.command == "smoke")
    elif a.command == "status":
        result = remote.status(a.run_id)
    elif a.command == "resume":
        result = remote.launch(resume_id=a.run_id)
    elif a.command == "cleanup":
        result = remote.cleanup(a.run_id)
    elif a.command == "fetch":
        from .hub import fetch
        r = remote.local_resources(a.run_id)
        out = PROJECT / "runs" / a.run_id
        result = fetch(a.run_id, r["repos"], out)
        atomic_json(out / "recovery_verified.json", result)
        bench = read_json(out / "benchmark.json")
        if bench:
            atomic_json(STATE / "benchmark.json", bench)
    elif a.command == "report":
        from .report import report
        result = report(PROJECT / "runs" / a.run_id, PROJECT / "reports" / a.run_id)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr, flush=True)
        raise SystemExit(1)
