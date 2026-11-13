from __future__ import annotations

import argparse
import json
import sys


def main():
    p = argparse.ArgumentParser(description="OPT-125M recursive synthetic-data experiment")
    sub = p.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--probe", action="store_true", help="Verify private upload/download checksums")
    diagnostic = sub.add_parser("diagnose-gradients", help="Guarded checkpoint diagnostic; no recursive training")
    diagnostic.add_argument("--config", required=True)
    diagnostic.add_argument("--preflight-only", action="store_true")
    launch = sub.add_parser("launch")
    launch.add_argument("--generations", type=int, choices=range(1, 11), default=None)
    launch_selection = launch.add_mutually_exclusive_group()
    launch_selection.add_argument("--mode", choices=("fresh_base", "continued"), default=None)
    launch_selection.add_argument("--config", help="Explicit experiment configuration")
    launch.add_argument("--gpu", help="Pin a new standalone run to this GPU; requires its own matching acceptance")
    smoke = sub.add_parser("smoke")
    smoke_selection = smoke.add_mutually_exclusive_group()
    smoke_selection.add_argument("--mode", choices=("fresh_base", "continued"), default=None)
    smoke_selection.add_argument("--config", help="Explicit smoke configuration")
    smoke.add_argument("--gpu", help="Pin a new standalone validation to this GPU")
    for name in ("status", "resume", "fetch", "report", "cleanup", "validate", "verify-detached"):
        s = sub.add_parser(name)
        s.add_argument("run_id")
        if name == "fetch":
            s.add_argument("--metadata-only", action="store_true", help="Keep model weights and token arrays in cloud storage")
        if name == "report":
            s.add_argument("--out", help="New analysis destination; published reports are protected")
            s.add_argument("--reference-comparison", help="Completed matched completion comparison ID to include")
    for name in ("worker", "supervisor", "watchdog"):
        s = sub.add_parser(name)
        s.add_argument("--root", required=True)
        s.add_argument("--validation", action="store_true")
    for name in ("prepare", "train", "generate", "evaluate"):
        s = sub.add_parser(name, help="Run one local stage (use launch for guarded remote execution)")
        s.add_argument("--root", required=True)
        s.add_argument("--generation", type=int, default=0)
        s.add_argument("--config", default="configs/full.yaml")
    a = p.parse_args()
    if a.command == "diagnose-gradients":
        from .cloud.gradient_diagnostic import launch
        print(json.dumps(launch(a.config, preflight_only=a.preflight_only), indent=2))
        return
    if a.command in ("prepare", "train", "generate", "evaluate"):
        from .stages import stage
        print(json.dumps(stage(a.command, a.root, a.generation, a.config), indent=2))
        return
    if a.command == "watchdog":
        from .supervisor import watchdog
        return watchdog(a.root)
    if a.command == "supervisor":
        from .supervisor import supervise
        return supervise(a.root, a.validation)
    if a.command == "worker":
        from .common import read_json
        from pathlib import Path
        from .workflow.registry import handler
        run = handler(read_json(Path(a.root) / "config.json"), "worker")
        from .engine import ValidationInterruption
        try:
            return run(a.root, a.validation)
        except ValidationInterruption as e:
            print(str(e), flush=True)
            raise SystemExit(75)
    from .common import PROJECT, STATE, atomic_json, read_json
    if a.command == "report":
        from .report import report
        options = {"reference_comparison": PROJECT / "runs" / a.reference_comparison} if a.reference_comparison else {}
        result = report(PROJECT / "runs" / a.run_id, a.out, **options)
        print(json.dumps(result, indent=2), flush=True)
        return
    if a.command == "validate":
        from .workflow.registry import handler
        root = PROJECT / "runs" / a.run_id
        result = handler(read_json(root / "config.json"), "validate")(root)
        print(json.dumps(result, indent=2), flush=True)
        return
    from . import remote
    if a.command == "preflight":
        result = remote.preflight(probe=a.probe)
    elif a.command in ("launch", "smoke"):
        generations = getattr(a, "generations", 2)
        if a.command == "launch" and generations is None and not a.config:
            generations = 10
        options = {"generations": generations, "smoke": a.command == "smoke", "mode": a.mode or "fresh_base"}
        if a.config:
            options["config_path"] = a.config
        if a.gpu:
            options["gpu"] = a.gpu
        result = remote.launch(**options)
    elif a.command == "status":
        result = remote.status(a.run_id)
    elif a.command == "verify-detached":
        result = remote.verify_detached(a.run_id)
    elif a.command == "resume":
        result = remote.launch(resume_id=a.run_id)
    elif a.command == "cleanup":
        result = remote.cleanup(a.run_id)
    elif a.command == "fetch":
        from .hub import fetch
        r = remote.local_resources(a.run_id)
        out = PROJECT / "runs" / a.run_id
        if read_json(out / "config.json", {}).get("workflow") == "paired":
            from .workflow.paired import fetch_pair
            result = fetch_pair(out, r, metadata_only=a.metadata_only)
        elif read_json(out / "config.json", {}).get("workflow") == "mixed_campaign":
            from .workflow.mixed_campaign import fetch_campaign
            result = fetch_campaign(out, r, metadata_only=a.metadata_only)
        elif read_json(out / "config.json", {}).get("workflow") == "completion_comparison":
            from .workflow.completion import fetch_comparison
            result = fetch_comparison(out, r, metadata_only=a.metadata_only)
        else:
            result = fetch(a.run_id, r["repos"], out, metadata_only=a.metadata_only)
        atomic_json(out / "recovery_verified.json", result)
        bench = read_json(out / "benchmark.json")
        if bench:
            workflow = read_json(out / "config.json", {}).get("workflow")
            if workflow == "completion_comparison":
                from .workflow.completion import benchmark_filename
                benchmark_file = benchmark_filename(read_json(out / "config.json"))
            else:
                benchmark_file = "benchmark_fresh_real.json" if workflow == "fresh_real" else "benchmark.json"
            atomic_json(STATE / benchmark_file, bench)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        raise SystemExit(1)
