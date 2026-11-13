"""Guarded one-Pod deployment of the standalone gradient diagnostic."""
from __future__ import annotations

import json
import os
import shlex
import time
import tarfile
from urllib.parse import quote
from datetime import datetime, timezone
from pathlib import Path

from ..common import PROJECT, STATE, atomic_json, config, digest, load_secrets, now, read_json
from ..runpod_api import Runpod
from . import execution as cloud
from ..workflow.gradient_diagnostic import ALLOCATION


def preflight(c):
    from ..artifacts.gradient_sources import Snapshot
    from .s3_artifacts import client, validate_reference
    load_secrets()
    api = Runpod()
    account = api.account()
    if account["isAutoPayEnabled"] or account["enableBackupPayments"]:
        raise RuntimeError("Automatic/backup payments must remain disabled")
    state = cloud.ledger()
    allocation = state.get("allocations", {}).get(ALLOCATION)
    if not allocation or allocation["budget_usd"] < c.get("task_budget_usd", 2.) or cloud.setup_budget(state) < c.get("setup_ceiling_usd", 5.20):
        raise RuntimeError("Diagnostic allocation/setup authorization is not recorded")
    used_launches = [x for x in state["launches"] if x.get("allocation") == ALLOCATION]
    supplement = c.get("phase") == "layer_norms"
    if used_launches and not supplement:
        raise RuntimeError("This diagnostic already owns a launch; inspect it instead of duplicating compute")
    if supplement:
        original = c.get("main_measurements", {}).get("run_id")
        if not original or not any(x["run_id"] == original and x.get("compute_stopped_at") for x in used_launches):
            raise RuntimeError("Layer supplement requires verified shutdown of the completed main diagnostic")
        if any(x["run_id"] != original for x in used_launches):
            raise RuntimeError("A layer supplement already exists; do not duplicate it")
    used_seconds = sum(max(0,x.get("compute_stopped_at",now())-x.get("compute_started_at",x["created_at"])) for x in used_launches)
    scope = {**state,"launches":used_launches,"volumes":{k:v for k,v in state.get("volumes",{}).items()
        if v.get("allocation")==ALLOCATION or k in {x["volume_id"] for x in used_launches}}}
    remaining_task = allocation["budget_usd"] - cloud.accrued(scope) - allocation.get("reserve_usd",.5)
    cloud.check_active_pods(c, api.request("GET", "/pods")["pods"], state)
    quoted = api.request("GET", "/catalog/gpus/" + quote(c["required_gpu"], safe="") + "?include=AVAILABILITY&product=POD&cloud=SECURE&count=1&minCudaVersion=12.8")
    available = [d for d in quoted.get("dataCenters", []) if d.get("availability") not in ("NONE", "UNAVAILABLE")]
    if not available:
        raise RuntimeError(c["required_gpu"] + " has no available stock; no resources provisioned")
    dc = next((d["id"] for d in available if d["id"] == "EU-RO-1"), available[0]["id"])
    price = quoted["price"]["secure"]
    reserve = 2.
    hours = .50 if supplement else 1.
    if c["max_compute_seconds"] - used_seconds < hours*3600:
        raise RuntimeError("Insufficient combined diagnostic compute time remaining")
    forecast = (price + 20 * .10 / 730 + 30 * .07 / 730) * 1.15 * hours + cloud.historical_running_storage(state, hours) + .10
    if min(min(110., cloud.task_budget(state)) - cloud.accrued(state) - reserve, account["clientBalance"] - reserve, remaining_task) < forecast:
        raise RuntimeError("Diagnostic cannot fit its cumulative/account/task admission after reserve")
    remaining_setup = cloud.setup_budget(state) - cloud.accrued(state, "smoke")
    used_validation = sum(x.get("validation_estimated_usd",0) for x in used_launches)
    remaining_validation = allocation["validation_cap_usd"] - used_validation
    if remaining_setup < remaining_validation or remaining_validation < .03:
        raise RuntimeError("Insufficient historical setup allowance for diagnostic acceptance")
    root = STATE / "gradient-source-preflight"
    parent = Snapshot(c["source_parent"], root / "parent")
    scientific = parent.json("config.json")
    snapshots = parent.json("child_snapshots.json")
    # Verify access and existence/size of every selected large checkpoint/state before renting.
    head_reads, connections = [], {}
    for arm in c["arms"]:
        source = Snapshot(snapshots[arm], root / arm)
        for checkpoint_cycle in (4, 9):
            for suffix in ("checkpoint/model.safetensors", "optimizer_state/cycle.pt"):
                relative = f"seed_42/gen_{checkpoint_cycle:02d}/" + suffix
                meta = source.index[relative]
                if "storage" in meta:
                    store = validate_reference(meta["storage"], meta["sha256"])
                    key = store["bucket"]
                    if key not in connections: connections[key] = client(store)
                    head = connections[key].head_object(Bucket=key, Key=meta["storage"]["key"])
                    if head["ContentLength"] != meta["bytes"]:
                        raise RuntimeError("Source bulk object size differs")
                else:
                    raise RuntimeError("Unexpected source checkpoint storage protocol")
                head_reads.append({"arm": arm, "path": relative, "bytes": meta["bytes"], "sha256": meta["sha256"]})
    bank = Snapshot(scientific["prepared_bank"], root / "bank")
    bank.get("manifest.json")
    if supplement:
        primary = Snapshot(c["main_measurements"], root / "primary")
        if not primary.json("completion.json").get("complete") or primary.json("config.json")["source_parent"] != c["source_parent"]:
            raise RuntimeError("Layer supplement primary source is incomplete or incompatible")
        from ..common import sha256
        original_bundle=primary.json("source_bundle.json")
        for relative in ("model_collapse/science/gradients.py","model_collapse/science/engine.py"):
            if sha256(PROJECT/relative)!=original_bundle["files"][relative]:
                raise RuntimeError("Supplement gradient/loss implementation differs from primary")
    from huggingface_hub import HfApi, get_token
    hf = HfApi(token=os.environ.get("HF_TOKEN") or get_token())
    repos = c["source_parent"]["repos"]
    if any(not hf.repo_info(repo, repo_type=kind).private for kind, repo in repos.items()):
        raise RuntimeError("Diagnostic backup repositories must be private")
    # Small real upload/download establishes result publication before compute.
    probe = "_probes/gradient-diagnostic-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"
    data = json.dumps({"diagnostic": True, "config_sha256": digest(c)}).encode()
    commit = hf.upload_file(path_or_fileobj=data, path_in_repo=probe, repo_id=repos["dataset"], repo_type="dataset", commit_message="Verify gradient diagnostic evidence recovery")
    from huggingface_hub import hf_hub_download
    restored = hf_hub_download(repos["dataset"], probe, repo_type="dataset", revision=commit.oid)
    if Path(restored).read_bytes() != data:
        raise RuntimeError("Diagnostic backup round trip differs")
    return {"account": account, "estimated_accrued_usd": cloud.accrued(state), "price_usd_per_hour": price,
            "data_center": dc, "gpu": quoted["id"], "forecast_upper_usd": forecast,
            "remaining_task_allowance_usd":remaining_task, "remaining_validation_usd":remaining_validation,
            "remaining_compute_seconds":c["max_compute_seconds"]-used_seconds,
            "bulk_head_checks": head_reads, "backup_round_trip": True, "repos": repos,
            "source_model": scientific["model"], "recorded_utc": datetime.now(timezone.utc).isoformat()}


def launch(config_path, *, preflight_only=False):
    from filelock import FileLock
    c = config(config_path)
    with FileLock(str(STATE / "launch.lock"), timeout=0):
        checked = preflight(c)
        atomic_json(STATE / "gradient-diagnostic-preflight.json", checked)
        if preflight_only:
            return checked
        from ..artifacts.gradient_sources import prepare_input_cache
        from ..common import sha256
        input_cache = STATE / "gradient-input-audit"
        cache_archive = STATE / "gradient-input-cache.tar.gz"
        cache_key = digest({"source": c["source_parent"], "config": c["source_config_sha256"], "arms": c["arms"],
                            "cycles": c["cycles"], "verifier": sha256(PROJECT / "model_collapse/artifacts/gradient_sources.py")})
        cache_receipt = read_json(STATE / "gradient-input-cache.json", {})
        if not (cache_archive.is_file() and cache_receipt.get("key") == cache_key and sha256(cache_archive) == cache_receipt.get("sha256")):
            prepare_input_cache(c, input_cache)
            with tarfile.open(cache_archive, "w:gz") as tar:
                for path in (input_cache / ".sources").rglob("*"):
                    if path.is_file() and not path.name.endswith((".pt", ".safetensors")):
                        tar.add(path, arcname=path.relative_to(input_cache).as_posix())
            atomic_json(STATE / "gradient-input-cache.json", {"key": cache_key, "sha256": sha256(cache_archive)})
        archive, code_hash = cloud.source_archive()
        api = Runpod()
        key = cloud.register_key(api)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ("-gradient-layers" if c.get("phase")=="layer_norms" else "-gradient-diagnostic")
        local = PROJECT / "runs" / run_id
        local.mkdir(parents=True)
        atomic_json(local / "config.json", c)
        atomic_json(local / "preflight.json", checked)
        # Image read-back is verified against the preserved source Pod, never a latest tag.
        historical = api.pod("oan7wv9pkt61fi")
        if historical["image"] != c["required_image"]:
            raise RuntimeError("Source Pod image differs from the matched diagnostic image")
        latest = api.request("GET", "/catalog/gpus/" + quote(c["required_gpu"], safe="") + "?include=AVAILABILITY&product=POD&cloud=SECURE&count=1&minCudaVersion=12.8")
        if not any(d.get("id") == checked["data_center"] and d.get("availability") not in ("NONE", "UNAVAILABLE") for d in latest.get("dataCenters", [])):
            raise RuntimeError("Selected GPU placement disappeared before provisioning; no resources created")
        start = now()
        volume = api.request("POST", "/network-volumes", {"name": "iib-" + run_id, "size": 30,
                              "dataCenter": checked["data_center"], "type": "STANDARD"})
        atomic_json(local / "pending_volume.json", {"volume": volume, "created_at": start})
        pod, client, resources = None, None, None
        try:
            payload = {"name": "iib-" + run_id, "image": historical["image"], "disk": 20, "cloud": "SECURE",
                       "gpu": {"id": checked["gpu"], "count": 1, "minCudaVersion": "12.8"},
                       "dataCenterIds": [checked["data_center"]], "startSsh": True, "startJupyter": False,
                       "ports": ["22/tcp"], "env": {"PUBLIC_KEY": key.with_suffix(".pub").read_text().strip()},
                       "mounts": {"network": [{"path": "/workspace", "volumeId": volume["id"]}]}}
            if c.get("provisioning_transport") == "graphql":
                payload["gpu"] = {"id": checked["gpu"], "count": 1, "allowedCudaVersions": ["13.2"]}
            atomic_json(local / "pending_pod_request.json", payload)
            pod_requested_at = now()
            pod = api.create_pod_graphql(payload) if c.get("provisioning_transport") == "graphql" else api.request("POST", "/pods", payload)
            # Start the paid-compute timer before the provider create request.
            created = pod_requested_at
            hourly = checked["price_usd_per_hour"]
            guard = (hourly + 20 * .10 / 730 + 30 * .07 / 730) * 1.15
            resources = {"run_id": run_id, "launch_id": run_id + "-01", "workflow": "gradient_diagnostic",
                         "allocation": ALLOCATION, "kind": "diagnostic", "pod_id": pod["id"], "volume_id": volume["id"],
                         "created_at": created, "volume_created_at": start, "volume_hourly_usd": 30 * .07 / 730,
                         "compute_hourly_usd": (hourly + 20 * .10 / 730) * 1.15, "guard_hourly_usd": guard,
                         "session_allowance_usd": checked["remaining_task_allowance_usd"], "network_volume_gb": 30, "container_disk_gb": 20,
                         "absolute_deadline": created + checked["remaining_compute_seconds"],
                         "acceptance_deadline": created + min(checked["remaining_validation_usd"] / guard * 3600,
                                                              checked["remaining_compute_seconds"]-120) - 20,
                         "data_center": checked["data_center"], "gpu": checked["gpu"], "image": historical["image"],
                         "code_sha256": code_hash, "runtime_sha256": code_hash, "configuration_sha256": digest(c),
                         "starting_balance_usd": checked["account"]["clientBalance"], "repos": checked["repos"],
                         "remote_root": "/workspace/runs/" + run_id}
            cloud.register_resources(resources); cloud.update_ledger(resources, state="provisioning")
            while now() < min(created + 420, resources["acceptance_deadline"] - 90):
                pod = api.pod(resources["pod_id"])
                if pod["status"] == "RUNNING" and pod.get("ssh", {}).get("direct"):
                    try:
                        client = cloud.ssh_connect(pod, key); break
                    except OSError:
                        pass
                time.sleep(5)
            if client is None:
                raise RuntimeError("Diagnostic Pod did not become SSH-ready within its acceptance allowance")
            if pod["gpu"]["id"] != checked["gpu"] or pod["dataCenterId"] != checked["data_center"]:
                raise RuntimeError("Diagnostic hardware read-back differs")
            cloud.command(client, "mkdir -p /workspace/project " + shlex.quote(resources["remote_root"]))
            sftp = client.open_sftp(); sftp.get_channel().settimeout(90)
            sftp.put(str(archive), "/workspace/project-code.tar.gz")
            sftp.put(str(cache_archive), "/workspace/gradient-input-cache.tar.gz")
            cloud.command(client, "tar -xzf /workspace/project-code.tar.gz -C /workspace/project")
            verify_cache = "import hashlib; assert hashlib.sha256(open('/workspace/gradient-input-cache.tar.gz','rb').read()).hexdigest()==" + repr(sha256(cache_archive))
            cloud.command(client, "python3 -c " + shlex.quote(verify_cache))
            cloud.command(client, "tar -xzf /workspace/gradient-input-cache.tar.gz -C " + shlex.quote(resources["remote_root"]))
            verify = "from model_collapse.common import read_json,verify_files; m=read_json('source_bundle.json'); assert m['runtime_sha256']==" + repr(code_hash) + "; verify_files('.',m['files'])"
            cloud.command(client, "cd /workspace/project && python3 -c " + shlex.quote(verify))
            cloud.upload_json(sftp, resources["remote_root"] + "/resources.json", resources)
            cloud.upload_json(sftp, resources["remote_root"] + "/config.json", c)
            cloud.upload_json(sftp, resources["remote_root"] + "/validation_guard_deadline.json", {"deadline": resources["acceptance_deadline"]})
            from huggingface_hub import get_token
            secrets = {"RUNPOD_API_KEY": api.key, "HF_TOKEN": os.environ.get("HF_TOKEN") or get_token(),
                       **{n: os.environ.get(n) for n in ("RUNPOD_S3_ACCESS_KEY_ID", "RUNPOD_S3_SECRET_ACCESS_KEY")}}
            if not all(secrets.values()):
                raise RuntimeError("Required diagnostic transfer credentials missing")
            with sftp.open("/run/iib-gradient-secrets", "w") as handle:
                handle.write("".join("export " + n + "=" + shlex.quote(v) + "\n" for n, v in secrets.items()))
            sftp.chmod("/run/iib-gradient-secrets", 0o600)
            prefix = "cd /workspace/project && . /run/iib-gradient-secrets && export PYTHONPATH=/workspace/project HF_HOME=/workspace/hf-cache TOKENIZERS_PARALLELISM=false && "
            cloud.command(client, prefix + "sh -c " + shlex.quote("setsid python3 -m model_collapse watchdog --root " + resources["remote_root"] + " >/workspace/watchdog.log 2>&1 </dev/null &"), timeout=30)
            cloud.command(client, prefix + "python3 -m venv --system-site-packages /opt/iib-venv && /opt/iib-venv/bin/pip install --progress-bar off -e . >/workspace/install.log 2>&1", timeout=420)
            if now() > resources["acceptance_deadline"] - 90:
                raise RuntimeError("Diagnostic bootstrap used its acceptance allowance")
            supervisor = "setsid /opt/iib-venv/bin/python -m model_collapse supervisor --root " + resources["remote_root"] + " >" + resources["remote_root"] + "/supervisor.log 2>&1 </dev/null &"
            cloud.command(client, prefix + "sh -c " + shlex.quote(supervisor), timeout=30)
            ready = False
            for _ in range(12):
                observe = "import pathlib,json; p=pathlib.Path(" + repr(resources["remote_root"]) + "); print(json.dumps({k:json.loads((p/(k+'.json')).read_text()) if (p/(k+'.json')).exists() else None for k in ['status','watchdog','supervisor']}))"
                observed = json.loads(cloud.command(client, "python3 -c " + shlex.quote(observe), timeout=20))
                if ((observed.get("status") or {}).get("state") == "failed"):
                    raise RuntimeError("Diagnostic failed before detached progress: " + str(observed["status"]))
                if all(observed.get(k) for k in ("status", "watchdog", "supervisor")):
                    ready = True; break
                time.sleep(3)
            if not ready:
                raise RuntimeError("Detached diagnostic supervision/progress was not observed")
            atomic_json(local / "detached_verification.json", observed)
            sftp.close(); client.close(); client = None
            cloud.update_ledger(resources, state="running")
            atomic_json(local / "launch_verified.json", {"detached": True, "runtime_sha256": code_hash, "time": now()})
            return {"run_id": run_id, "pod_id": resources["pod_id"], "hourly_usd": hourly,
                    "deadline": resources["absolute_deadline"], "state": "launched_pending_acceptance_and_progress"}
        except BaseException:
            if client is not None: client.close()
            if pod is not None:
                api.stop(pod["id"])
                if resources:
                    stopped, proof = cloud._provider_shutdown(api, api.pod(pod["id"]), resources)
                    atomic_json(local / "provider_shutdown.json", proof)
                    cloud.update_ledger(resources, state="failed", stopped_at=stopped, provider_shutdown_verified=True)
            elif volume:
                # A failed create can have succeeded remotely; reconcile the unique name first.
                matches = [p for p in api.request("GET", "/pods")["pods"] if p.get("name") == "iib-" + run_id]
                for found in matches:
                    api.stop(found["id"])
                if not matches:
                    api.request("DELETE", "/network-volumes/" + volume["id"])
            raise
