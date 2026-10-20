from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import posixpath
import shlex
import subprocess
import tarfile
import time
from pathlib import Path

import paramiko
from huggingface_hub import HfApi, get_token

from .common import PROJECT, STATE, atomic_json, config, digest, load_secrets, now, read_json, sha256
from .hub import destinations
from .runpod_api import Runpod

TOTAL_BUDGET = 25.0
SETUP_BUDGET = 3.0
RESERVE = 2.0
VOLUME_HOURLY = 30 * 0.07 / 730
CONTAINER_HOURLY = 20 * 0.10 / 730


def ledger():
    return read_json(STATE / "ledger.json", {"total_budget_usd": TOTAL_BUDGET, "launches": []})


def update_ledger(resources, **updates):
    d = ledger()
    old = next((x for x in d["launches"] if x["launch_id"] == resources["launch_id"]), None)
    if old is None:
        old = {k: resources[k] for k in ("launch_id", "run_id", "kind", "pod_id", "volume_id", "created_at", "guard_hourly_usd")}
        d["launches"].append(old)
    old.update(updates)
    atomic_json(STATE / "ledger.json", d)
    # Private control copy survives a lost local checkout. No secrets in this ledger.
    repos = resources.get("repos")
    if repos:
        HfApi(token=os.environ.get("HF_TOKEN") or get_token()).upload_file(path_or_fileobj=str(STATE / "ledger.json"),
            path_in_repo="_control/ledger.json", repo_id=repos["dataset"], repo_type="dataset", commit_message="Update task cost ledger")
    return d


def accrued(d, kind=None):
    total = 0.0
    for x in d["launches"]:
        if kind and x["kind"] != kind:
            continue
        estimate = x.get("estimated_usd", max(0, now() - x["created_at"]) / 3600 * x["guard_hourly_usd"])
        total += max(estimate, x.get("actual_usd", 0))
    return total


def preflight(probe=False):
    load_secrets()
    r = Runpod()
    account = r.account()
    if account["isAutoPayEnabled"] or account["enableBackupPayments"]:
        raise RuntimeError("Automatic payments must be disabled before launching")
    hub = destinations(probe=probe)
    candidates = r.candidates()
    result = {"account": account, "hugging_face": hub, "gpu_candidates": candidates[:5],
              "total_task_budget_usd": TOTAL_BUDGET, "estimated_accrued_task_usd": accrued(ledger())}
    atomic_json(STATE / "preflight.json", result)
    return result


def source_archive():
    files = [p for folder in ("model_collapse", "configs") for p in (PROJECT / folder).rglob("*")
             if p.is_file() and "__pycache__" not in p.parts]
    files += [PROJECT / "pyproject.toml"]
    code_hash = digest({p.relative_to(PROJECT).as_posix(): sha256(p) for p in sorted(files)})
    archive = STATE / f"code-{code_hash}.tar.gz"
    STATE.mkdir(exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        for p in files:
            tar.add(p, arcname=p.relative_to(PROJECT).as_posix())
    return archive, code_hash


def register_key(api):
    key = STATE / "runpod_ed25519"
    STATE.mkdir(exist_ok=True)
    if not key.exists():
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        private = Ed25519PrivateKey.generate()
        key.write_bytes(private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.OpenSSH, serialization.NoEncryption()))
        key.with_suffix(".pub").write_bytes(private.public_key().public_bytes(serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH) + b" iib-week1")
        if os.name != "nt":
            key.chmod(0o600)
    public = key.with_suffix(".pub").read_text().strip()
    registered = api.request("GET", "/account/ssh-keys")["keys"]
    if public not in registered:
        api.request("PUT", "/account/ssh-keys", {"keys": [*registered, public]})
        if public not in api.request("GET", "/account/ssh-keys")["keys"]:
            raise RuntimeError("SSH key registration did not persist")
    return key


def ssh_connect(pod, key):
    endpoint = pod.get("ssh", {}).get("direct")
    if not endpoint:
        raise RuntimeError("Pod direct SSH is not ready")
    client = paramiko.SSHClient()
    known = STATE / "known_hosts"
    if known.exists():
        client.load_host_keys(str(known))
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(endpoint["host"], port=endpoint["port"], username=endpoint["username"], key_filename=str(key),
                   timeout=20, auth_timeout=20, banner_timeout=20, look_for_keys=False, allow_agent=False)
    client.get_transport().set_keepalive(15)
    client.save_host_keys(str(known))
    return client


def command(client, text, timeout=600):
    _, stdout, stderr = client.exec_command(text, timeout=timeout)
    out, err = stdout.read().decode(errors="replace"), stderr.read().decode(errors="replace")
    status = stdout.channel.recv_exit_status()
    if status:
        # Setup logs contain package output, never secret values or shell xtrace.
        raise RuntimeError(f"Remote command failed ({status}): {(out + err)[-2500:]}")
    return out.strip()


def upload_json(sftp, path, value):
    with sftp.open(path, "w") as f:
        f.write(json.dumps(value))


def local_resources(run_id):
    p = PROJECT / "runs" / run_id / "resources.json"
    value = read_json(p)
    if not value or value.get("run_id") != run_id or not any(x["pod_id"] == value["pod_id"] for x in ledger()["launches"]):
        raise RuntimeError("Run is not in this task's resource registry")
    return value


def launch(generations=10, smoke=False, resume_id=None):
    from .pipeline import forecast
    load_secrets()
    api = Runpod()
    p = preflight(probe=True)
    if any(x["status"] not in ("EXITED", "TERMINATED") for x in api.request("GET", "/pods")["pods"]):
        raise RuntimeError("An existing Pod is active; refusing another GPU")
    choices = api.candidates()
    if not choices:
        raise RuntimeError("No compatible Secure Cloud GPU with Standard network storage is available")
    archive, code_hash = source_archive()
    previous = local_resources(resume_id) if resume_id else None
    c = read_json(PROJECT / "runs" / resume_id / "config.json") if previous else config(PROJECT / "configs" / ("smoke.yaml" if smoke else "full.yaml"))
    if not previous:
        c["generations"] = generations if not smoke else 2
    if previous and previous["code_sha256"] != code_hash and c["kind"] != "smoke":
        raise RuntimeError("Resume requires the original code; restore its recorded commit")
    if previous:
        choices = [x for x in choices if x["data_center"] == previous["data_center"]]
        if not choices:
            raise RuntimeError("No compatible GPU is available beside this run's retained volume")
    choice = choices[0]
    spent = accrued(ledger())
    allowance = min(TOTAL_BUDGET - spent - RESERVE, p["account"]["clientBalance"] - RESERVE)
    if c["kind"] == "smoke":
        allowance = min(allowance, SETUP_BUDGET - accrued(ledger(), "smoke") - 0.15)
    if allowance <= 0.1:
        raise RuntimeError("Insufficient budget after shutdown/storage reserve")
    if c["kind"] == "full":
        b = read_json(STATE / "benchmark.json")
        if not b:
            raise RuntimeError("Run the validated smoke test and fetch its benchmark before launching full data")
        estimate = forecast(b, c["generations"], choice["hourly_usd"] + VOLUME_HOURLY + CONTAINER_HOURLY)
        if estimate["estimated_compute_and_running_storage_usd"] > allowance:
            raise RuntimeError("Conservative full-run estimate exceeds remaining budget; select fewer generations")
    templates = api.request("GET", "/catalog/templates?source=official")["templates"]
    template = next(t for t in templates if t["id"] == "runpod-torch-v280")
    key = register_key(api)
    run_id = resume_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + c["kind"]
    launch_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    local = PROJECT / "runs" / run_id
    local.mkdir(parents=True, exist_ok=True)
    atomic_json(local / "config.json", c)
    print(json.dumps({"provisioning": choice, "image": template["image"], "session_allowance_usd": allowance,
                      "network_volume_gb": 30, "network_volume_usd_per_hour": VOLUME_HOURLY}), flush=True)
    volume = api.request("GET", "/network-volumes/" + previous["volume_id"]) if previous else api.request("POST", "/network-volumes", {
        "name": "iib-" + run_id, "size": 30, "dataCenter": choice["data_center"], "type": "STANDARD"})
    atomic_json(local / "pending_volume.json", {"volume_id": volume["id"], "created_by_task": not bool(previous)})
    check_volume = api.request("GET", "/network-volumes/" + volume["id"])
    if check_volume["size"] != 30 or check_volume["dataCenter"] != choice["data_center"]:
        raise RuntimeError("Volume read-back mismatch")
    created = now()
    try:
        pod = api.request("POST", "/pods", {"name": "iib-" + run_id, "cloud": "SECURE", "image": template["image"],
            "disk": 20, "gpu": {"id": choice["gpu"], "count": 1, "minCudaVersion": "12.8"},
            "dataCenterIds": [choice["data_center"]], "mounts": {"network": [{"volumeId": volume["id"], "path": "/workspace"}]},
            "startSsh": True, "startJupyter": False, "ports": ["22/tcp"]})
    except Exception:
        # Do not retry a timed-out create: reconcile by exact unique name first.
        matches = [x for x in api.request("GET", "/pods")["pods"] if x["name"] == "iib-" + run_id]
        if len(matches) != 1:
            raise RuntimeError("Pod create failed/uncertain; pending_volume.json records the allocated storage. No blind retry.") from None
        pod = matches[0]
    resources = {"run_id": run_id, "launch_id": launch_id, "kind": c["kind"], "pod_id": pod["id"], "volume_id": volume["id"],
                 "data_center": choice["data_center"], "gpu": choice["gpu"], "image": template["image"], "created_at": created,
                 "quoted_gpu_hourly_usd": choice["hourly_usd"], "guard_hourly_usd": (choice["hourly_usd"] + VOLUME_HOURLY + CONTAINER_HOURLY) * 1.15,
                 "session_allowance_usd": allowance, "repos": p["hugging_face"]["repos"], "code_sha256": code_hash,
                 "starting_balance_usd": p["account"]["clientBalance"], "remote_root": "/workspace/runs/" + run_id}
    atomic_json(local / "resources.json", resources)
    client = None
    try:
        update_ledger(resources, state="provisioning")
        ready_by = now() + 600  # Technical provisioning timeout, not a training runtime limit.
        while now() < ready_by:
            pod = api.pod(pod["id"])
            if pod.get("ssh", {}).get("direct") and pod["status"] == "RUNNING":
                try:
                    client = ssh_connect(pod, key)
                    break
                except (OSError, paramiko.SSHException):
                    pass
            time.sleep(8)
        if client is None:
            raise RuntimeError("Pod did not become SSH-ready within provisioning health timeout")
        if pod["gpu"]["id"] != choice["gpu"] or pod["dataCenterId"] != choice["data_center"]:
            raise RuntimeError("Pod read-back configuration mismatch")
        resources["actual_pod_hourly_usd"] = pod["cost"]
        resources["guard_hourly_usd"] = max(resources["guard_hourly_usd"], (pod["cost"] + VOLUME_HOURLY + CONTAINER_HOURLY) * 1.15)
        atomic_json(local / "resources.json", resources)
        command(client, "mkdir -p /workspace/project " + shlex.quote(resources["remote_root"]))
        sftp = client.open_sftp()
        sftp.get_channel().settimeout(120)
        sftp.put(str(archive), "/workspace/project-code.tar.gz")
        command(client, "tar -xzf /workspace/project-code.tar.gz -C /workspace/project")
        atomic_json(local / "launch_phase.json", {"phase": "code_transferred", "time": now()})
        upload_json(sftp, resources["remote_root"] + "/resources.json", resources)
        upload_json(sftp, resources["remote_root"] + "/config.json", c)
        reset = "import pathlib; p=pathlib.Path(" + repr(resources["remote_root"]) + "); [(p/n).unlink(missing_ok=True) for n in ['stop_request.json','supervisor.json','watchdog.json','watchdog_exit.json','validation_guard_deadline.json']]"
        command(client, "python3 -c " + shlex.quote(reset))
        # Secrets live only on ephemeral disk and in process environments, never in archives/backups.
        secret_path = "/run/iib-week1-secrets"
        with sftp.open(secret_path, "w") as f:
            f.write("export RUNPOD_API_KEY=" + shlex.quote(api.key) + "\nexport HF_TOKEN=" + shlex.quote(os.environ.get("HF_TOKEN") or get_token()) + "\n")
        sftp.chmod(secret_path, 0o600)
        prefix = "cd /workspace/project && . /run/iib-week1-secrets && export PYTHONPATH=/workspace/project HF_HOME=/workspace/hf-cache TOKENIZERS_PARALLELISM=false HF_HUB_DOWNLOAD_TIMEOUT=120 && "
        guard_cmd = "setsid python3 -m model_collapse watchdog --root " + shlex.quote(resources["remote_root"]) + " >/workspace/watchdog.log 2>&1 </dev/null &"
        command(client, prefix + "sh -c " + shlex.quote(guard_cmd), timeout=30)
        atomic_json(local / "launch_phase.json", {"phase": "watchdog_started", "time": now()})
        command(client, prefix + "python3 -m venv --system-site-packages .venv && .venv/bin/pip install --quiet -e '.[test]' >/workspace/install.log 2>&1", timeout=600)
        atomic_json(local / "launch_phase.json", {"phase": "dependencies_installed", "time": now()})
        gpu = command(client, prefix + ".venv/bin/python -c 'import torch,json; print(json.dumps({\"torch\":torch.__version__,\"cuda\":torch.cuda.is_available(),\"bf16\":torch.cuda.is_bf16_supported(),\"gpu\":torch.cuda.get_device_name()}))'")
        atomic_json(local / "gpu_check.json", json.loads(gpu))
        if not json.loads(gpu)["cuda"] or not json.loads(gpu)["bf16"]:
            raise RuntimeError("Pod CUDA/BF16 check failed")
        command(client, prefix + ".venv/bin/pip freeze >" + shlex.quote(resources["remote_root"] + "/requirements-tested.txt"))
        supervisor_cmd = ("setsid .venv/bin/python -m model_collapse supervisor --root " + shlex.quote(resources["remote_root"]) +
            (" --validation" if c["kind"] == "smoke" else "") + " >" + shlex.quote(resources["remote_root"] + "/supervisor.log") + " 2>&1 </dev/null &")
        command(client, prefix + "sh -c " + shlex.quote(supervisor_cmd), timeout=30)
        for _ in range(30):
            try:
                live = json.loads(command(client, ". /run/iib-week1-secrets; python3 -c " + shlex.quote(
                    "import json,pathlib; p=pathlib.Path(" + repr(resources["remote_root"]) + "); print(json.dumps({k:json.loads((p/(k+'.json')).read_text()) for k in ['watchdog','supervisor','status']}))")))
                if now() - live["watchdog"]["time"] < 20 and now() - live["supervisor"]["time"] < 20:
                    atomic_json(local / "launch_verified.json", live)
                    update_ledger(resources, state="running")
                    print(json.dumps({"run_id": run_id, "pod_id": pod["id"], "state": "running", "detached_supervisor_and_watchdog_verified": True}), flush=True)
                    return resources
            except (RuntimeError, ValueError):
                pass
            time.sleep(3)
        raise RuntimeError("Supervisor/worker/watchdog did not all become healthy")
    except BaseException:
        api.stop(resources["pod_id"])
        update_ledger(resources, state="failed", estimated_usd=(now() - created) / 3600 * resources["guard_hourly_usd"])
        raise
    finally:
        if client:
            client.close()


def status(run_id):
    r = local_resources(run_id)
    api = Runpod()
    pod = api.pod(r["pod_id"])
    result = {"run_id": run_id, "pod_id": r["pod_id"], "pod_state": pod["status"], "current_pod_hourly_usd": pod["cost"]}
    if pod["status"] == "RUNNING":
        client = ssh_connect(pod, STATE / "runpod_ed25519")
        try:
            text = command(client, "python3 -c " + shlex.quote("import pathlib,json; p=pathlib.Path(" + repr(r["remote_root"]) + "); print(json.dumps({x:json.loads((p/(x+'.json')).read_text()) if (p/(x+'.json')).exists() else None for x in ['status','watchdog','supervisor']}))"))
            result["remote"] = json.loads(text)
            result["log_tail"] = command(client, "tail -c 3000 " + shlex.quote(r["remote_root"] + "/pipeline.log"))
        finally:
            client.close()
    else:
        from huggingface_hub import hf_hub_download
        for name in ("status.json", "watchdog_exit.json"):
            try:
                p = hf_hub_download(r["repos"]["dataset"], f"{run_id}/{name}", repo_type="dataset", force_download=True)
                result[name] = read_json(p)
            except Exception:
                pass
    atomic_json(PROJECT / "runs" / run_id / "last_status.json", result)
    return result


def cleanup(run_id):
    r = local_resources(run_id)
    local = PROJECT / "runs" / run_id
    if not read_json(local / "recovery_verified.json"):
        raise RuntimeError("Fetch and checksum-verify artifacts before removing the recovery volume")
    api = Runpod()
    p = api.pod(r["pod_id"])
    if p["status"] not in ("EXITED", "TERMINATED"):
        api.stop(r["pod_id"])
    api.request("DELETE", "/pods/" + r["pod_id"])
    if any(p["id"] == r["pod_id"] for p in api.request("GET", "/pods")["pods"]):
        raise RuntimeError("Pod deletion not verified")
    api.request("DELETE", "/network-volumes/" + r["volume_id"])
    if any(v["id"] == r["volume_id"] for v in api.request("GET", "/network-volumes")["networkVolumes"]):
        raise RuntimeError("Volume deletion not verified")
    actual = max(0, r["starting_balance_usd"] - api.account()["clientBalance"])
    d = update_ledger(r, state="removed", stopped_at=now(), estimated_usd=(now() - r["created_at"]) / 3600 * r["guard_hourly_usd"], actual_usd=actual)
    atomic_json(local / "cleanup_verified.json", {"pod_removed": True, "volume_removed": True, "actual_account_balance_delta_usd": actual, "ledger": d})
    return {"run_id": run_id, "pod_removed": True, "volume_removed": True, "account_balance_delta_usd": actual}
