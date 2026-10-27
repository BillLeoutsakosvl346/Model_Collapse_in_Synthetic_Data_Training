from __future__ import annotations

import datetime as dt
import hashlib
import io
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
        if resources.get("compute_hourly_usd") is not None:
            old.update(compute_started_at=resources["created_at"], compute_hourly_usd=resources["compute_hourly_usd"])
        d["launches"].append(old)
    elif not same_owner(old, resources):
        raise RuntimeError("Cost update does not match the registered launch ownership tuple")
    if old.get("compute_started_at") is not None:
        old["compute_hourly_usd"] = max(old["compute_hourly_usd"], resources.get("compute_hourly_usd", 0))
    # A settled compute interval cannot grow when a recovery volume is cleaned up days later.
    if old.get("stopped_at") and updates.get("stopped_at"):
        updates["stopped_at"] = min(old["stopped_at"], updates["stopped_at"])
    if old.get("compute_stopped_at") and updates.get("compute_stopped_at"):
        updates["compute_stopped_at"] = min(old["compute_stopped_at"], updates["compute_stopped_at"])
    if old.get("stopped_at") and "estimated_usd" in old:
        updates.pop("estimated_usd", None)
    old.update(updates)
    if old.get("compute_started_at") is not None and old.get("stopped_at") is not None:
        old.setdefault("compute_stopped_at", old["stopped_at"])
    volume_id = resources.get("volume_id")
    if volume_id:
        # Storage belongs to a volume, not to each Pod launch using that volume.
        volumes = d.setdefault("volumes", {})
        if volume_id not in volumes:
            siblings = [x for x in d["launches"] if x.get("volume_id") == volume_id]
            legacy = [x for x in siblings if "compute_started_at" not in x]
            retained = [x["volume_retained_since"] for x in legacy if x.get("volume_retained_since")]
            volumes[volume_id] = {"run_id": resources["run_id"], "volume_id": volume_id,
                "created_at": min(x["created_at"] for x in siblings),
                # Legacy running estimates already included storage. Do not charge it twice.
                "charge_started_at": min(retained) if retained else (None if legacy else resources.get("volume_created_at", resources["created_at"])),
                "hourly_usd": resources.get("volume_hourly_usd", VOLUME_HOURLY)}
        volume = volumes[volume_id]
        if volume["run_id"] != resources["run_id"]:
            raise RuntimeError("Volume ledger owner mismatch")
        if updates.get("volume_deleted_at") is not None:
            volume.setdefault("deleted_at", updates["volume_deleted_at"])
        if updates.get("volume_retained_since") and volume.get("charge_started_at") is None:
            volume["charge_started_at"] = updates["volume_retained_since"]
    d["schema_version"] = 2
    atomic_json(STATE / "ledger.json", d)
    # Private control copy survives a lost local checkout. No secrets in this ledger.
    repos = resources.get("repos")
    if repos:
        HfApi(token=os.environ.get("HF_TOKEN") or get_token()).upload_file(path_or_fileobj=str(STATE / "ledger.json"),
            path_in_repo="_control/ledger.json", repo_id=repos["dataset"], repo_type="dataset", commit_message="Update task cost ledger")
    return d


def accrued(d, kind=None):
    from .cloud.budget import accrued_cost
    return accrued_cost(d, now(), kind, default_volume_hourly=VOLUME_HOURLY)


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
    files = [p for p in (PROJECT / "model_collapse").rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")
             and not any(part.startswith(".") for part in p.relative_to(PROJECT).parts)]
    files += [PROJECT / "pyproject.toml"]
    code_hash = digest({p.relative_to(PROJECT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in sorted(files)})
    archive = STATE / f"code-{code_hash}.tar.gz"
    STATE.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {"schema_version": 1, "runtime_sha256": code_hash, "git_commit": commit,
                "runtime_identity_excludes": ["configs", "experiments", "comparisons", "docs", "reports"],
                "files": {p.relative_to(PROJECT).as_posix(): sha256(p) for p in sorted(files)}}
    if archive.exists():
        # Runtime identity can stay unchanged across documentation-only commits. Never
        # overwrite the exact archive that an earlier acceptance test actually deployed.
        with tarfile.open(archive, "r:gz") as existing:
            member = existing.extractfile("source_bundle.json")
            saved = json.load(member) if member is not None else None
        if not saved or saved.get("runtime_sha256") != code_hash:
            raise RuntimeError("Existing source archive lacks the matching bundle manifest")
        atomic_json(STATE / "source_bundle.json", saved)
        return archive, code_hash
    atomic_json(STATE / "source_bundle.json", manifest)
    with tarfile.open(archive, "w:gz") as tar:
        for p in files:
            tar.add(p, arcname=p.relative_to(PROJECT).as_posix())
        payload = (json.dumps(manifest, indent=2) + "\n").encode()
        info = tarfile.TarInfo("source_bundle.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
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


def same_owner(left, right):
    """A Pod id alone cannot authorize acting on a run's recovery volume."""
    keys = ("run_id", "launch_id", "pod_id", "volume_id")
    return all(key in left and key in right and left[key] == right[key] for key in keys)


def register_resources(resources):
    run_id = resources["run_id"]
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("Invalid run id")
    registry = STATE / "registry" / run_id
    atomic_json(registry / "resources.json", resources)
    atomic_json(registry / "launches" / (resources["launch_id"] + ".json"), resources)
    atomic_json(PROJECT / "runs" / run_id / "resources.json", resources)


def local_resources(run_id):
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("Invalid run id")
    registry_path = STATE / "registry" / run_id / "resources.json"
    value = read_json(registry_path)
    entries = [x for x in ledger()["launches"] if x.get("run_id") == run_id]
    if value is None:
        # Upgrade a legacy control record only when it is the latest complete ledger tuple.
        candidate = read_json(PROJECT / "runs" / run_id / "resources.json")
        latest = max(entries, key=lambda x: x.get("created_at", 0), default={})
        if not candidate or not same_owner(candidate, latest):
            raise RuntimeError("Run is not in this task's resource registry")
        value = candidate
        register_resources(value)
    if value.get("run_id") != run_id or not any(same_owner(x, value) for x in entries):
        raise RuntimeError("Run is not in this task's resource registry")
    return value


def verify_start_checkpoint(c):
    if c.get("initialization") not in ("continued", "reset_s0"):
        return
    start = c["start_checkpoint"]
    entries = HfApi().get_paths_info(start["repo"], [f"{start['path']}/{name}" for name in start["files"]],
                                    revision=start["revision"])
    if len(entries) != len(start["files"]):
        raise RuntimeError("Pinned initial checkpoint is incomplete or inaccessible")
    for entry in entries:
        if getattr(entry, "lfs", None) and entry.lfs.sha256 != start["files"][entry.path.rsplit("/", 1)[-1]]:
            raise RuntimeError("Starting checkpoint metadata checksum mismatch")


def launch(generations=10, smoke=False, resume_id=None, mode="fresh_base", config_path=None):
    from filelock import FileLock
    STATE.mkdir(exist_ok=True)
    with FileLock(str(STATE / "launch.lock"), timeout=0):
        return _launch(generations, smoke, resume_id, mode, config_path)


def _launch(generations=10, smoke=False, resume_id=None, mode="fresh_base", config_path=None):
    from .pipeline import forecast
    if resume_id:
        local = PROJECT / "runs" / resume_id
        if read_json(local / "completion.json", {}).get("complete") or read_json(local / "status.json", {}).get("state") == "complete":
            raise RuntimeError("Completed runs cannot resume; create a new scientific branch run instead")
    if config_path is not None and mode not in (None, "fresh_base"):
        raise ValueError("--config and --mode are mutually exclusive")
    if mode not in (None, "fresh_base", "continued", "reset_s0"):
        raise ValueError("Unknown experiment mode")
    previous = local_resources(resume_id) if resume_id else None
    suffix = "-continued" if mode == "continued" else ""
    if previous:
        if config_path is not None:
            raise ValueError("Resume uses the recorded configuration; --config is not permitted")
        c = read_json(PROJECT / "runs" / resume_id / "config.json")
    else:
        c = config(config_path if config_path is not None else PROJECT / "configs" / (("smoke" if smoke else "full") + suffix + ".yaml"))
        if generations is not None and c["kind"] != "smoke":
            c["generations"] = generations
        if c["kind"] != ("smoke" if smoke else "full"):
            raise ValueError("The command must match the configuration kind (smoke or full)")
    from .common import validate_config
    validate_config(c)
    load_secrets()
    if c.get("initialization") == "reset_s0":
        from .workflow.branch import resolve_branch_source
        c = resolve_branch_source(c)
    api = Runpod()
    p = preflight(probe=True)
    if any(x["status"] not in ("EXITED", "TERMINATED") for x in api.request("GET", "/pods")["pods"]):
        raise RuntimeError("An existing Pod is active; refusing another GPU")
    choices = api.candidates()
    if c.get("required_gpu"):
        required = c["required_gpu"].casefold().removeprefix("nvidia ")
        choices = [x for x in choices if any(str(x.get(key, "")).casefold().removeprefix("nvidia ") == required
                    for key in ("gpu", "name", "display_name", "gpu_name"))]
    if not choices:
        raise RuntimeError("No compatible Secure Cloud GPU with Standard network storage is available")
    archive, code_hash = source_archive()
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
        allowance = min(allowance, SETUP_BUDGET - accrued(ledger(), "smoke") - 0.15, 0.75)
    if allowance <= 0.1:
        raise RuntimeError("Insufficient budget after shutdown/storage reserve")
    if c["kind"] == "full":
        validation_file = {"continued": "validation_continued.json", "reset_s0": "validation_reset_s0.json"}.get(c.get("initialization"), "validation.json")
        validation = read_json(STATE / validation_file)
        if not validation or validation.get("source_code_sha256") != code_hash:
            raise RuntimeError("Validate the complete smoke test for this source version before a full-data launch")
        b = read_json(STATE / "benchmark.json")
        if not b:
            raise RuntimeError("Run the validated smoke test and fetch its benchmark before launching full data")
        estimate_kwargs = {"continued": c.get("initialization") in ("continued", "reset_s0")}
        if c.get("initialization") == "reset_s0":
            estimate_kwargs["first_new_cycle"] = 2
        estimate = forecast(b, c["generations"], choice["hourly_usd"] + VOLUME_HOURLY + CONTAINER_HOURLY, **estimate_kwargs)
        if estimate["estimated_compute_and_running_storage_usd"] > allowance:
            raise RuntimeError("Conservative full-run estimate exceeds remaining budget; select fewer generations")
    # Resolve the private source, including its weight-file checksum, before spending.
    verify_start_checkpoint(c)
    templates = api.request("GET", "/catalog/templates?source=official")["templates"]
    template = next(t for t in templates if t["id"] == "runpod-torch-v280")
    key = register_key(api)
    run_suffix = {"continued": "-continued", "reset_s0": "-reset-s0"}.get(c.get("initialization"), "")
    run_id = resume_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + c["kind"] + run_suffix
    launch_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    local = PROJECT / "runs" / run_id
    local.mkdir(parents=True, exist_ok=True)
    atomic_json(local / "config.json", c)
    print(json.dumps({"provisioning": choice, "image": template["image"], "session_allowance_usd": allowance,
                      "network_volume_gb": 30, "network_volume_usd_per_hour": VOLUME_HOURLY}), flush=True)
    volume_created_at = previous.get("volume_created_at", previous["created_at"]) if previous else now()
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
        matches = [x for x in api.request("GET", "/pods")["pods"] if x["name"] == "iib-" + run_id and x["status"] not in ("EXITED", "TERMINATED")]
        if len(matches) != 1:
            if not matches and not previous:
                api.request("DELETE", "/network-volumes/" + volume["id"])
            raise RuntimeError("Pod create failed/uncertain; pending_volume.json records the allocated storage. No blind retry.") from None
        pod = matches[0]
    resources = {"run_id": run_id, "launch_id": launch_id, "kind": c["kind"], "pod_id": pod["id"], "volume_id": volume["id"],
                 "data_center": choice["data_center"], "gpu": choice["gpu"], "image": template["image"], "created_at": created,
                 "quoted_gpu_hourly_usd": choice["hourly_usd"], "guard_hourly_usd": (choice["hourly_usd"] + VOLUME_HOURLY + CONTAINER_HOURLY) * 1.15,
                 "compute_hourly_usd": (choice["hourly_usd"] + CONTAINER_HOURLY) * 1.15,
                 "volume_hourly_usd": VOLUME_HOURLY, "volume_created_at": volume_created_at,
                 "session_allowance_usd": allowance, "repos": p["hugging_face"]["repos"], "code_sha256": code_hash,
                 "starting_balance_usd": p["account"]["clientBalance"], "remote_root": "/workspace/runs/" + run_id}
    bundle = read_json(STATE / "source_bundle.json", {})
    resources.update(runtime_sha256=code_hash, git_commit=bundle.get("git_commit"))
    resources["configuration_sha256"] = digest(c)
    register_resources(resources)
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
        resources["compute_hourly_usd"] = max(resources["compute_hourly_usd"], (pod["cost"] + CONTAINER_HOURLY) * 1.15)
        register_resources(resources)
        command(client, "mkdir -p /workspace/project " + shlex.quote(resources["remote_root"]))
        sftp = client.open_sftp()
        sftp.get_channel().settimeout(120)
        sftp.put(str(archive), "/workspace/project-code.tar.gz")
        command(client, "tar -xzf /workspace/project-code.tar.gz -C /workspace/project")
        verify_bundle = ("import pathlib,json,hashlib; p=pathlib.Path('/workspace/project'); "
                         "m=json.loads((p/'source_bundle.json').read_text()); "
                         "assert m['runtime_sha256']==" + repr(code_hash) + "; "
                         "assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in m['files'].items())")
        command(client, "python3 -c " + shlex.quote(verify_bundle))
        atomic_json(local / "launch_phase.json", {"phase": "code_transferred", "time": now()})
        upload_json(sftp, resources["remote_root"] + "/resources.json", resources)
        upload_json(sftp, resources["remote_root"] + "/config.json", c)
        if previous and read_json(local / "benchmark.json") and not read_json(local / "benchmark_resources.json"):
            atomic_json(local / "benchmark_resources.json", previous)
            upload_json(sftp, resources["remote_root"] + "/benchmark_resources.json", previous)
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
        command(client, prefix + "python3 -m venv --system-site-packages /opt/iib-venv && /opt/iib-venv/bin/pip install --progress-bar off -e '.[test]' >/workspace/install.log 2>&1", timeout=900)
        atomic_json(local / "launch_phase.json", {"phase": "dependencies_installed", "time": now()})
        gpu = command(client, prefix + "/opt/iib-venv/bin/python -c 'import torch,json; print(json.dumps({\"torch\":torch.__version__,\"cuda\":torch.cuda.is_available(),\"bf16\":torch.cuda.is_bf16_supported(),\"gpu\":torch.cuda.get_device_name()}))'")
        atomic_json(local / "gpu_check.json", json.loads(gpu))
        if not json.loads(gpu)["cuda"] or not json.loads(gpu)["bf16"]:
            raise RuntimeError("Pod CUDA/BF16 check failed")
        if c.get("required_environment"):
            versions = command(client, prefix + "/opt/iib-venv/bin/python -c " + shlex.quote(
                "import importlib.metadata as m,json; print(json.dumps({k:m.version(k) for k in " +
                repr(list(c["required_environment"])) + "}))"))
            if json.loads(versions) != c["required_environment"]:
                raise RuntimeError("Pod package versions differ from the required matched execution environment")
            atomic_json(local / "environment_match.json", {"required": c["required_environment"],
                                                           "actual": json.loads(versions), "verified": True})
        command(client, prefix + "/opt/iib-venv/bin/pip freeze >" + shlex.quote(resources["remote_root"] + "/requirements-tested.txt"))
        supervisor_cmd = ("setsid /opt/iib-venv/bin/python -m model_collapse supervisor --root " + shlex.quote(resources["remote_root"]) +
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
        update_ledger(resources, state="failed", stopped_at=now(),
                      volume_retained_since=now(), volume_hourly_usd=VOLUME_HOURLY,
                      estimated_usd=(now() - created) / 3600 * resources["guard_hourly_usd"])
        raise
    finally:
        if client:
            client.close()


def status(run_id):
    r = local_resources(run_id)
    api = Runpod()
    pod = api.pod(r["pod_id"])
    result = {"run_id": run_id, "pod_id": r["pod_id"], "pod_state": pod["status"], "reported_pod_rate_usd": pod["cost"],
              "compute_running": pod["status"] not in ("EXITED", "TERMINATED")}
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
        # Observation of a stopped provider state closes compute once. Storage remains open.
        update_ledger(r, state="exited", stopped_at=now(), volume_retained_since=now(),
                      volume_hourly_usd=r.get("volume_hourly_usd", VOLUME_HOURLY))
    atomic_json(PROJECT / "runs" / run_id / "last_status.json", result)
    return result


def cleanup(run_id):
    r = local_resources(run_id)
    local = PROJECT / "runs" / run_id
    if not read_json(local / "recovery_verified.json"):
        raise RuntimeError("Fetch and checksum-verify artifacts before removing the recovery volume")
    from .hub import verify_recovered_files
    index = read_json(local / "artifact_index.json")
    if not index:
        raise RuntimeError("Artifact index missing; fetch the latest run state before cleanup")
    if read_json(local / "recovery_verified.json").get("metadata_only"):
        from .continued import verify_recovery
        verify_recovery(local)
        if not read_json(local / "validation_results.json", {}).get("checks", {}).get("independent_budget_guard_and_provider_stop_verified"):
            raise RuntimeError("Validate cloud-restored artifacts before cleanup")
    else:
        verify_recovered_files(local, index["files"])
    api = Runpod()
    if read_json(local / "cleanup_verified.json"):
        return read_json(local / "cleanup_verified.json")
    launches = [x for x in ledger()["launches"] if x["run_id"] == run_id]
    # Full production records are linked to the authoritative current volume. Missing
    # launch ids are supported only for legacy caller-provided, already validated records.
    owned = {x["pod_id"] for x in launches if ("launch_id" not in r or
             (x.get("volume_id") == r["volume_id"] and all(x.get(k) for k in ("launch_id", "pod_id"))))}
    for p in api.request("GET", "/pods")["pods"]:
        if p["id"] not in owned:
            continue
        if p["status"] not in ("EXITED", "TERMINATED"):
            api.stop(p["id"])
        entry = next(x for x in launches if x["pod_id"] == p["id"])
        if entry.get("launch_id"):
            session = {**r, **entry}
            update_ledger(session, state="exited", stopped_at=now(), volume_retained_since=now(),
                          volume_hourly_usd=r.get("volume_hourly_usd", VOLUME_HOURLY))
        api.request("DELETE", "/pods/" + p["id"])
    if any(p["id"] in owned for p in api.request("GET", "/pods")["pods"]):
        raise RuntimeError("Pod deletion not verified")
    api.request("DELETE", "/network-volumes/" + r["volume_id"])
    if any(v["id"] == r["volume_id"] for v in api.request("GET", "/network-volumes")["networkVolumes"]):
        raise RuntimeError("Volume deletion not verified")
    actual = max(0, r["starting_balance_usd"] - api.account()["clientBalance"])
    try:
        atomic_json(local / "billing_read.json", api.billing())
    except RuntimeError as e:
        atomic_json(local / "billing_read.json", {"unavailable": str(e)})
    d = update_ledger(r, state="removed", stopped_at=now(), volume_deleted_at=now(),
                      observed_account_balance_delta_usd=actual,
                      account_delta_includes_other_retained_storage=True)
    atomic_json(local / "cleanup_verified.json", {"pod_removed": True, "volume_removed": True, "actual_account_balance_delta_usd": actual, "ledger": d})
    return {"run_id": run_id, "pod_removed": True, "volume_removed": True, "account_balance_delta_usd": actual}
