from __future__ import annotations

import datetime as dt
import json
import math
import os
import posixpath
import re
import shlex
import time
from pathlib import Path

import paramiko
from huggingface_hub import HfApi, get_token

from ..common import PROJECT, STATE, atomic_json, config, digest, load_secrets, now, read_json, sha256
from ..hub import destinations, retry_transfer
from ..runpod_api import Runpod
from .ownership import latest_session, same_owner, terminal_state

TOTAL_BUDGET = 25.0
SETUP_BUDGET = 3.0
RESERVE = 2.0
VOLUME_HOURLY = 30 * 0.07 / 730
CONTAINER_HOURLY = 20 * 0.10 / 730


def ledger():
    return read_json(STATE / "ledger.json", {"total_budget_usd": TOTAL_BUDGET, "launches": []})


def update_ledger(resources, **updates):
    from filelock import FileLock
    with FileLock(str(STATE / "ledger.lock"), timeout=60):
        return _update_ledger(resources, **updates)


def _update_ledger(resources, **updates):
    from .ledger import apply_update
    d = apply_update(ledger(), resources, updates, volume_hourly=VOLUME_HOURLY)
    atomic_json(STATE / "ledger.json", d)
    # Private control copy survives a lost local checkout. No secrets in this ledger.
    repos = resources.get("repos")
    if repos:
        retry_transfer(HfApi(token=os.environ.get("HF_TOKEN") or get_token()).upload_file, path_or_fileobj=str(STATE / "ledger.json"),
            path_in_repo="_control/ledger.json", repo_id=repos["dataset"], repo_type="dataset", commit_message="Update task cost ledger")
    return d


def accrued(d, kind=None):
    from ..cloud.budget import accrued_cost
    return accrued_cost(d, now(), kind, default_volume_hourly=VOLUME_HOURLY)


def campaign_accrued(d, campaign):
    launches = [x for x in d["launches"] if x.get("acceptance_campaign") == campaign]
    ids = {x["volume_id"] for x in launches}
    ids.update(k for k,v in d.get("volumes",{}).items() if v.get("acceptance_campaign") == campaign)
    selected = {**d, "launches": launches, "volumes": {k: v for k, v in d.get("volumes", {}).items() if k in ids}}
    return accrued(selected)


def campaign_budget(d):
    """An explicit private operational authorization, separate from scientific defaults."""
    amount = d.get("total_budget_usd")
    if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount <= RESERVE:
        raise RuntimeError("Mixed campaign requires an explicit authorized cumulative budget in the task ledger")
    return float(amount)


def task_budget(d):
    """Use recorded authorization; retain the original default for legacy ledgers."""
    return campaign_budget(d) if "total_budget_usd" in d else TOTAL_BUDGET


def setup_budget(d):
    """Honor the recorded setup sublimit, preserving the legacy default."""
    amount = d.get("setup_budget_usd", SETUP_BUDGET)
    if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount <= 0:
        raise RuntimeError("Setup allowance must be a positive finite recorded amount")
    return float(amount)


def configured_placements(c, choices):
    """Respect an explicit live-verified GPU/DC instead of silently changing placement."""
    if c.get("required_gpu"):
        required = c["required_gpu"].casefold().removeprefix("nvidia ")
        choices = [x for x in choices if any(str(x.get(key, "")).casefold().removeprefix("nvidia ") == required
                   for key in ("gpu", "name", "display_name", "gpu_name"))]
    if c.get("required_data_center"):
        choices = [x for x in choices if x["data_center"] == c["required_data_center"]]
    return choices


def configured_gpu_request(c, choice):
    request = {"id": choice["gpu"], "count": 1, "minCudaVersion": "12.8"}
    version = c.get("required_host_cuda_version")
    if version is not None:
        if not isinstance(version, str) or not re.fullmatch(r"[0-9]+\.[0-9]+", version):
            raise ValueError("required_host_cuda_version must be an explicit major.minor version")
        request.pop("minCudaVersion")
        request["allowedCudaVersions"] = [version]
    return request


def configured_container_disk(c):
    size = c.get("container_disk_gb", 20)
    if type(size) is not int or not 20 <= size <= 100:
        raise ValueError("Container disk must be an integer from 20 to 100 GB")
    return size, size * .10 / 730


def check_active_pods(c, pods, ledger_state):
    if c.get("parallel_campaign"):
        from ..cloud.parallel import check_active
        return check_active(c, pods, ledger_state)
    active = [p for p in pods if p["status"] not in ("EXITED", "TERMINATED")]
    if not active:
        return
    protected = c.get("parallel_validation_with")
    from ..workflow.completion import MIXTURE_ALLOCATION
    authorization = ledger_state.get("allocations", {}).get(MIXTURE_ALLOCATION, {})
    if (c.get("workflow") != "completion_comparison" or c.get("kind") != "smoke" or
            c.get("completion_arms") is None or not protected or
            authorization.get("parallel_validation_with") != protected or len(active) != 1):
        raise RuntimeError("An existing Pod is active; refusing another GPU")
    owner = local_resources(protected)
    pod = active[0]
    volume_ids = {x["volumeId"] for x in pod.get("mounts", {}).get("network", [])}
    if (owner.get("kind") != "full" or owner.get("workflow") != "completion_comparison" or
            pod["id"] != owner["pod_id"] or pod.get("name") != "iib-" + protected or
            owner["volume_id"] not in volume_ids):
        raise RuntimeError("Active Pod differs from the specifically authorized protected comparison")


def historical_running_storage(d, hours, *, exclude_volume=None):
    """Conservatively reserve other retained resources during the proposed session."""
    from ..cloud.budget import accrued_cost
    timestamp = now()
    amount = accrued_cost(d, timestamp + hours * 3600) - accrued_cost(d, timestamp)
    volume = d.get("volumes", {}).get(exclude_volume, {})
    if volume.get("charge_started_at") is not None and not volume.get("deleted_at"):
        amount -= hours * volume["hourly_usd"]
    return max(0.0, amount)


def provisioning_deadline(resources):
    """Leave shutdown time even if a short validation budget expires during boot."""
    from ..supervisor import deadline
    return min(now() + 600, deadline(resources) - 30)


def _provider_shutdown(api, pod, resources):
    """Obtain the provider's stop time rather than the later observation time."""
    observed = now()
    candidates = [pod]
    if not any(pod.get(k) for k in ("lastStatusChange", "stoppedAt", "lastStoppedAt")):
        query = "query { pod(input: { podId: " + json.dumps(resources["pod_id"]) + " }) { id desiredStatus lastStatusChange } }"
        value = api.request("POST", "", {"query": query}, graphql=True)
        candidate = (value or {}).get("data", {}).get("pod")
        if candidate:
            candidates.append(candidate)
    for candidate in candidates:
        if candidate.get("id", resources["pod_id"]) != resources["pod_id"]:
            continue
        state = candidate.get("status", candidate.get("desiredStatus", pod.get("status")))
        if state not in ("EXITED", "TERMINATED"):
            continue
        for key in ("stoppedAt", "lastStoppedAt", "lastStatusChange"):
            raw = candidate.get(key)
            if not isinstance(raw, str):
                continue
            parsed = None
            try:
                value = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
                if value.tzinfo is not None:
                    parsed = value.timestamp()
            except ValueError:
                match = re.search(r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) [A-Za-z]{3} \d{1,2} \d{4} \d{2}:\d{2}:\d{2} GMT[+-]\d{4}", raw)
                if match:
                    parsed = dt.datetime.strptime(match.group(0), "%a %b %d %Y %H:%M:%S GMT%z").timestamp()
            if parsed is not None and resources["created_at"] <= parsed <= observed + 300:
                return parsed, {"provider_pod_id": resources["pod_id"], "provider_status": state,
                                "field": key, "value": raw, "provider_shutdown_epoch": parsed,
                                "observed_at": observed}
    raise RuntimeError("Provider stopped compute but no verified shutdown timestamp is available; ledger interval remains unsettled")


def preflight(probe=False, *, network_storage=True):
    load_secrets()
    r = Runpod()
    account = r.account()
    if account["isAutoPayEnabled"] or account["enableBackupPayments"]:
        raise RuntimeError("Automatic payments must be disabled before launching")
    hub = destinations(probe=probe)
    candidates = r.candidates() if network_storage else r.candidates(network_storage=False)
    result = {"account": account, "hugging_face": hub, "gpu_candidates": candidates,
              "total_task_budget_usd": task_budget(ledger()), "estimated_accrued_task_usd": accrued(ledger())}
    atomic_json(STATE / "preflight.json", result)
    return result


def source_archive():
    from ..artifacts.deployment import source_archive as archive
    return archive(PROJECT, STATE)


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
    try:
        client.connect(endpoint["host"], port=endpoint["port"], username=endpoint["username"], key_filename=str(key),
                       timeout=20, auth_timeout=20, banner_timeout=20, look_for_keys=False, allow_agent=False)
    except BaseException:
        client.close()
        raise
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


def _transport_read(operation, cutoff):
    """Retry transient reads, including an auth handshake timeout, never rejected keys."""
    for attempt in range(4):
        try:
            return operation()
        except paramiko.AuthenticationException as error:
            if (str(error).strip().rstrip(".").casefold() != "authentication timeout" or
                    attempt == 3 or now() + min(2 ** attempt, 4) >= cutoff):
                raise
        except (paramiko.BadHostKeyException, paramiko.PasswordRequiredException):
            raise
        except (OSError, EOFError, paramiko.SSHException):
            if attempt == 3 or now() + min(2 ** attempt, 4) >= cutoff:
                raise
        except RuntimeError as error:
            transient = re.search(r"HTTP (?:429|5\d\d)\b", str(error)) or str(error) == "Pod direct SSH is not ready"
            if not transient or attempt == 3 or now() + min(2 ** attempt, 4) >= cutoff:
                raise
        time.sleep(min(2 ** attempt, 4))


def _startup_bad_states(observation, resources):
    """A retained pre-launch pause is pending state, never a successful current handoff."""
    stale, current = False, False
    for key in ("status", "child_status"):
        state = observation.get(key, {})
        if state.get("state") not in ("failed", "budget_paused", "paused"):
            continue
        timestamp = state.get("time")
        old = (not isinstance(timestamp, bool) and isinstance(timestamp, (int, float))
               and math.isfinite(timestamp) and timestamp < resources["created_at"])
        stale = stale or old
        current = current or not old
    return stale, current


def _active_child_snapshot_script(child_root):
    """Read a newly selected child safely before its worker creates status/log files."""
    return ("import pathlib,json; p=pathlib.Path(" + repr(child_root) + "); "
            "s=p/'status.json'; log=p/'pipeline.log'; "
            "print(json.dumps({'status':json.loads(s.read_text()) if s.exists() else "
            "{'state':'starting','status_pending':True}, "
            "'log_tail':log.read_bytes()[-3000:].decode('utf-8',errors='replace') if log.exists() else ''}))")


def upload_json(sftp, path, value):
    with sftp.open(path, "w") as f:
        f.write(json.dumps(value))


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
    if c.get("workflow") in ("fresh_real", "completion_comparison"):
        info = HfApi().model_info(c["model"], revision=c["model_revision"], files_metadata=True)
        if info.sha != c["model_revision"] or not any(item.rfilename.endswith((".bin", ".safetensors")) and item.lfs for item in info.siblings):
            raise RuntimeError("Pinned pretrained model weight metadata is missing")
        return
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


def reject_completed_cloud_resume(run_id, repos):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    revision = HfApi().repo_info(repos["dataset"], repo_type="dataset").sha
    try:
        path = hf_hub_download(repos["dataset"], f"{run_id}/completion.json", repo_type="dataset", revision=revision)
    except EntryNotFoundError:
        return
    if read_json(path, {}).get("complete"):
        atomic_json(PROJECT / "runs" / run_id / "completed_resume_rejected.json", {"revision": revision, "sha256": sha256(path)})
        raise RuntimeError("Completed cloud runs cannot resume; fetch their results instead")


def launch(generations=10, smoke=False, resume_id=None, mode="fresh_base", config_path=None, gpu=None):
    from filelock import FileLock
    STATE.mkdir(exist_ok=True)
    with FileLock(str(STATE / "launch.lock"), timeout=0):
        return _launch(generations, smoke, resume_id, mode, config_path, gpu)


def _launch(generations=10, smoke=False, resume_id=None, mode="fresh_base", config_path=None, gpu=None):
    from ..analysis.forecasts import forecast
    if gpu is not None and (resume_id or not isinstance(gpu, str) or not gpu.strip()):
        raise ValueError("--gpu requires a nonempty GPU name on a new standalone run; resume keeps its recorded hardware")
    if gpu == "auto" and not smoke:
        raise ValueError("Full training must pin the GPU that actually passed acceptance; auto is smoke-only")
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
            if c.get("workflow") == "paired":
                for arm in c["arms"].values():
                    arm["generations"] = generations
            elif c.get("workflow") == "mixed_campaign":
                for job in c["jobs"].values():
                    job["generations"] = generations
            else:
                c["generations"] = generations
        if c["kind"] != ("smoke" if smoke else "full"):
            raise ValueError("The command must match the configuration kind (smoke or full)")
    if gpu is not None:
        if c.get("workflow") in ("paired", "mixed_campaign"):
            raise ValueError("--gpu override currently applies only to standalone conditions")
        if gpu == "auto":
            c.pop("required_gpu", None)
        else:
            c["required_gpu"] = gpu.strip()
    from ..common import validate_config
    validate_config(c)
    container_disk_gb, container_hourly = configured_container_disk(c)
    if c.get("workflow") == "completion_arm":
        raise ValueError("Completion arms are launched by their owning comparison coordinator")
    load_secrets()
    if previous:
        reject_completed_cloud_resume(resume_id, previous["repos"])
    paired = c.get("workflow") == "paired"
    campaign = c.get("workflow") == "mixed_campaign"
    fresh_real = c.get("workflow") == "fresh_real"
    completion = c.get("workflow") == "completion_comparison"
    pod_storage = c.get("storage_backend") == "pod_persistent"
    if previous and previous.get("storage_backend", "network") != c.get("storage_backend", "network"):
        raise RuntimeError("Resume storage backend differs from its registered owner")
    if paired:
        from ..workflow.paired import resolve_pair_inputs
        c = resolve_pair_inputs(c)
    elif campaign:
        from ..workflow.mixed_campaign import resolve_campaign_inputs
        c = resolve_campaign_inputs(c)
    elif c.get("initial_dataset"):
        from ..workflow.initial_data import resolve_initial_dataset
        c = resolve_initial_dataset(c)
    if completion:
        from ..workflow.completion import resolve_inputs
        c = resolve_inputs(c)
    if c.get("initialization") == "reset_s0":
        from ..workflow.branch import resolve_branch_source
        c = resolve_branch_source(c)
    resume_work = None
    if campaign and previous:
        from ..workflow.mixed_campaign import verified_resume_jobs
        resume_work = verified_resume_jobs(PROJECT / "runs" / resume_id, previous, c)
    api = Runpod()
    if completion and c.get("completion_profile") == "smollm2_completion64_v1":
        from ..workflow.smollm import resolve_backup
        c = resolve_backup(c, api)
    p = preflight(probe=True, network_storage=False) if pod_storage else preflight(probe=True)
    # Use preflight for the initial screen. Native-model launches recheck after
    # the slower storage/authentication work, immediately before acquisition.
    choices = p.get("gpu_candidates")
    if choices is None:  # Compatibility with older preflight adapters.
        choices = api.candidates()
    if gpu == "auto":
        modern = ("ada", "blackwell", "a100", "a40", "a4000", "a4500", "a5000", "a6000",
                  "l4", "h100", "h200", "b200", "b300", "geforce rtx 30", "geforce rtx 40", "geforce rtx 50")
        choices = [x for x in choices if x["hourly_usd"] <= 1.19 and " mig " not in x["gpu"].casefold()
                   and any(name in x["gpu"].casefold() for name in modern)]
    if fresh_real or completion:
        modern = ("ada", "blackwell", "a100", "a40", "a4000", "a4500", "a5000", "a6000", "l4", "h100", "h200", "b200", "b300", "geforce rtx 30", "geforce rtx 40", "geforce rtx 50")
        choices = [item for item in choices if item["vram_gb"] >= 16 and " mig " not in item["gpu"].casefold()
                   and any(value in item["gpu"].casefold() for value in modern)]
    choices = configured_placements(c, choices)
    if not choices:
        raise RuntimeError("No compatible Secure Cloud GPU is currently available" if pod_storage else
                           "No compatible Secure Cloud GPU with Standard network storage is available")
    check_active_pods(c, api.request("GET", "/pods")["pods"], ledger())
    storage_admission = None
    if completion and (c.get("completion_arms") is not None or c.get("decoding_arms") is not None or c.get("training_chunk_policy") == "repeat_fixed"):
        from ..cloud.hf_storage import check_storage
        from ..workflow.completion import required_backup_bytes
        destination = read_json(STATE / "completion64-new-backup-destination.json", {})
        if not destination.get("free_plan") or destination.get("repos") != p["hugging_face"]["repos"]:
            raise RuntimeError("Mixtures require the verified dedicated free private backup namespace")
        metadata_only = bool(c.get("backup_store"))
        if metadata_only:
            from ..cloud.s3_artifacts import probe, check_capacity
            bulk_probe = probe(c["backup_store"])
            volume_info = api.request("GET", "/network-volumes/" + c["backup_store"]["bucket"])
            if volume_info["dataCenter"] != c["backup_store"]["region"]:
                raise RuntimeError("Bulk backup volume capacity/region differs from admission")
            bulk_capacity = check_capacity(c["backup_store"], volume_info["size"], required_backup_bytes(c))
        storage_admission = check_storage(p["hugging_face"]["repos"], destination["organization"],
            100_000_000_000, 50_000_000 if metadata_only else required_backup_bytes(c), 10_000_000_000,
            HfApi(token=os.environ.get("HF_TOKEN") or get_token()))
        if metadata_only:
            storage_admission["bulk_round_trip"] = bulk_probe
            storage_admission["bulk_capacity"] = bulk_capacity
    archive, code_hash = source_archive()
    if previous and previous["code_sha256"] != code_hash and c["kind"] != "smoke":
        raise RuntimeError("Resume requires the original code; restore its recorded commit")
    if previous:
        choices = [x for x in choices if x["data_center"] == previous["data_center"]]
        if not choices:
            raise RuntimeError("No compatible GPU is available beside this run's retained volume")
    choice = choices[0]
    gpu_request = configured_gpu_request(c, choice)
    if gpu == "auto" or fresh_real or completion:
        c["required_gpu"] = choice["gpu"]  # Seal actual hardware into the protocol before certification.
    volume_gb = c["network_volume_gb"] if campaign or completion else 30
    # Pod volume stopped storage is more expensive; reserve that rate throughout.
    volume_hourly = volume_gb * (.20 if pod_storage else .07) / 730
    ledger_state = ledger()
    total_budget = campaign_budget(ledger_state) if campaign else task_budget(ledger_state)
    spent = accrued(ledger_state)
    allowance = min(total_budget - spent - RESERVE, p["account"]["clientBalance"] - RESERVE)
    if c.get("parallel_campaign"):
        from ..cloud.parallel import reserved_sibling_allowance, scope
        authorization = scope(c, ledger_state)
        if code_hash != authorization.get("runtime_sha256"):
            raise RuntimeError("Parallel runtime differs from its recorded authorization")
        allowance -= reserved_sibling_allowance(c, ledger_state, accrued)
    if completion:
        from ..workflow.completion import allocation_id
        allocation_name = allocation_id(c)
        allocation = ledger_state.get("allocations", {}).get(allocation_name)
        if not allocation:
            raise RuntimeError("Completion comparison requires its recorded allocation")
        allocated = {**ledger_state, "launches": [x for x in ledger_state["launches"] if x.get("allocation") == allocation_name]}
        ids = {x["volume_id"] for x in allocated["launches"]}
        if c.get("completion_profile") == "smollm2_completion64_v1":
            ids.update(k for k,v in ledger_state.get("volumes",{}).items() if v.get("allocation") == allocation_name)
        allocated["volumes"] = {k: v for k, v in ledger_state.get("volumes", {}).items() if k in ids}
        allowance = min(allowance, allocation["budget_usd"] - accrued(allocated) - allocation["reserve_usd"])
    if c.get("max_session_usd") is not None:
        allowance = min(allowance, c["max_session_usd"])
    if c["kind"] == "smoke":
        allowance = min(allowance, setup_budget(ledger_state) - accrued(ledger_state, "smoke") - 0.15, 0.75)
        if c.get("parallel_campaign"):
            validation_entries = [x for x in ledger_state["launches"] if
                x.get("parallel_campaign") == c["parallel_campaign"] and x.get("kind") == "smoke"]
            ids = {x["volume_id"] for x in validation_entries}
            validation_state = {**ledger_state, "launches": validation_entries,
                "volumes": {k: v for k, v in ledger_state.get("volumes", {}).items() if k in ids}}
            allowance = min(allowance, authorization["validation_cap_usd"] - accrued(validation_state))
        if completion:
            from ..workflow.completion import acceptance_campaign
            allowance = min(allowance, allocation.get("validation_cap_usd", .50) - campaign_accrued(ledger_state, acceptance_campaign(c)))
        elif paired:
            allowance = min(allowance, .75 - campaign_accrued(ledger(), "shared_start_final_epoch5"))
        elif fresh_real:
            from ..workflow.fresh_real import ACCEPTANCE
            allowance = min(allowance, .50 - campaign_accrued(ledger(), ACCEPTANCE))
        elif campaign:
            from ..workflow.mixed_campaign import CAMPAIGN
            allowance = min(allowance, .75 - campaign_accrued(ledger(), CAMPAIGN))
        else:
            from ..mixing import CAMPAIGN, mixture_enabled
            if mixture_enabled(c):
                allowance = min(allowance, .75 - campaign_accrued(ledger(), CAMPAIGN))
    if completion and c.get("completion_profile") == "smollm2_completion64_v1" and c["kind"] == "smoke":
        elapsed = sum(max(0, x.get("compute_stopped_at", x.get("stopped_at", now())) - x.get("compute_started_at", x["created_at"])) for x in ledger_state["launches"] if x.get("allocation") == allocation_name and x.get("kind") == "smoke")
        if elapsed >= c["validation_max_seconds"]:
            raise RuntimeError("Combined fast-validation paid-time allowance exhausted")
    if allowance <= 0.1:
        raise RuntimeError("Insufficient budget after shutdown/storage reserve")
    if c["kind"] == "full":
        from ..mixing import certificate_candidates
        validation = next((record for name in certificate_candidates(c, code_hash)
                           if (record := read_json(STATE / name)) is not None), None)
        from ..workflow.profiles import require_acceptance
        require_acceptance(c, validation, code_hash)
        if completion:
            from ..workflow.completion import benchmark_filename
            benchmark_file = benchmark_filename(c)
        else:
            benchmark_file = "benchmark_fresh_real.json" if fresh_real else "benchmark.json"
        b = read_json(STATE / benchmark_file)
        if not b:
            raise RuntimeError("Run the validated smoke test and fetch its benchmark before launching full data")
        estimate_kwargs = {"continued": c.get("initialization") in ("continued", "reset_s0")}
        if c.get("initial_dataset"):
            estimate_kwargs["imported_d1"] = True
        if c.get("synthetic_fraction", 1.0) == 0:
            estimate_kwargs["real_only"] = True
        if "epochs" in c:
            estimate_kwargs["epochs"] = c["epochs"]
        if c.get("initialization") == "reset_s0":
            estimate_kwargs["initialization"] = "reset_s0"
            estimate_kwargs["first_new_cycle"] = 2
        if completion:
            from ..workflow.completion import forecast as completion_forecast
            estimate = completion_forecast(b, c, choice["hourly_usd"] + volume_hourly + container_hourly)
            projection_state = ledger_state
            if c.get("parallel_campaign"):
                # Sibling compute is already reserved above. Its storage remains
                # in the projection and the campaign retention reserve.
                projection_state = {**ledger_state, "launches": [x for x in ledger_state["launches"]
                    if x.get("parallel_campaign") != c["parallel_campaign"]]}
            estimate["historical_storage_during_run_usd"] = historical_running_storage(projection_state, estimate["estimated_hours"],
                exclude_volume=previous.get("volume_id") if previous else None)
            estimate["seven_day_volume_storage_usd"] = volume_hourly * 24 * 7
            allowance -= estimate["historical_storage_during_run_usd"]
            if c.get("completion_profile") == "smollm2_completion64_v1":
                from ..workflow.smollm import retention_forecast
                estimate.update(retention_forecast(c, storage_admission["bulk_capacity"]["quota_bytes"], RESERVE))
                allowance -= estimate["retention_above_existing_reserve_usd"]
            if estimate["estimated_compute_and_running_storage_usd"] * 1.15 > allowance:
                raise RuntimeError("Completion comparison forecast exceeds the padded watchdog allowance")
        elif fresh_real:
            from ..workflow.fresh_real import forecast as fresh_forecast
            estimate = fresh_forecast(b, c, choice["hourly_usd"] + volume_hourly + container_hourly)
            estimate["historical_storage_during_run_usd"] = historical_running_storage(ledger_state, estimate["estimated_hours"],
                exclude_volume=previous.get("volume_id") if previous else None)
            allowance -= estimate["historical_storage_during_run_usd"]
            if estimate["estimated_compute_and_running_storage_usd"] * 1.15 > allowance:
                raise RuntimeError("FineWeb forecast cannot fit the padded watchdog allowance")
        elif paired:
            from ..analysis.forecasts import forecast_pair
            estimate = forecast_pair(b, c["arms"]["continued"]["generations"], choice["hourly_usd"] + volume_hourly + container_hourly)
        elif campaign:
            from ..analysis.forecasts import forecast_campaign
            forecast_options = {"completed_jobs": resume_work["sealed_completed_jobs"]} if resume_work else {}
            estimate = forecast_campaign(b, c["jobs"]["continued20"]["generations"], choice["hourly_usd"] + volume_hourly + container_hourly,
                                         **forecast_options)
            if resume_work:
                estimate["resume_work_verification"] = resume_work
            estimate["historical_storage_during_run_usd"] = historical_running_storage(
                ledger_state, estimate["estimated_hours"], exclude_volume=previous.get("volume_id") if previous else None)
            allowance -= estimate["historical_storage_during_run_usd"]
            estimate["padded_watchdog_required_usd"] = estimate["estimated_compute_and_running_storage_usd"] * 1.15
            if estimate["padded_watchdog_required_usd"] > allowance:
                raise RuntimeError("Conservative campaign runtime cannot fit the padded watchdog deadline after historical storage and reserve")
        else:
            estimate = forecast(b, c["generations"], choice["hourly_usd"] + volume_hourly + container_hourly, **estimate_kwargs)
        if estimate["estimated_compute_and_running_storage_usd"] > allowance:
            raise RuntimeError("Conservative full-run estimate exceeds remaining budget; select fewer generations")
    # Resolve the private source, including its weight-file checksum, before spending.
    verify_start_checkpoint(c["arms"]["continued"] if paired else c["jobs"]["continued20"] if campaign else c)
    templates = api.request("GET", "/catalog/templates?source=official")["templates"]
    template = next(t for t in templates if t["id"] == "runpod-torch-v280")
    if c.get("required_image") and template["image"] != c["required_image"]:
        raise RuntimeError("Official template differs from the pinned matched image")
    key = register_key(api)
    run_suffix = "-completion64" if completion else "-fresh-real" if fresh_real else "-mixed-campaign" if campaign else "-paired-final5" if paired else {"continued": "-continued", "reset_s0": "-reset-s0"}.get(c.get("initialization"), "")
    if completion and c.get("decoding_arms") is not None:
        run_suffix = "-completion64-decoding"
    if completion and c.get("completion_arms") is not None:
        from ..workflow.completion import arm_specs
        definitions = arm_specs(c)
        run_suffix = "-completion64-" + definitions[0]["id"] if len(definitions) == 1 else "-completion64-mixtures"
    run_id = resume_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + c["kind"] + run_suffix
    launch_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    local = PROJECT / "runs" / run_id
    local.mkdir(parents=True, exist_ok=True)
    atomic_json(local / "config.json", c)
    print(json.dumps({"provisioning": choice, "image": template["image"], "session_allowance_usd": allowance,
                      "storage_backend": "pod_persistent" if pod_storage else "network",
                      "persistent_storage_gb": volume_gb, "conservative_storage_usd_per_hour": volume_hourly}), flush=True)
    client, resources, pod, volume = None, None, None, None
    pod_request_started, pod_absent_verified = False, False
    created = now()
    try:
        volume_created_at = previous.get("volume_created_at", previous["created_at"]) if previous else now()
        volume = None
        if not pod_storage:
            volume = api.request("GET", "/network-volumes/" + previous["volume_id"]) if previous else api.request("POST", "/network-volumes", {
                "name": "iib-" + run_id, "size": volume_gb, "dataCenter": choice["data_center"], "type": "STANDARD"})
            atomic_json(local / "pending_volume.json", {"volume_id": volume["id"], "created_by_task": not bool(previous)})
            check_volume = api.request("GET", "/network-volumes/" + volume["id"])
            if check_volume["size"] != volume_gb or check_volume["dataCenter"] != choice["data_center"]:
                raise RuntimeError("Volume read-back mismatch")
        created = now()
        try:
            if completion and c.get("completion_profile") == "smollm2_completion64_v1":
                live_choices = configured_placements(c, api.candidates())
                if not live_choices or live_choices[0]["hourly_usd"] > choice["hourly_usd"]:
                    raise RuntimeError("GPU placement disappeared or became more expensive before acquisition")
            payload = {"name": "iib-" + run_id, "cloud": "SECURE", "image": template["image"],
                "disk": container_disk_gb, "gpu": gpu_request,
                "dataCenterIds": [choice["data_center"]],
                "mounts": {"persistent": {"size": volume_gb, "path": "/workspace"}} if pod_storage else {"network": [{"volumeId": volume["id"], "path": "/workspace"}]},
                "startSsh": True, "startJupyter": False, "ports": ["22/tcp"]}
            atomic_json(local / "pending_pod_request.json", payload)
            if pod_storage and previous:
                old = api.pod(previous["pod_id"])
                if old["status"] != "EXITED" or previous["volume_id"] != previous["pod_id"]:
                    raise RuntimeError("Pod-persistent resume requires its exact stopped owner")
                pod_request_started = True
                api.request("POST", "/pods/" + previous["pod_id"] + "/action", {"action": "start"})
                pod = api.pod(previous["pod_id"])
            else:
                pod_request_started = True
                if c.get("pod_create_api", "rest_v2") == "graphql":
                    pod = api.create_pod_graphql(payload)
                elif c.get("pod_create_api", "rest_v2") == "rest_v2":
                    pod = api.request("POST", "/pods", payload)
                else:
                    raise ValueError("Unsupported Pod creation API")
        except Exception as error:
            detail = str(error)
            for secret in (api.key if hasattr(api, "key") else None, os.environ.get("HF_TOKEN")):
                if secret:
                    detail = detail.replace(secret, "[REDACTED]")
            try:
                atomic_json(local / "pod_create_error.json", {"error_type": type(error).__name__, "error": detail[:2000]})
            except (OSError, ValueError):
                pass  # Diagnostic failure must not prevent acquired-Pod reconciliation.
            # Do not retry a timed-out create: reconcile by exact unique name first.
            matches = [x for x in api.request("GET", "/pods")["pods"] if x["name"] == "iib-" + run_id and x["status"] not in ("EXITED", "TERMINATED")]
            pod_absent_verified = not matches
            if len(matches) != 1:
                if matches:
                    # Preserve an uncertain mounted volume for reconciliation.
                    volume = None
                raise RuntimeError("Pod create failed/uncertain; inspect pod_create_error.json and pending resource records. No blind retry.") from None
            pod = matches[0]
        if pod_storage:
            volume = {"id": pod["id"]}  # The persistent disk is a component owned by this exact Pod.
        resources = {"run_id": run_id, "launch_id": launch_id, "kind": c["kind"], "pod_id": pod["id"], "volume_id": volume["id"],
                     "container_disk_gb": container_disk_gb,
                     "data_center": choice["data_center"], "gpu": choice["gpu"], "image": template["image"], "created_at": created,
                     "quoted_gpu_hourly_usd": choice["hourly_usd"], "guard_hourly_usd": (choice["hourly_usd"] + volume_hourly + container_hourly) * 1.15,
                     "compute_hourly_usd": (choice["hourly_usd"] + container_hourly) * 1.15,
                     "volume_hourly_usd": volume_hourly, "network_volume_gb": 0 if pod_storage else volume_gb, "volume_created_at": volume_created_at,
                     "session_allowance_usd": allowance, "repos": p["hugging_face"]["repos"], "code_sha256": code_hash,
                     "starting_balance_usd": p["account"]["clientBalance"], "remote_root": "/workspace/runs/" + run_id}
        if completion and c.get("completion_profile") == "smollm2_completion64_v1" and c["kind"] == "smoke":
            allocation_name = allocation_id(c) if completion else None
            used_seconds = sum(max(0, x.get("compute_stopped_at", x.get("stopped_at", now())) - x.get("compute_started_at", x["created_at"])) for x in ledger()["launches"] if x.get("allocation") == allocation_name and x.get("kind") == "smoke")
            resources["absolute_deadline"] = created + max(0, c["validation_max_seconds"] - used_seconds)
        bundle = read_json(STATE / "source_bundle.json", {})
        resources.update(runtime_sha256=code_hash, git_commit=bundle.get("git_commit"))
        resources["configuration_sha256"] = digest(c)
        if c.get("parallel_campaign"):
            resources["parallel_campaign"] = c["parallel_campaign"]
        if c.get("backup_store"):
            resources["backup_store"] = c["backup_store"]
        if pod_storage:
            resources.update(storage_backend="pod_persistent", persistent_volume_gb=volume_gb)
        if completion:
            from ..workflow.completion import allocation_id, acceptance_campaign
            resources.update(workflow="completion_comparison", allocation=allocation_id(c))
            if storage_admission:
                atomic_json(local / "storage_admission.json", storage_admission)
            if c["kind"] == "smoke":
                resources["acceptance_campaign"] = acceptance_campaign(c)
            else:
                resources["admission_forecast"] = estimate
        elif fresh_real:
            resources["workflow"] = "fresh_real"
            if c["kind"] == "smoke":
                from ..workflow.fresh_real import ACCEPTANCE
                resources["acceptance_campaign"] = ACCEPTANCE
            else:
                resources["admission_forecast"] = estimate
        elif campaign:
            resources.update(workflow="mixed_campaign", authorized_cumulative_budget_usd=total_budget)
            fields = ("run_id", "launch_id", "pod_id", "volume_id")
            resources["provider_sessions"] = [{k: item[k] for k in fields} for item in ledger_state["launches"]
                    if item.get("run_id") == run_id and item.get("volume_id") == volume["id"] and all(k in item for k in fields)]
            resources["provider_sessions"].append({k: resources[k] for k in fields})
            if c["kind"] == "smoke":
                from ..workflow.mixed_campaign import CAMPAIGN
                resources["acceptance_campaign"] = CAMPAIGN
            else:
                resources["admission_forecast"] = estimate
        elif paired:
            resources["workflow"] = "paired"
            if c["kind"] == "smoke":
                resources["acceptance_campaign"] = "shared_start_final_epoch5"
        elif c["kind"] == "smoke":
            from ..mixing import CAMPAIGN, mixture_enabled
            if mixture_enabled(c):
                resources["acceptance_campaign"] = CAMPAIGN
        register_resources(resources)
        update_ledger(resources, state="provisioning")
        ready_by = provisioning_deadline(resources)
        while now() < ready_by:
            pod = _transport_read(lambda: api.pod(resources["pod_id"]), ready_by) if campaign else api.pod(pod["id"])
            if pod.get("ssh", {}).get("direct") and pod["status"] == "RUNNING":
                try:
                    client = ssh_connect(pod, key)
                    break
                except (OSError, paramiko.SSHException) as error:
                    if campaign and isinstance(error, (paramiko.AuthenticationException, paramiko.BadHostKeyException, paramiko.PasswordRequiredException)):
                        raise
                    pass
            time.sleep(8)
        if client is None:
            raise RuntimeError("Pod did not become SSH-ready within provisioning health timeout")
        if pod["gpu"]["id"] != choice["gpu"] or pod["dataCenterId"] != choice["data_center"]:
            raise RuntimeError("Pod read-back configuration mismatch")
        if c.get("required_host_cuda_version"):
            if pod.get("cudaVersion") != c["required_host_cuda_version"]:
                raise RuntimeError("Pod host CUDA read-back differs from its pinned placement")
            resources["host_cuda_version"] = pod["cudaVersion"]
        if pod_storage:
            mounted = pod.get("mounts", {}).get("persistent", {})
            if mounted.get("path") != "/workspace" or mounted.get("size") != volume_gb:
                raise RuntimeError("Pod persistent-disk read-back mismatch")
        resources["actual_pod_hourly_usd"] = pod["cost"]
        resources["verified_ssh_direct"] = pod["ssh"]["direct"]
        resources["guard_hourly_usd"] = max(resources["guard_hourly_usd"], (pod["cost"] + volume_hourly + container_hourly) * 1.15)
        resources["compute_hourly_usd"] = max(resources["compute_hourly_usd"], (pod["cost"] + container_hourly) * 1.15)
        register_resources(resources)
        if campaign and c["kind"] == "full" and resources["session_allowance_usd"] / resources["guard_hourly_usd"] < estimate["estimated_hours"]:
            raise RuntimeError("Actual provider rate makes the campaign's padded watchdog deadline unaffordable")
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
            if c.get("backup_store"):
                for name in ("RUNPOD_S3_ACCESS_KEY_ID", "RUNPOD_S3_SECRET_ACCESS_KEY"):
                    if not os.environ.get(name):
                        raise RuntimeError("Missing bulk backup credential: " + name)
                    f.write("export " + name + "=" + shlex.quote(os.environ[name]) + "\n")
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
        if campaign and c["kind"] == "full":
            from ..supervisor import deadline
            remaining_required = max(0, estimate["estimated_hours"] * 3600 - 300)
            if deadline(resources) - now() < remaining_required:
                raise RuntimeError("Campaign dependency setup consumed the padded runtime budget; refusing to start training")
        supervisor_cmd = ("setsid /opt/iib-venv/bin/python -m model_collapse supervisor --root " + shlex.quote(resources["remote_root"]) +
            (" --validation" if c["kind"] == "smoke" else "") + " >" + shlex.quote(resources["remote_root"] + "/supervisor.log") + " 2>&1 </dev/null &")
        command(client, prefix + "sh -c " + shlex.quote(supervisor_cmd), timeout=30)
        for _ in range(30):
            try:
                read_live = lambda: json.loads(command(client, ". /run/iib-week1-secrets; python3 -c " + shlex.quote(
                    "import json,pathlib; p=pathlib.Path(" + repr(resources["remote_root"]) + "); print(json.dumps({k:json.loads((p/(k+'.json')).read_text()) for k in ['watchdog','supervisor','status']}))"), timeout=30 if campaign else 600))
                live = _transport_read(read_live, now() + 30) if campaign else read_live()
            except (RuntimeError, ValueError) as error:
                if campaign and re.search(r"HTTP (?:401|403)\b", str(error)):
                    raise
                time.sleep(3)
                continue
            if campaign and _startup_bad_states(live, resources)[1]:
                raise RuntimeError("Campaign failed or paused during startup; refusing a running handoff")
            if now() - live["watchdog"]["time"] < 20 and now() - live["supervisor"]["time"] < 20:
                atomic_json(local / "launch_verified.json", live)
                update_ledger(resources, state="running")
                if campaign:
                    # The launcher itself proves new training and progress after closing its setup SSH connection.
                    client.close()
                    client = None
                    backup_only = bool(resume_work and len(resume_work["sealed_completed_jobs"]) == 3)
                    terminal = None
                    if backup_only:
                        terminal = verify_campaign_terminal(run_id, max_wait_seconds=600)
                    else:
                        try:
                            progress = verify_detached(run_id, max_wait_seconds=600)
                        except RuntimeError:
                            # A compatible resume can finish only backup/seal repair before a new epoch exists.
                            if not previous or api.pod(resources["pod_id"])["status"] not in ("EXITED", "TERMINATED"):
                                raise
                            terminal = verify_campaign_terminal(run_id, max_wait_seconds=10)
                        else:
                            atomic_json(local / "launch_verified.json", {**live, "detached_progress": progress})
                    if terminal:
                        atomic_json(local / "launch_verified.json", {**live, "terminal_repair": terminal})
                        print(json.dumps({"run_id": run_id, "pod_id": pod["id"], "state": "complete",
                                          "completed_backup_repair_and_provider_shutdown_verified": True}), flush=True)
                        return {**resources, "state": "complete", "terminal_repair": terminal}
                elif fresh_real or completion:
                    client.close()
                    client = None
                    progress = verify_detached(run_id, max_wait_seconds=600)
                    atomic_json(local / "launch_verified.json", {**live, "detached_progress": progress})
                print(json.dumps({"run_id": run_id, "pod_id": pod["id"], "state": "running",
                                  "detached_supervisor_and_watchdog_verified": True,
                                  **({"actual_new_training_progress_verified": True} if campaign or fresh_real or completion else {})}), flush=True)
                return resources
            time.sleep(3)
        raise RuntimeError("Supervisor/worker/watchdog did not all become healthy")
    except BaseException:
        if pod is None:
            if (volume is not None and not previous and not pod_storage
                    and (not pod_request_started or pod_absent_verified)):
                api.request("DELETE", "/network-volumes/" + volume["id"])
            raise
        api.stop(pod["id"])
        if resources is None:
            raise
        stopped = now()
        if campaign or fresh_real or completion:
            stopped, proof = _provider_shutdown(api, api.pod(resources["pod_id"]), resources)
            atomic_json(local / "provider_shutdown.json", proof)
        update_ledger(resources, state="failed", stopped_at=stopped,
                      volume_retained_since=stopped, volume_hourly_usd=volume_hourly,
                      estimated_usd=(now() - created) / 3600 * resources["guard_hourly_usd"])
        raise
    finally:
        if client:
            client.close()


def _detached_observation(resources):
    api = Runpod()
    pod = api.pod(resources["pod_id"])
    if pod["status"] != "RUNNING":
        raise RuntimeError("Detached progress verification requires the active paired Pod")
    if not pod.get("ssh", {}).get("direct") and resources.get("verified_ssh_direct"):
        # Reuse the address that successfully authenticated this exact provider session.
        # Existing known-host verification still rejects a changed SSH host key.
        pod = {**pod, "ssh": {**pod.get("ssh", {}), "direct": resources["verified_ssh_direct"]}}
    client = ssh_connect(pod, STATE / "runpod_ed25519")
    try:
        allowed = [resources["run_id"] + "-continued", resources["run_id"] + "-reset-s0"]
        if resources.get("workflow") == "mixed_campaign":
            from ..workflow.mixed_campaign import job_roots
            allowed = [p.name for p in job_roots(Path(resources["run_id"])).values()]
        elif resources.get("workflow") == "completion_comparison":
            from ..workflow.completion import child_roots
            arm_c = read_json(PROJECT / "runs" / resources["run_id"] / "config.json", {})
            allowed = [p.name for p in child_roots(Path(resources["run_id"]), arm_c).values()]
        script = ("import pathlib,json,math; p=pathlib.Path(" + repr(resources["remote_root"]) + "); "
                  "read=lambda q: json.loads(q.read_text()) if q.exists() else {}; "
                  "s=read(p/'status.json'); n=s.get('active_child'); "
                  "assert n is None or n in " + repr(allowed) + "; "
                  "q=p.parent/n if n else p; "
                  "events=q/'events.jsonl'; histories=list(q.glob('seed_*/gen_*/training_history.json')); "
                  "first=2 if read(q/'config.json').get('initialization')=='reset_s0' else 1; "
                  "new=[x for x in histories if int(x.parent.name.removeprefix('gen_'))>=first and not (x.parent/'import.json').exists()]; "
                  "rows=[row for x in new for row in read(x)]; "
                  "print(json.dumps({'time':__import__('time').time(),'status':s,'child_status':read(q/'status.json'),"
                  "'event_bytes':events.stat().st_size if events.exists() else 0,'epochs':sum(len(read(x)) for x in histories),"
                  "'new_epochs':len(rows),'finite_new_training':bool(rows) and all(math.isfinite(row[k]) for row in rows for k in ['train_loss','validation_loss']),"
                  "'completed_new_stages':sum((x.parent/'stage_completion.json').exists() for x in new),"
                  "'supervisor':read(p/'supervisor.json'),'watchdog':read(p/'watchdog.json')}))")
        return json.loads(command(client, "python3 -c " + shlex.quote(script),
                                  timeout=30 if resources.get("workflow") == "mixed_campaign" else 600))
    finally:
        client.close()


def verify_campaign_terminal(run_id, *, max_wait_seconds=600):
    """Accept backup-only resume completion, without demanding or fabricating new training."""
    from ..supervisor import deadline
    from ..workflow.mixed_campaign import JOB_ORDER, verified_resume_jobs
    resources = local_resources(run_id)
    root = PROJECT / "runs" / run_id
    c = read_json(root / "config.json")
    if c.get("workflow") != "mixed_campaign":
        raise ValueError("Terminal repair verification requires a mixed campaign")
    api = Runpod()
    cutoff = min(now() + max_wait_seconds, deadline(resources) - 2)
    while now() <= cutoff:
        pod = _transport_read(lambda: api.pod(resources["pod_id"]), cutoff)
        if pod["status"] in ("EXITED", "TERMINATED"):
            recovered = verified_resume_jobs(root, resources, c)
            index = read_json(root / "artifact_index.json")
            completion = read_json(root / "completion.json", {})
            if (set(recovered["sealed_completed_jobs"]) != set(JOB_ORDER)
                    or "completion.json" not in index["files"] or not completion.get("complete")
                    or completion.get("workflow") != "mixed_campaign"
                    or completion.get("source20_seal_sha256") != sha256(root / "continued20_seal.json")):
                raise RuntimeError("Stopped campaign has no verified completed private backup")
            stopped, provider = _provider_shutdown(api, pod, resources)
            update_ledger(resources, state="exited", stopped_at=stopped, volume_retained_since=stopped,
                          volume_hourly_usd=resources["volume_hourly_usd"], provider_shutdown_verified=True,
                          stop_observed_at=provider["observed_at"])
            receipt = {"run_id": run_id, "pod_id": resources["pod_id"], "state": "complete",
                       "provider_shutdown": provider, "recovered_completed_jobs": recovered,
                       "completion_sha256": sha256(root / "completion.json"),
                       "scope": "Completed backup/seal repair verified from pinned private metadata; no fresh training was required."}
            atomic_json(root / "terminal_repair_verified.json", receipt)
            return receipt
        time.sleep(3)
    raise RuntimeError("Campaign terminal repair did not complete with verified provider shutdown within its health deadline")


def verify_detached(run_id, *, max_wait_seconds=None):
    """Observe real stage/epoch progress across fresh, closed SSH connections."""
    r = local_resources(run_id)
    c = read_json(PROJECT / "runs" / run_id / "config.json", {})
    from ..mixing import mixture_enabled
    if c.get("workflow") not in ("paired", "mixed_campaign", "fresh_real", "completion_comparison") and not mixture_enabled(c):
        raise ValueError("This evidence command applies to paired or mixed-data runs")
    campaign = c.get("workflow") == "mixed_campaign"
    fresh_real = c.get("workflow") in ("fresh_real", "completion_comparison")
    completion = c.get("workflow") == "completion_comparison"
    wait_seconds = max_wait_seconds if max_wait_seconds is not None else (600 if campaign else 120)
    cutoff = now() + wait_seconds
    if campaign or completion:
        from ..supervisor import deadline
        cutoff = min(cutoff, deadline(r) - 10)
    def observe():
        while True:
            try:
                if campaign or completion:
                    return _transport_read(lambda: _detached_observation(r), cutoff)
                return _detached_observation(r)
            except RuntimeError as error:
                if not completion or str(error) != "Pod direct SSH is not ready" or now() + 3 >= cutoff:
                    raise
                time.sleep(3)
    first = observe()
    if campaign and _startup_bad_states(first, r)[1]:
        raise RuntimeError("Campaign failed or paused before detached training progress was verified")
    def progress(x):
        s, cs = x["status"], x["child_status"]
        return (s.get("phase"), s.get("active_child"), cs.get("generation"), cs.get("stage"), x["event_bytes"], x["epochs"])
    attempt = 0
    while now() < cutoff:
        time.sleep(3)
        attempt += 1
        second = observe()
        stale, failed = _startup_bad_states(second, r) if campaign or fresh_real else (False, False)
        if failed:
            raise RuntimeError("Campaign failed or paused before detached training progress was verified")
        fresh = all(second[k].get("time", 0) > first[k].get("time", 0) for k in ("supervisor", "watchdog"))
        trained = not (campaign or fresh_real) or (not stale and second.get("new_epochs", 0) >= 1 and second.get("finite_new_training") is True)
        if completion:
            trained = trained and second.get("completed_new_stages", 0) >= 1
        if fresh and trained and progress(second) != progress(first):
            receipt = {"pod_id": r["pod_id"], "launch_id": r["launch_id"], "run_id": run_id,
                       "progress_after_disconnect": True, "observations": [first, second],
                       **({"new_training_after_import_verified": True} if campaign else {}),
                       "observations_sha256": digest([first, second]),
                       "scope": "Independent SSH sessions closed between observations; stage/epoch/event progress and advancing remote heartbeats."}
            local = PROJECT / "runs" / run_id
            atomic_json(local / "detached_verification.json", receipt)
            client = ssh_connect(Runpod().pod(r["pod_id"]), STATE / "runpod_ed25519")
            try:
                sftp = client.open_sftp()
                temp = r["remote_root"] + "/.detached_verification.json.partial"
                upload_json(sftp, temp, receipt)
                command(client, "python3 -c " + shlex.quote("from pathlib import Path; Path(" + repr(temp) + ").replace(" + repr(r["remote_root"] + "/detached_verification.json") + ")"))
            finally:
                client.close()
            return receipt
        if campaign and attempt % 10 == 0:
            print(json.dumps({"run_id": run_id, "startup_health": "waiting_for_finite_new_training",
                              "new_epochs": second.get("new_epochs", 0)}), flush=True)
    raise RuntimeError("No real stage/epoch progress observed across SSH disconnection within the health check")


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
            workflow = read_json(PROJECT / "runs" / run_id / "config.json", {}).get("workflow")
            if workflow in ("paired", "mixed_campaign", "completion_comparison"):
                child = result["remote"].get("status", {}).get("active_child")
                if child:
                    allowed = (run_id + "-continued", run_id + "-reset-s0")
                    if workflow == "mixed_campaign":
                        from ..workflow.mixed_campaign import job_roots
                        allowed = [p.name for p in job_roots(Path(run_id)).values()]
                    elif workflow == "completion_comparison":
                        from ..workflow.completion import child_roots
                        arm_c = read_json(PROJECT / "runs" / run_id / "config.json", {})
                        allowed = [p.name for p in child_roots(Path(run_id), arm_c).values()]
                    if child not in allowed:
                        raise RuntimeError("Invalid paired child status association")
                    child_root = str(Path(r["remote_root"]).parent).replace("\\", "/") + "/" + child
                    snapshot = json.loads(command(client, "python3 -c " + shlex.quote(_active_child_snapshot_script(child_root)), timeout=30))
                    result["active_child"] = snapshot["status"]
                    result["log_tail"] = snapshot["log_tail"]
        finally:
            client.close()
    elif terminal_state(pod["status"]):
        from huggingface_hub import hf_hub_download
        for name in (("status.json", "watchdog_exit.json", "acceptance.json") if r.get("workflow") == "gradient_diagnostic" else ("status.json", "watchdog_exit.json")):
            try:
                p = hf_hub_download(r["repos"]["dataset"], f"{run_id}/{name}", repo_type="dataset", force_download=True)
                result[name] = read_json(p)
            except Exception:
                pass
        # Observation of a stopped provider state closes compute once. Storage remains open.
        stopped, updates = now(), {}
        if r.get("workflow") in ("mixed_campaign", "fresh_real", "completion_comparison", "gradient_diagnostic") or r.get("storage_backend") == "pod_persistent":
            stopped, proof = _provider_shutdown(api, pod, r)
            atomic_json(PROJECT / "runs" / run_id / "provider_shutdown.json", proof)
            result["provider_shutdown"] = proof
            updates = {"provider_shutdown_verified": True, "stop_observed_at": proof["observed_at"]}
        if r.get("workflow") == "gradient_diagnostic":
            accepted = result.get("acceptance.json")
            validation_end = min(stopped, accepted["time"]) if accepted else stopped
            updates["validation_estimated_usd"] = max(0, validation_end-r["created_at"]) / 3600 * r["guard_hourly_usd"]
        update_ledger(r, state="exited", stopped_at=stopped, volume_retained_since=stopped, **updates,
                      volume_hourly_usd=r.get("volume_hourly_usd", VOLUME_HOURLY))
    atomic_json(PROJECT / "runs" / run_id / "last_status.json", result)
    return result


def cleanup(run_id):
    r = local_resources(run_id)
    local = PROJECT / "runs" / run_id
    if not read_json(local / "recovery_verified.json"):
        raise RuntimeError("Fetch and checksum-verify artifacts before removing the recovery volume")
    from ..hub import verify_recovered_files
    index = read_json(local / "artifact_index.json")
    if not index:
        raise RuntimeError("Artifact index missing; fetch the latest run state before cleanup")
    paired = read_json(local / "config.json", {}).get("workflow") == "paired"
    campaign = read_json(local / "config.json", {}).get("workflow") == "mixed_campaign"
    completion = read_json(local / "config.json", {}).get("workflow") == "completion_comparison"
    pod_storage = r.get("storage_backend") == "pod_persistent"
    if pod_storage and r["volume_id"] != r["pod_id"]:
        raise RuntimeError("Pod persistent-disk owner differs from the registered Pod")
    if completion:
        from ..workflow.completion import verify_recovery
        verify_recovery(local)
        verify_recovered_files(local, index["files"])
        if any(v["metadata_only"] for v in read_json(local / "recovery_verified.json")["children"].values()):
            from ..workflow.smollm import enabled
            shutdown_check = "independent_watchdog_and_provider_stop_verified" if enabled(read_json(local / "config.json")) else "independent_budget_guard_and_provider_stop_verified"
            if not read_json(local / "validation_results.json", {}).get("checks", {}).get(shutdown_check):
                raise RuntimeError("Completion cleanup requires full recovery or validated private restores")
    elif campaign:
        from ..workflow.mixed_campaign import verify_campaign_recovery
        verify_campaign_recovery(local)
        verify_recovered_files(local, index["files"])
        if any(x.get("metadata_only") for x in read_json(local / "recovery_verified.json")["children"].values()):
            if not read_json(local / "validation_results.json", {}).get("checks", {}).get("independent_budget_guard_and_provider_stop_verified"):
                raise RuntimeError("Validate all three campaign cloud restorations and shutdown before metadata-only cleanup")
    elif paired:
        from ..workflow.paired import verify_pair_recovery
        verify_pair_recovery(local)
        verify_recovered_files(local, index["files"])
        if any(x.get("metadata_only") for x in read_json(local / "recovery_verified.json")["children"].values()):
            if not read_json(local / "validation_results.json", {}).get("checks", {}).get("independent_budget_guard_and_provider_stop_verified"):
                raise RuntimeError("Validate paired cloud restoration and shutdown before metadata-only cleanup")
    elif read_json(local / "recovery_verified.json").get("metadata_only"):
        if read_json(local / "config.json", {}).get("workflow") == "fresh_real":
            from ..workflow.fresh_validation import verify_recovery
        else:
            from ..continued import verify_recovery
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
        entry = latest_session(launches, r, p["id"])
        if entry.get("launch_id"):
            session = {**r, **entry}
            stopped = now()
            if campaign or completion or pod_storage or r.get("workflow") == "fresh_real":
                stopped, proof = _provider_shutdown(api, api.pod(p["id"]), session)
                atomic_json(local / ("provider_shutdown_" + entry["launch_id"] + ".json"), proof)
            update_ledger(session, state="exited", stopped_at=stopped, volume_retained_since=stopped,
                          volume_hourly_usd=r.get("volume_hourly_usd", VOLUME_HOURLY))
        api.request("DELETE", "/pods/" + p["id"])
    if any(p["id"] in owned for p in api.request("GET", "/pods")["pods"]):
        raise RuntimeError("Pod deletion not verified")
    if not pod_storage:
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
