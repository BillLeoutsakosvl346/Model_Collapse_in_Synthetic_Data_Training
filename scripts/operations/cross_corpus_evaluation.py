"""Explicit paid evaluation of existing checkpoints. Never starts training.

Run manually with --execute after preparing the frozen corpus bank.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import shutil
import tarfile
import time
from pathlib import Path

from filelock import FileLock
from huggingface_hub import get_token

from model_collapse import hub, remote
from model_collapse.common import PROJECT, STATE, atomic_json, now, read_json, sha256, verify_files

SOURCE = "20261106T151257Z-full-fresh-real"
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
GPU = "NVIDIA RTX PRO 4000 Blackwell"
CAP = .50


def prepare_inputs(bank, local):
    """Verify frozen inputs and the exact original checkpoint/evaluation evidence."""
    manifest = read_json(bank / "manifest.json")
    verify_files(bank, manifest["files"])
    source = PROJECT / "runs" / SOURCE / "completion_metadata"
    receipt = read_json(PROJECT / "docs/validation/2026-11-06_fineweb-full-completion-status.json")
    if sha256(source / "artifact_index.json") != receipt["artifact_index_sha256"]:
        raise RuntimeError("Historical source index changed")
    index = read_json(source / "artifact_index.json")
    verify_files(source, {"config.json": index["files"]["config.json"]["sha256"]})
    c = read_json(source / "config.json")
    metrics = {}
    for label, cycle in (("pretrained", 0), ("fineweb_epoch1", 1)):
        relative = f"seed_42/gen_{cycle:02d}/metrics.json"
        verify_files(source, {relative: index["files"][relative]["sha256"]})
        metrics[label] = read_json(source / relative)
    files, collections = {}, {}
    for name in manifest["configuration"]["sources"]:
        collections[name] = manifest["collections"][f"{name}/test"]
        destination = local / "inputs" / name
        destination.mkdir(parents=True, exist_ok=True)
        for file in ("tokens.npy", "masks.npy", "manifest.json"):
            original = bank / name / "test" / file
            shutil.copyfile(original, destination / file)
            files[f"inputs/{name}/{file}"] = sha256(original)
            if name == "fineweb_edu" and file.endswith(".npy"):
                if sha256(original) != index["files"][f"data/test/{file}"]["sha256"]:
                    raise RuntimeError("FineWeb test arrays differ from historical evaluation")
    spec = {"source_run_id": SOURCE, "source_repos": remote.local_resources(SOURCE)["repos"],
            "source_revisions": receipt["revisions"], "checkpoint_index_sha256": sha256(source / "artifact_index.json"),
            "bank_manifest_sha256": sha256(bank / "manifest.json"), "bank_contract": manifest["contract"],
            "model_config": c, "input_files": files, "collections": collections,
            "expected_weights": {"pretrained": metrics["pretrained"]["weights_sha256"],
                                 "fineweb_epoch1": metrics["fineweb_epoch1"]["training"]["selected_weights_sha256"]},
            "original_fineweb_metrics": metrics, "evaluator_sha256": sha256(PROJECT / "scripts/data/evaluate.py"),
            "training_updates_requested": 0,
            "fineweb_reproduction_nll_tolerance": 1e-4,
            "source_gpu": GPU,
            "evaluation_hardware_policy": "Both checkpoints on one compatible GPU. Exact checkpoint/array hashes required. Historical loss reproduction is checked at 1e-4 NLL on matching hardware. Cross-GPU drift is recorded separately."}
    resolved = hub.resolve_snapshot(spec["source_repos"], spec["source_revisions"])
    if resolved != spec["source_revisions"]:
        raise RuntimeError("Source backup commits differ")
    shutil.copyfile(source / "artifact_index.json", local / "checkpoint_index.json")
    shutil.copyfile(PROJECT / "scripts/data/evaluate.py", local / "evaluator.py")
    shutil.copyfile(bank / "manifest.json", local / "bank_manifest.json")
    atomic_json(local / "experiment.json", spec)
    atomic_json(local / "config.json", {"workflow": "cross_corpus_evaluation", "training_updates": 0})
    return spec


def deploy(api, resources, local, archive, key, *, watchdog_started=False):
    client = None
    try:
        cutoff = remote.provisioning_deadline(resources)
        while now() < cutoff:
            pod = api.pod(resources["pod_id"])
            if pod["status"] == "RUNNING" and pod.get("ssh", {}).get("direct"):
                try:
                    client = remote.ssh_connect(pod, key)
                    break
                except OSError:
                    pass
            time.sleep(5)
        if client is None:
            raise RuntimeError("Evaluation Pod did not become SSH-ready")
        if pod["gpu"]["id"] != resources["gpu"] or pod["dataCenterId"] != resources["data_center"]:
            raise RuntimeError("Evaluation hardware read-back differs")
        if pod["cost"] > resources["quoted_gpu_hourly_usd"] + .01:
            raise RuntimeError("Provider rate exceeds admitted price")
        root = resources["remote_root"]
        remote.command(client, "mkdir -p /workspace/project " + shlex.quote(root))
        sftp = client.open_sftp()
        sftp.get_channel().settimeout(120)
        sftp.put(str(archive), "/workspace/project-code.tar.gz")
        remote.command(client, "tar -xzf /workspace/project-code.tar.gz -C /workspace/project")
        check = ("import pathlib,json,hashlib; p=pathlib.Path('/workspace/project'); "
                 "m=json.loads((p/'source_bundle.json').read_text()); "
                 "assert m['runtime_sha256']==" + repr(resources["code_sha256"]) + "; "
                 "assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in m['files'].items())")
        remote.command(client, "python3 -c " + shlex.quote(check))
        for name in ("resources.json", "config.json", "experiment.json", "checkpoint_index.json", "evaluator.py", "bank_manifest.json"):
            sftp.put(str(local / name), root + "/" + name)
        with sftp.open("/run/iib-week1-secrets", "w") as f:
            f.write("export RUNPOD_API_KEY=" + shlex.quote(api.key) + "\nexport HF_TOKEN=" +
                    shlex.quote(os.environ.get("HF_TOKEN") or get_token()) + "\n")
        sftp.chmod("/run/iib-week1-secrets", 0o600)
        prefix = ("cd /workspace/project && . /run/iib-week1-secrets && "
                  "export PYTHONPATH=/workspace/project HF_HOME=/workspace/hf-cache TOKENIZERS_PARALLELISM=false HF_HUB_DOWNLOAD_TIMEOUT=120 && ")
        guard = "setsid python3 -m model_collapse watchdog --root " + shlex.quote(root) + " >/workspace/watchdog.log 2>&1 </dev/null &"
        if not watchdog_started:
            remote.command(client, prefix + "sh -c " + shlex.quote(guard), timeout=30)
        payload = local / "inputs-transfer.tar.gz"
        with tarfile.open(payload, "w:gz") as tar:
            tar.add(local / "inputs", arcname="inputs")
        sftp.put(str(payload), "/workspace/inputs.tar.gz")
        remote.command(client, "tar -xzf /workspace/inputs.tar.gz -C " + shlex.quote(root))
        payload.unlink()
        remote.command(client, prefix + "python3 -m venv --system-site-packages /opt/iib-venv && /opt/iib-venv/bin/pip install --progress-bar off -e '.[test]' >/workspace/install.log 2>&1", timeout=900)
        script = "import torch,json; print(json.dumps({'torch':torch.__version__,'cuda':torch.cuda.is_available(),'bf16':torch.cuda.is_bf16_supported(),'gpu':torch.cuda.get_device_name()}))"
        actual = json.loads(remote.command(client, prefix + "/opt/iib-venv/bin/python -c " + shlex.quote(script)))
        if not actual["cuda"] or not actual["bf16"] or actual["gpu"] != resources["gpu"]:
            raise RuntimeError("CUDA/BF16/GPU verification failed")
        atomic_json(local / "gpu_check.json", actual)
        remote.upload_json(sftp, root + "/gpu_check.json", actual)
        remote.command(client, prefix + "/opt/iib-venv/bin/pip freeze >" + shlex.quote(root + "/requirements-tested.txt"))
        worker = "setsid /opt/iib-venv/bin/python " + shlex.quote(root + "/evaluator.py") + " --root " + shlex.quote(root) + " >" + shlex.quote(root + "/pipeline.log") + " 2>&1 </dev/null &"
        remote.command(client, prefix + "sh -c " + shlex.quote(worker), timeout=30)
    finally:
        if client:
            client.close()


def execute(bank):
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-cross-corpus-evaluation"
    local = PROJECT / "runs" / run_id
    local.mkdir(parents=True)
    spec = prepare_inputs(bank, local)
    check = remote.preflight(probe=True)
    api = remote.Runpod()
    if any(p["status"] not in ("EXITED", "TERMINATED", "ERROR") for p in api.request("GET", "/pods")["pods"]):
        raise RuntimeError("Another Pod is active")
    compatible = {GPU, "NVIDIA RTX 2000 Ada Generation", "NVIDIA RTX 4000 Ada Generation",
                  "NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090", "NVIDIA L4", "NVIDIA RTX A4000", "NVIDIA RTX A5000"}
    choices = [p for p in check["gpu_candidates"] if p["gpu"] in compatible and p["hourly_usd"] <= .60]
    if not choices:
        raise RuntimeError("Affordable compatible GPU unavailable, evaluation remains pending")
    choice = choices[0]
    if choice["hourly_usd"] > .60:
        raise RuntimeError("Matching GPU price exceeds evaluation admission")
    template = next(t for t in api.request("GET", "/catalog/templates?source=official")["templates"] if t["id"] == "runpod-torch-v280")
    if template["image"] != IMAGE:
        raise RuntimeError("Official PyTorch image differs")
    ledger = remote.ledger()
    allowance = min(CAP, remote.task_budget(ledger) - remote.accrued(ledger) - remote.RESERVE -
                    remote.historical_running_storage(ledger, 1), check["account"]["clientBalance"] - remote.RESERVE)
    if allowance < .35:
        raise RuntimeError("Insufficient evaluation allowance with reserve and historical storage")
    archive, code = remote.source_archive()
    key = remote.register_key(api)
    atomic_json(local / "authorization.json", {"scope": "Compare existing pretrained and epoch-one checkpoints on five held-out corpora", "no_training": True,
                "session_cap_usd": CAP, "cumulative_ceiling_usd": remote.task_budget(ledger), "reserve_usd": remote.RESERVE,
                "preflight": check, "evaluator_sha256": spec["evaluator_sha256"], "time": now()})
    volume_time = now()
    volume = api.request("POST", "/network-volumes", {"name": "iib-" + run_id, "size": 10, "dataCenter": choice["data_center"], "type": "STANDARD"})
    atomic_json(local / "pending_volume.json", volume)
    verified_volume = api.request("GET", "/network-volumes/" + volume["id"])
    if verified_volume["size"] != 10 or verified_volume["dataCenter"] != choice["data_center"]:
        raise RuntimeError("Volume read-back differs")
    created = now()
    payload = {"name": "iib-" + run_id, "cloud": "SECURE", "image": IMAGE, "disk": 20,
               "gpu": {"id": choice["gpu"], "count": 1, "minCudaVersion": "12.8"}, "dataCenterIds": [choice["data_center"]],
               "mounts": {"network": [{"volumeId": volume["id"], "path": "/workspace"}]},
               "startSsh": True, "startJupyter": False, "ports": ["22/tcp"]}
    try:
        pod = api.request("POST", "/pods", payload)
    except Exception:
        matches = [p for p in api.request("GET", "/pods")["pods"] if p["name"] == "iib-" + run_id]
        if len(matches) != 1:
            if not matches:
                api.request("DELETE", "/network-volumes/" + volume["id"])
                atomic_json(local / "empty_resource_cleanup.json", {"volume_id": volume["id"], "deleted_at": now()})
            raise RuntimeError("Pod creation failed or uncertain, no blind retry") from None
        pod = matches[0]
    storage = 10 * .07 / 730
    resources = {"run_id": run_id, "launch_id": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
                 "kind": "analysis", "workflow": "cross_corpus_evaluation", "pod_id": pod["id"], "volume_id": volume["id"],
                 "data_center": choice["data_center"], "gpu": choice["gpu"], "image": IMAGE, "created_at": created,
                 "quoted_gpu_hourly_usd": choice["hourly_usd"], "guard_hourly_usd": (choice["hourly_usd"] + storage + remote.CONTAINER_HOURLY) * 1.15,
                 "compute_hourly_usd": (choice["hourly_usd"] + remote.CONTAINER_HOURLY) * 1.15,
                 "volume_hourly_usd": storage, "network_volume_gb": 10, "volume_created_at": volume_time,
                 "session_allowance_usd": allowance, "repos": check["hugging_face"]["repos"], "code_sha256": code,
                 "remote_root": "/workspace/runs/" + run_id, "starting_balance_usd": check["account"]["clientBalance"]}
    try:
        remote.register_resources(resources)
        remote.update_ledger(resources, state="provisioning")
        print(json.dumps({"run_id": run_id, "pod_id": pod["id"], "evaluation_only": True}), flush=True)
        deploy(api, resources, local, archive, key)
        remote.update_ledger(resources, state="running")
        from model_collapse.supervisor import deadline
        while api.pod(pod["id"])["status"] not in ("EXITED", "TERMINATED"):
            if now() > deadline(resources) + 120:
                raise RuntimeError("Evaluation exceeded local monitoring window")
            time.sleep(12)
    except BaseException:
        api.stop(pod["id"])
        raise
    finally:
        stopped = api.pod(pod["id"])
        if stopped["status"] in ("EXITED", "TERMINATED"):
            timestamp, proof = remote._provider_shutdown(api, stopped, resources)
            atomic_json(local / "provider_shutdown.json", proof)
            remote.update_ledger(resources, state="exited", stopped_at=timestamp, volume_retained_since=timestamp)
    recovered = hub.fetch(run_id, resources["repos"], local, metadata_only=False)
    atomic_json(local / "recovery_verified.json", recovered)
    completion = read_json(local / "completion.json", {})
    if not completion.get("complete") or completion.get("training_updates_executed") != 0:
        raise RuntimeError("Evaluation failed, preserve its recovered evidence and inspect logs")
    if read_json(local / "watchdog_exit.json", {}).get("reason") != "complete":
        raise RuntimeError("Watchdog completion receipt missing")
    print(json.dumps({"cleanup": remote.cleanup(run_id), "comparison": read_json(local / "analysis/comparison.json")}, indent=2), flush=True)
    return run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=PROJECT / "data/corpus_bank/cross_corpus_v1")
    parser.add_argument("--execute", action="store_true", help="Provision a temporary GPU for evaluation only")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Paid execution requires explicit --execute")
    with FileLock(str(STATE / "launch.lock"), timeout=0):
        execute(args.bank)


if __name__ == "__main__":
    main()
