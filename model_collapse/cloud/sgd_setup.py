"""Verified setup inputs and local handoff; uses the shared Pod launcher."""
from __future__ import annotations

import shlex
import shutil
import tarfile
from pathlib import Path

from ..common import PROJECT, STATE, atomic_json, digest, read_json, sha256, verify_files


def verify_calibration_source(c):
    from ..workflow.sgd_profile import require_calibration
    from ..workflow.sgd_calibration import calibration_contract
    from ..artifacts.gradient_sources import Snapshot
    require_calibration(c)
    from ..common import load_secrets
    load_secrets()
    ref = c["calibration_reference"]
    if not ref.get("snapshot") or ref.get("relative_path") != "calibration/selection.json":
        raise RuntimeError("Full SGD requires its immutable private calibration snapshot")
    source = Snapshot(ref["snapshot"], STATE / "sgd-calibration-recovery")
    path = source.get(ref["relative_path"])
    receipt = read_json(path)
    if (sha256(path) != ref["receipt_sha256"] or receipt.get("selected_learning_rate") != c["learning_rate"]
            or receipt.get("contract_sha256") != calibration_contract(c) or not receipt.get("complete")):
        raise RuntimeError("Full SGD calibration source differs")
    return verify_benchmark_source(c, source=source)


def _benchmark_reference(c):
    ref = c.get("benchmark_reference") or {}
    calibration = c.get("calibration_reference") or {}
    if (ref.get("relative_path") != "benchmark.json" or len(ref.get("sha256", "")) != 64
            or not ref.get("snapshot") or ref["snapshot"] != calibration.get("snapshot")
            or not ref.get("source_runtime_sha256")):
        raise RuntimeError("Full SGD requires its pinned accepted setup benchmark")
    return ref


def verify_benchmark_source(c, *, source=None, expected_runtime=None):
    from ..artifacts.gradient_sources import Snapshot
    from ..workflow.sgd_runtime import verify_benchmark_record
    ref = _benchmark_reference(c)
    from ..common import load_secrets
    load_secrets()
    source = source or Snapshot(ref["snapshot"], STATE / "sgd-benchmark-recovery")
    path = source.get(ref["relative_path"])
    if sha256(path) != ref["sha256"]:
        raise RuntimeError("Pinned SGD benchmark checksum differs")
    value = read_json(path)
    if value.get("source_runtime_sha256") != ref["source_runtime_sha256"]:
        raise RuntimeError("Pinned SGD benchmark runtime differs")
    return verify_benchmark_record(value, c, expected_runtime=expected_runtime or ref["source_runtime_sha256"])


def restore_benchmark(c, root, *, expected_runtime=None):
    """Restore exact accepted timings into a fresh root; never rerun training."""
    from ..artifacts.gradient_sources import Snapshot
    from ..workflow.sgd_runtime import verify_benchmark_record
    root = Path(root)
    target = root / "benchmark.json"
    ref = _benchmark_reference(c)
    if target.exists():
        if sha256(target) != ref.get("sha256"):
            raise RuntimeError("Local full-run SGD benchmark differs from its pinned source")
        value = read_json(target)
        if value.get("source_runtime_sha256") != ref.get("source_runtime_sha256"):
            raise RuntimeError("Local full-run SGD benchmark runtime differs")
        return verify_benchmark_record(value, c, expected_runtime=expected_runtime or ref.get("source_runtime_sha256"))
    value = verify_benchmark_source(c, expected_runtime=expected_runtime)
    source = Snapshot(ref["snapshot"], STATE / "sgd-benchmark-recovery")
    root.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".partial")
    shutil.copyfile(source.get(ref["relative_path"]), temporary)
    if sha256(temporary) != ref["sha256"]:
        raise RuntimeError("Transferred SGD benchmark checksum differs")
    temporary.replace(target)
    atomic_json(root / "benchmark_reuse.json", {"source": ref, "calibration_reexecuted": False,
                "additional_optimizer_updates": 0, "scope": "Reuse accepted setup measurements; their calibration timing/update counts are historical evidence."})
    return value


def bank_cache(c):
    bank = Path(c["bank_local_path"])
    if not bank.is_absolute(): bank = PROJECT / bank
    ref = c["prepared_bank"]
    if sha256(bank / "manifest.json") != ref["manifest_sha256"]:
        raise RuntimeError("Local frozen native bank differs")
    if sha256(bank / "artifact_index.json") != ref["artifact_index_sha256"]:
        raise RuntimeError("Native bank cache index differs")
    files = read_json(bank / "artifact_index.json")["files"]
    from ..artifacts.paths import safe_relative
    for name in files:
        safe_relative(name)
    verify_files(bank, files)
    archive = STATE / ("sgd-bank-" + ref["artifact_index_sha256"] + ".tar.gz")
    record = read_json(archive.with_suffix(".json"), {})
    if not archive.exists() or record.get("sha256") != sha256(archive):
        with tarfile.open(archive, "w:gz") as tar:
            for relative in sorted(files):
                tar.add(bank / relative, arcname=relative)
            tar.add(bank / "artifact_index.json", arcname="artifact_index.json")
        atomic_json(archive.with_suffix(".json"), {"sha256": sha256(archive), "bank": ref["manifest_sha256"]})
    return archive


def upload_bank(client, sftp, archive, c):
    from .execution import command
    remote = "/workspace/sgd-bank-cache.tar.gz"
    destination = "/workspace/completion_banks/" + c["prepared_bank"]["run_id"]
    sftp.put(str(archive), remote)
    check = "import hashlib; assert hashlib.sha256(open(" + repr(remote) + ",'rb').read()).hexdigest()==" + repr(sha256(archive))
    command(client, "python3 -c " + shlex.quote(check))
    command(client, "mkdir -p " + shlex.quote(destination) + " && tar -xzf " + shlex.quote(remote) + " -C " + shlex.quote(destination))


def adopt_resolved_config(root):
    root = Path(root)
    c = read_json(root / "config.json")
    from ..workflow.sgd_profile import require_calibration
    require_calibration(c)
    if sha256(root / "calibration/selection.json") != c["calibration_reference"]["receipt_sha256"]:
        raise RuntimeError("Fetched resolved rate differs from the checked calibration receipt")
    atomic_json(root / "local_configuration_adoption.json", {
        "original_setup_configuration_sha256": read_json(root / "resources.json")["configuration_sha256"],
        "resolved_configuration_sha256": digest(c), "recovery": read_json(root / "recovery_snapshot.json"),
        "ownership_registry_changed": False})


def write_full_configuration(root, destination):
    """Only called after accepted setup recovery; never provisions or launches."""
    import yaml
    from ..workflow.sgd_profile import apply_calibration, validate, CHECKS
    root, destination = Path(root), Path(destination)
    accepted = read_json(root / "validation_results.json", {})
    if (not all(accepted.get("checks", {}).get(k) is True for k in CHECKS) or
            accepted.get("source_code_sha256") != read_json(root / "resources.json")["code_sha256"]):
        raise RuntimeError("Accept the recovered SGD smoke before publishing its full configuration")
    raw = yaml.safe_load(destination.read_text())
    receipt = read_json(root / "calibration/selection.json")
    resolved = apply_calibration(raw, receipt, sha256(root / "calibration/selection.json"))
    snapshot = read_json(root / "recovery_snapshot.json")
    resolved["calibration_reference"].update(relative_path="calibration/selection.json", snapshot={
        k: snapshot[k] for k in ("run_id", "repos", "revisions", "artifact_index_sha256")})
    from ..workflow.sgd_runtime import verify_benchmark_record
    benchmark_path = root / "benchmark.json"
    indexed = read_json(root / "artifact_index.json", {}).get("files", {}).get("benchmark.json", {})
    if (sha256(root / "artifact_index.json") != snapshot["artifact_index_sha256"]
            or indexed.get("sha256") != sha256(benchmark_path)):
        raise RuntimeError("Accepted SGD benchmark is not pinned by the recovered setup index")
    benchmark = verify_benchmark_record(read_json(benchmark_path), resolved,
                                        expected_runtime=accepted["source_code_sha256"])
    resolved["benchmark_reference"] = {"snapshot": resolved["calibration_reference"]["snapshot"],
        "relative_path": "benchmark.json", "sha256": sha256(benchmark_path),
        "source_runtime_sha256": benchmark["source_runtime_sha256"]}
    validate(resolved)
    destination.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    return {"learning_rate": resolved["learning_rate"], "configuration_sha256": digest(resolved),
            "full_experiment_launched": False, "calibration_snapshot": resolved["calibration_reference"]["snapshot"]}
