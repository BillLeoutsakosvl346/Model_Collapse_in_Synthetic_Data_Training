"""Setup resolution and measured forecasts for the opt-in SGD comparison."""
from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path

import numpy as np

from ..common import atomic_json, digest, now, read_json, sha256


def benchmark_contract(c):
    """Effective measured settings, excluding smoke scope and artifact locations."""
    keys = ("completion_profile", "model", "model_revision", "dataset", "dataset_revision",
            "seed", "precision", "optimizer", "learning_rate", "learning_rate_policy", "scheduler_policy",
            "momentum", "dampening", "weight_decay", "nesterov", "maximize", "foreach", "fused",
            "microbatch", "effective_batch", "eval_batch", "generation_batch", "block_size", "new_tokens",
            "generation_decoder", "num_beams", "do_sample", "repetition_penalty", "update_diagnostics",
            "completion_arms", "secondary_corpora", "required_gpu", "required_image")
    return digest({"settings": {k: c.get(k) for k in keys},
                   "bank_manifest_sha256": c["prepared_bank"]["manifest_sha256"],
                   "calibration_receipt_sha256": c["calibration_reference"]["receipt_sha256"]})


def verify_benchmark_record(b, c, *, expected_runtime=None, require_observer=True):
    """A fetched timing file cannot stand in for an accepted matching benchmark."""
    if (b.get("benchmark_contract_sha256") != benchmark_contract(c)
            or b.get("calibration_receipt_sha256") != c["calibration_reference"]["receipt_sha256"]
            or not b.get("source_runtime_sha256")
            or expected_runtime is not None and b["source_runtime_sha256"] != expected_runtime):
        raise RuntimeError("SGD benchmark source runtime or scientific settings differ")
    if (b.get("benchmark_updates_discarded") != 0
            or any(b.get(k) != c[k] for k in ("microbatch", "effective_batch", "generation_batch"))
            or b.get("gpu") != c["required_gpu"]):
        raise RuntimeError("SGD benchmark arithmetic, batches or GPU differ")
    for key in ("training_target_tokens_per_second", "generation_new_tokens_per_second",
                "primary_target_tokens_per_second", "secondary_target_tokens_per_second", "full_secondary_targets"):
        if not isinstance(b.get(key), (int, float)) or not math.isfinite(b[key]) or b[key] <= 0:
            raise RuntimeError("SGD benchmark has invalid measured throughput")
    if set(b.get("full_primary_targets", {})) != {"validation", "test"} or any(
            type(v) is not int or v <= 0 for v in b["full_primary_targets"].values()):
        raise RuntimeError("SGD benchmark has invalid full evaluation counts")
    if require_observer:
        steps = b.get("step_seconds_by_arm", {})
        if (b.get("training_includes_update_observer") is not True
                or set(steps) != {"real", "mixed99", "synthetic"}
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in steps.values())
                or not 0 < b.get("calibration_physical_updates", 0) <= 192):
            raise RuntimeError("SGD benchmark lacks accepted source-attribution measurements")
    return b


def prepare(root, c, resources, bank):
    from . import sgd_profile
    from .sgd_calibration import calibrate
    root = Path(root)
    if c.get("calibration_required"):
        if c["kind"] != "smoke":
            raise RuntimeError("Full SGD runs require a frozen calibration before provisioning")
        if not (root / "setup_request.json").exists():
            atomic_json(root / "setup_request.json", c)
        def progress(details):
            atomic_json(root / "status.json", {"state": "running", "phase": "calibration", **details, "time": now()})
        receipt = calibrate(root, c, bank, progress=progress)
        c = sgd_profile.apply_calibration(c, receipt, sha256(root / "calibration/selection.json"))
        atomic_json(root / "config.json", c)
        atomic_json(root / "configuration_resolution.json", {
            "setup_request_sha256": digest(read_json(root / "setup_request.json")),
            "resolved_configuration_sha256": digest(c), "receipt_sha256": c["calibration_reference"]["receipt_sha256"]})
    sgd_profile.require_calibration(c)
    if c["kind"] == "smoke" and sha256(root / "calibration/selection.json") != c["calibration_reference"]["receipt_sha256"]:
        raise RuntimeError("Resolved calibration receipt changed")
    if c["kind"] == "full":
        from ..cloud.sgd_setup import restore_benchmark
        restore_benchmark(c, root, expected_runtime=resources["code_sha256"])
    return c


def benchmark(root, c, bank):
    path = Path(root) / "benchmark.json"
    if c["kind"] == "full":
        # Full roots intentionally have no calibration work. Reuse pinned accepted
        # measurements, without recalibration, extra optimizer steps or warmups.
        from ..cloud.sgd_setup import restore_benchmark
        resources = read_json(Path(root) / "resources.json", {})
        return restore_benchmark(c, root, expected_runtime=resources.get("code_sha256"))
    import torch
    from ..science import engine
    from ..completion_data import load_pairs
    from .completion import _primary, _visible_prompt
    previous = read_json(path)
    if previous:
        return verify_benchmark_record(previous, c, require_observer=False)
    histories = list((Path(root) / "calibration/.work").glob("*/seed_42/gen_*/training_history.json"))
    rates = [row["training_target_tokens_per_second"] for p in histories for row in read_json(p)]
    if not rates:
        raise RuntimeError("SGD benchmark requires actual calibration training, without extra warmup updates")
    model = engine.load_model(c)
    prompt, _, _ = load_pairs(bank, "cycle_01", c["generation_batch"])
    ids = engine.tensor(_visible_prompt(prompt, c["bos_token_id"]))
    durations = []
    model.eval()
    for _ in range(2):
        engine.sync(); start = time.perf_counter()
        with torch.inference_mode():
            model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **engine.decoding_settings(c))
        engine.sync(); durations.append(time.perf_counter() - start)
    primary = _primary(model, bank, "validation", c)
    tokens = np.load(Path(bank) / "secondary/fineweb_edu/tokens.npy", mmap_mode="r")[:32]
    masks = np.load(Path(bank) / "secondary/fineweb_edu/masks.npy", mmap_mode="r")[:32]
    secondary = engine.evaluate(model, tokens, c, masks=masks)
    result = {"training_target_tokens_per_second": float(np.median(rates)),
              "generation_new_tokens_per_second": len(prompt) * 64 / durations[-1],
              "primary_target_tokens_per_second": primary["scored_tokens"] / primary["seconds"],
              "secondary_target_tokens_per_second": secondary["scored_tokens"] / secondary["seconds"],
              "full_primary_targets": {s: read_json(Path(bank) / s / "manifest.json")["target_tokens"] for s in ("validation", "test")},
              "full_secondary_targets": sum(int(np.load(Path(bank) / "secondary" / s / "masks.npy", mmap_mode="r")[:, 1:].sum()) for s in c["secondary_corpora"]),
              "benchmark_updates_discarded": 0, "calibration_update_limit": 192,
              "generation_batch_seconds": durations, "generation_batch": c["generation_batch"],
              "microbatch": c["microbatch"], "effective_batch": c["effective_batch"],
              "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu", "training_includes_update_observer": False,
              "benchmark_contract_sha256": benchmark_contract(c),
              "calibration_receipt_sha256": c["calibration_reference"]["receipt_sha256"],
              "source_runtime_sha256": read_json(Path(root) / "resources.json")["code_sha256"]}
    atomic_json(path, result)
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result


def finish_benchmark(root, c):
    from .completion import child_roots
    from ..science.sgd_effects import verify_seal
    path = Path(root) / "benchmark.json"; b = read_json(path)
    verify_benchmark_record(b, c, require_observer=False)
    seconds = {}
    for arm, child in child_roots(root, c).items():
        records = []
        for cycle in range(1, c["generations"] + 1):
            stage = child / f"seed_42/gen_{cycle:02d}"
            seal = verify_seal(stage)
            records.extend(json.loads(line) for line in (stage / seal["records_path"]).read_text().splitlines())
        full = [r["step_seconds"] for r in records if len(r["batch_indices"]) == c["effective_batch"]]
        seconds[arm] = max(full)  # Includes source backward, FP32 shadow evaluation and diagnostic bookkeeping.
    b.update(step_seconds_by_arm=seconds, training_includes_update_observer=True,
             training_target_tokens_per_second=128 * 64 / max(seconds.values()),
             calibration_physical_updates=read_json(Path(root) / "calibration/physical_updates.json")["recorded_updates"])
    atomic_json(path, b)
    return b


def forecast(b, c, hourly):
    if not b.get("training_includes_update_observer"):
        raise RuntimeError("SGD full forecast requires measured update-attribution overhead")
    cycles = c["generations"]; stages = 3 * cycles; sets = stages + 1
    updates = cycles * math.ceil(10000 / 128)
    generated = cycles * (9900 + 10000)
    train = sum(b["step_seconds_by_arm"].values()) * updates
    seconds = (train + (generated + sets * 256) * 64 / b["generation_new_tokens_per_second"] +
               sets * sum(b["full_primary_targets"].values()) / b["primary_target_tokens_per_second"] +
               sets * b["full_secondary_targets"] / b["secondary_target_tokens_per_second"] +
               stages * 640000 / b["primary_target_tokens_per_second"])
    hours = (seconds * 1.5 + stages * 60 + 600) / 3600
    return {"estimated_hours": hours, "estimated_compute_and_running_storage_usd": hours * hourly,
            "unique_training_stages": stages, "epochs": stages, "optimizer_updates": 3 * updates,
            "generated_training_continuations": generated, "generated_training_tokens": generated * 64,
            "training_target_exposures": stages * 640000, "probe_sets": sets, "forecast_margin": 1.5,
            "update_attribution_included": True, "storage_and_backup_fixed_overhead_seconds": stages * 60 + 600}
