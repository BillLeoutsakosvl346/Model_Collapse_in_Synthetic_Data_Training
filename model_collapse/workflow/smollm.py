"""Model-specific completion policies; historical OPT defaults stay unchanged."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from ..common import STATE, read_json, sha256, verify_files

MODEL = "HuggingFaceTB/SmolLM2-135M"
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
PROFILE = "smollm2_completion64_v1"
ALLOCATION = "smollm2_completion64_fractions"
ARMS = [dict(id=name, synthetic_fraction=f) for name, f in (
    ("real", 0.), ("mixed50", .5), ("mixed80", .8), ("mixed95", .95),
    ("mixed99", .99), ("mixed99p9", .999), ("synthetic", 1.))]
CHECKS = {"two_optimizer_updates", "native_special_tokens_and_masks", "preceding_model_and_optimizer",
          "finite_direct_conditional_nll", "exact_mixed_targets", "terminal_checkpoint_reload",
          "private_terminal_restore", "seven_fraction_offline_checks", "detached_progress_verified",
          "independent_watchdog_and_provider_stop_verified"}


def enabled(c):
    return c.get("completion_profile") == PROFILE


def specs(c):
    if c.get("completion_arms") != ARMS:
        raise ValueError("SmolLM2 requires the seven fixed ordered fractions")
    if c.get("kind") == "smoke":
        return [ARMS[1]]
    return ARMS


def validate(c):
    from .configuration import validate_training_numbers
    validate_training_numbers(c)
    specs(c)
    from ..science.decoding import decoding_settings
    decoding_settings(c)
    expected = {"model": MODEL, "model_revision": REVISION, "dataset": "HuggingFaceFW/fineweb-edu",
        "dataset_revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", "dataset_config": "sample-10BT",
        "epochs": 1, "checkpoint_selection": "final_epoch", "initialization": "continued",
        "optimizer_policy": "carry_across_cycles", "full_cycles": 10, "full_train_pairs": 10000,
        "max_sequence_length": 129, "block_size": 64, "new_tokens": 64, "num_beams": 5,
        "repetition_penalty": 1., "learning_rate": 5e-6, "min_learning_rate": 1e-6,
        "weight_decay": .01, "microbatch": 32, "effective_batch": 128, "generation_batch": 32,
        "eval_batch": 32, "precision": "bf16", "seed": 42, "storage_backend": "network",
        "backup_capacity_gb": 200, "backup_policy": "runpod_s3_pinned_hf_index",
        "mixture_selection_policy": "nested_seedsequence_6401_v1"}
    if any(c.get(k) != v for k,v in expected.items()):
        raise ValueError("SmolLM2 scientific settings differ from the matched protocol")
    if c.get("workflow") not in ("completion_comparison", "completion_arm"):
        raise ValueError("SmolLM2 uses the shared completion workflow")
    kind = c.get("kind")
    target = {"full": (10, 10000, None, 256, 200), "smoke": (2, 128, 32, 32, 30)}.get(kind)
    if target != tuple(c.get(k) for k in ("generations", "train_pairs_per_cycle", "evaluation_pairs", "probe_count", "network_volume_gb")):
        raise ValueError("SmolLM2 full/fast-smoke scope differs")
    if c.get("generation_shard") != (32 if kind == "smoke" else 256) or c.get("decoding_arms") or c.get("endpoint_arms") or c.get("training_chunk_policy", "fresh_per_cycle") != "fresh_per_cycle":
        raise ValueError("SmolLM2 requires fresh chunks and fixed five-beam decoding")
    if kind == "smoke" and (c.get("validation_max_seconds") != 900 or c.get("max_session_usd") != .5):
        raise ValueError("Fast SmolLM2 validation requires its 15-minute/$0.50 limits")
    if c.get("workflow") == "completion_arm":
        spec = next((x for x in specs(c) if x["id"] == c.get("arm_id")), None)
        data_kind = "real" if spec and spec["synthetic_fraction"] == 0 else "synthetic" if spec and spec["synthetic_fraction"] == 1 else "mixed"
        if not spec or c.get("data_kind") != data_kind or c.get("synthetic_fraction") != spec["synthetic_fraction"]:
            raise ValueError("SmolLM2 child differs from its registered condition")


def resolve_backup(c, api):
    """Never mistake the future capacity declaration for allocated storage."""
    from copy import deepcopy
    c=deepcopy(c)
    if c.get('kind') != 'full':
        return c
    store=c.get('backup_store') or read_json(STATE/'smollm2_backup_store.json')
    if not store:
        raise RuntimeError('Full SmolLM2 launch needs a verified 200 GB private backup volume; setup does not allocate full-run storage. Configure .state/smollm2_backup_store.json before launch.')
    from ..cloud.s3_artifacts import validate_store
    validate_store(store)
    volume=api.request('GET','/network-volumes/'+store['bucket'])
    if volume['size'] < c['backup_capacity_gb'] or volume['dataCenter'] != store['region']:
        raise RuntimeError('SmolLM2 full backup capacity/region is below the declared 200 GB requirement')
    c['backup_store']=store
    return c


def retention_forecast(c, backup_capacity_bytes, reserve_usd):
    """Cover both full volumes for seven days without counting the reserve twice."""
    working = c['network_volume_gb'] * .07 / 730 * 168
    backup = backup_capacity_bytes / 1_000_000_000 * .07 / 730 * 168
    return {'seven_day_volume_storage_usd': working,
            'seven_day_backup_storage_usd': backup,
            'seven_day_combined_storage_usd': working + backup,
            'retention_above_existing_reserve_usd': max(0., working + backup - reserve_usd)}


def resolve_inputs(c):
    from copy import deepcopy
    from ..science.completion_bank_v2 import bank_contract
    c = deepcopy(c)
    ref = c.get("prepared_bank") or read_json(STATE / "smollm2_completion64_bank.json")
    if not ref or ref.get("contract") != bank_contract(c):
        raise RuntimeError("Prepare the model-native SmolLM2 bank before provisioning")
    if any(len(v) != 40 for v in ref["revisions"].values()):
        raise RuntimeError("SmolLM2 bank requires immutable backup commits")
    local = Path(c["bank_local_path"])
    if not local.is_absolute():
        from ..paths import PROJECT
        local = PROJECT / local
    manifest_path = local / "manifest.json"
    if not manifest_path.exists():
        from huggingface_hub import hf_hub_download
        manifest_path = Path(hf_hub_download(ref["repos"]["dataset"], ref["run_id"] + "/manifest.json", repo_type="dataset", revision=ref["revisions"]["dataset"]))
    if sha256(manifest_path) != ref["manifest_sha256"]:
        raise RuntimeError("SmolLM2 native bank manifest differs")
    manifest = read_json(manifest_path)
    if manifest.get("contract") != ref["contract"] or any(manifest["collections"].get(f"cycle_{i:02d}", {}).get("pairs") != 10000 for i in range(1,11)):
        raise RuntimeError("SmolLM2 bank is incomplete")
    for split in ("validation", "test"):
        if manifest["collections"][split]["pairs"] < 256:
            raise RuntimeError("Insufficient native held-out pairs")
    c["prepared_bank"] = ref
    return c


def probe_diagnostics(values, ids):
    from ..science.metrics import probe_diagnostics as legacy
    result = {k:v for k,v in legacy(values, newline_token_id=-1).items()
              if not k.startswith("newline_") and k != "all_newline_outputs"}
    mask = np.isin(values, ids)
    counts = mask.sum(axis=1)
    result.update(newline_token_count=int(counts.sum()), newline_token_share=float(mask.mean()),
        newline_heavy_outputs_ge32=int((counts >= 32).sum()), all_newline_outputs=int((counts == 64).sum()),
        newline_policy="tokens_containing_lf_v1")
    return result
