"""A longer repeated-real control preserves the bank and uses a flat schedule."""
import copy
import io
from pathlib import Path

import pytest
import torch
import yaml

from model_collapse.completion_data import bank_contract
from model_collapse.workflow import completion
from model_collapse.workflow.profiles import protocol_fingerprint
from model_collapse.workflow.selection import training_contract


def configs():
    root = Path("comparisons/fineweb_completion64_repeated_real_constant20")
    return [yaml.safe_load((root / f"{kind}.yaml").read_text()) for kind in ("full", "smoke")]


def test_real_only_twenty_epochs_keep_the_exact_frozen_bank():
    full, smoke = configs()
    original = yaml.safe_load(Path("comparisons/fineweb_completion64_repeated_chunk/full.yaml").read_text())
    for c in (full, smoke):
        completion.validate_config(c)
        assert completion.arm_specs(c) == [{"id": "real", "synthetic_fraction": 0.0}]
        assert bank_contract(c) == bank_contract(original)
        assert c["learning_rate"] == c["min_learning_rate"] == 5e-6
        assert "constant_learning_rate_all_updates" in completion.acceptance_checks(c)
        assert "recursive_same_prompt_targets" not in completion.acceptance_checks(c)
        assert "generation_shard_recovery" not in completion.acceptance_checks(c)
        assert "private_restore" in completion.acceptance_checks(c)
        assert completion.allocation_id(c).endswith("real_20epochs")
    assert protocol_fingerprint(full) == protocol_fingerprint(smoke)
    assert protocol_fingerprint(full) != protocol_fingerprint(original)
    assert completion.benchmark_filename(full) != completion.benchmark_filename(original)
    for cycle in range(1, 21):
        assert completion.source_collection(full, cycle) == "cycle_01"
    with pytest.raises(ValueError):
        completion.source_collection(full, 21)
    assert completion.required_backup_bytes(full) == 20 * 1_530_000_000
    assert completion.required_backup_bytes(smoke) == 23 * 1_530_000_000


@pytest.mark.parametrize("changes", [
    {"training_cycles": 21}, {"training_cycles": True},
    {"endpoint_arms": []}, {"endpoint_arms": ["real", "real"]},
    {"endpoint_arms": ["unknown"]}, {"generations": 10},
    {"training_chunk_policy": "fresh_per_cycle", "fixed_chunk": None},
])
def test_invalid_control_profiles_are_rejected(changes):
    c = {**configs()[0], **changes}
    with pytest.raises(ValueError):
        completion.validate_config(c)


def test_constant_lr_at_every_update_and_after_state_restoration():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    opt = torch.optim.AdamW([parameter], lr=5e-6, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1580, eta_min=5e-6)
    for step in range(1580):
        assert opt.param_groups[0]["lr"] == 5e-6
        opt.zero_grad(); parameter.square().sum().backward(); opt.step(); scheduler.step()
        assert opt.param_groups[0]["lr"] == 5e-6
        if step == 396:
            stream = io.BytesIO()
            torch.save({"optimizer": opt.state_dict(), "scheduler": scheduler.state_dict()}, stream)
            stream.seek(0)
            state = torch.load(stream, weights_only=False)
            opt = torch.optim.AdamW([parameter], lr=5e-6, weight_decay=.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1580, eta_min=5e-6)
            opt.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
    assert scheduler.last_epoch == 1580


def test_twenty_epoch_forecast_has_no_generated_training_targets():
    c = configs()[0]
    bench = {"full_primary_targets": {"validation": 468608, "test": 457664},
             "full_secondary_targets": 10000, "training_target_tokens_per_second": 10000,
             "generation_new_tokens_per_second": 10000, "primary_target_tokens_per_second": 10000,
             "secondary_target_tokens_per_second": 10000}
    result = completion.forecast(bench, c, .72)
    assert result["unique_training_stages"] == result["epochs"] == 20
    assert result["training_target_exposures"] == 12_800_000
    assert result["generated_training_continuations"] == 0
    assert result["probe_sets"] == 21
    altered = copy.deepcopy(c); altered["min_learning_rate"] = 1e-6
    assert training_contract(completion.arm_config(c, "real")) != training_contract(completion.arm_config(altered, "real"))


def test_persistent_disk_fallback_is_limited_to_single_repeated_endpoint():
    full, smoke = configs()
    for c in (full, smoke):
        c["storage_backend"] = "pod_persistent"
        completion.validate_config(c)
    changed = copy.deepcopy(full)
    changed["endpoint_arms"] = ["real", "synthetic"]
    with pytest.raises(ValueError, match="one repeated endpoint"):
        completion.validate_config(changed)
