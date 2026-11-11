"""The repeated control changes source reuse, not training or evaluation rules."""
import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from model_collapse.common import atomic_json, read_json, sha256
from model_collapse.workflow import completion
from model_collapse.workflow.profiles import protocol_fingerprint, require_acceptance


def configurations():
    base = Path("comparisons/fineweb_completion64_repeated_chunk")
    return [yaml.safe_load((base / f"{kind}.yaml").read_text()) for kind in ("full", "smoke")]


def test_profiles_and_separate_acceptance_allocation():
    full, smoke = configurations()
    for c in (full, smoke):
        completion.validate_config(c)
        assert completion.allocation_id(c) == completion.REPEATED_ALLOCATION
        assert completion.benchmark_filename(c) == "benchmark_completion_repeated_chunk.json"
    assert protocol_fingerprint(full) == protocol_fingerprint(smoke)
    historical = copy.deepcopy(full)
    historical.pop("training_chunk_policy")
    historical.pop("fixed_chunk")
    assert completion.allocation_id(historical) != completion.allocation_id(full)
    assert protocol_fingerprint(historical) != protocol_fingerprint(full)
    from model_collapse.workflow.selection import training_contract
    assert training_contract(completion.arm_config(full, "real")) != training_contract(completion.arm_config(historical, "real"))
    certificate = {"source_code_sha256": "runtime", "workflow": "completion_comparison",
                   "protocol_fingerprint": protocol_fingerprint(historical), "checks": {}}
    with pytest.raises(RuntimeError, match="protocol"):
        require_acceptance(full, certificate, "runtime")


@pytest.mark.parametrize("cycle", range(1, 11))
def test_source_selection_reuses_first_chunk_with_historical_default(cycle):
    assert completion.source_collection({}, cycle) == f"cycle_{cycle:02d}"
    assert completion.source_collection({"training_chunk_policy": "repeat_fixed", "fixed_chunk": 1}, cycle) == "cycle_01"


@pytest.mark.parametrize("changes", [
    {"training_chunk_policy": "typo"}, {"fixed_chunk": 2}, {"fixed_chunk": True},
    {"fixed_chunk": None}, {"training_chunk_policy": "fresh_per_cycle"},
])
def test_invalid_reuse_policy_is_rejected(changes):
    c = {**configurations()[0], **changes}
    with pytest.raises(ValueError):
        completion.validate_config(c)


def test_input_hashes_repeated_but_generator_and_epoch_remain_separate(tmp_path, monkeypatch):
    c = configurations()[1]
    c.update(bos_token_id=2, data_kind="real")
    bank = tmp_path / "bank"
    for cycle in (1, 2):
        atomic_json(bank / f"cycle_{cycle:02d}/manifest.json", {"cycle": cycle})
    monkeypatch.setattr(completion, "bank_path", lambda *_: bank)
    rng = np.random.default_rng(42)
    prompt, target = (rng.integers(3, 40, (4, 64), dtype=np.int32) for _ in range(2))
    root = tmp_path / "arm"
    rows = []
    for cycle in (1, 2):
        arrays, lineage = completion.training_input(root, cycle, prompt, target, c, None, str(cycle), None)
        rows.append(read_json(root / f"seed_42/gen_{cycle:02d}/training_input/manifest.json"))
        assert np.array_equal(arrays[0][:, 1:65], prompt)
        assert np.array_equal(arrays[0][:, 65:], target)
        assert not arrays[2][:, :65].any() and arrays[2][:, 65:].sum() == 4 * 64
        assert lineage["source_collection"] == "cycle_01"
        assert lineage["training_epoch"] == cycle
        assert lineage["chunk_manifest_sha256"] == sha256(bank / "cycle_01/manifest.json")
    assert rows[0]["prompt_content_sha256"] == rows[1]["prompt_content_sha256"]
    assert rows[0]["target_content_sha256"] == rows[1]["target_content_sha256"]
    changed = prompt.copy()
    changed[0, 0] += 1
    with pytest.raises(RuntimeError, match="changed"):
        completion.training_input(root, 2, changed, target, c, None, "2", None, create=False)
    generated_by = []
    def generate(root, cycle, prompts, config, checkpoint, weights, *args, **kwargs):
        generated_by.append((cycle, checkpoint, weights, prompts.copy()))
        return np.full_like(prompts, cycle + 3), "generation" + str(cycle)
    monkeypatch.setattr(completion, "_targets", generate)
    c["data_kind"] = "synthetic"
    synth = tmp_path / "synthetic"
    for cycle in (1, 2):
        _, lineage = completion.training_input(synth, cycle, prompt, target, c, f"model{cycle-1}", f"weights{cycle-1}", None)
        assert lineage["generator_weights_sha256"] == f"weights{cycle-1}"
    assert all(np.array_equal(item[3], prompt) for item in generated_by)
    assert [item[2] for item in generated_by] == ["weights0", "weights1"]
