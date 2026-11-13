import hashlib
import json
import tarfile
from pathlib import Path

import pytest
import yaml

from model_collapse.common import atomic_json, read_json, sha256
from model_collapse.cloud import sgd_setup
from model_collapse.workflow import sgd_profile
from model_collapse.workflow import sgd_runtime


def _benchmark(c, runtime="a" * 64):
    return {"benchmark_contract_sha256": sgd_runtime.benchmark_contract(c),
        "calibration_receipt_sha256": c["calibration_reference"]["receipt_sha256"],
        "source_runtime_sha256": runtime, "benchmark_updates_discarded": 0,
        "microbatch": 32, "effective_batch": 128, "generation_batch": 32, "gpu": c["required_gpu"],
        "training_target_tokens_per_second": 10000., "generation_new_tokens_per_second": 2000.,
        "primary_target_tokens_per_second": 50000., "secondary_target_tokens_per_second": 50000.,
        "full_primary_targets": {"validation": 476608, "test": 467392}, "full_secondary_targets": 1000,
        "training_includes_update_observer": True, "step_seconds_by_arm": {"real": .1, "mixed99": .3, "synthetic": .1},
        "calibration_physical_updates": 192}


def _source(tmp_path):
    from model_collapse.workflow.sgd_calibration import calibration_contract
    root = tmp_path / "accepted";root.mkdir()
    c = yaml.safe_load(Path("comparisons/smollm2_sgd_greedy_effects/full.yaml").read_text())
    c.update(calibration_required=True, calibration_reference=None)
    receipt = {"complete": True, "selection_policy": "bounded_real_only_192_updates_v1",
               "selected_learning_rate": .001, "contract_sha256": calibration_contract(c)}
    atomic_json(root / "calibration/selection.json", receipt)
    c = sgd_profile.apply_calibration(c, receipt, sha256(root / "calibration/selection.json"))
    b = _benchmark(c);atomic_json(root / "benchmark.json", b)
    atomic_json(root / "artifact_index.json", {"files": {
        name: {"sha256": sha256(root / name), "bytes": (root / name).stat().st_size}
        for name in ("benchmark.json", "calibration/selection.json")}})
    snapshot = {"run_id": "accepted-smoke", "repos": {"model": "a/m", "dataset": "a/d"},
        "revisions": {"model": "b" * 40, "dataset": "c" * 40},
        "artifact_index_sha256": sha256(root / "artifact_index.json")}
    c["calibration_reference"].update(snapshot=snapshot, relative_path="calibration/selection.json")
    c["benchmark_reference"] = {"snapshot": snapshot, "relative_path": "benchmark.json",
        "sha256": sha256(root / "benchmark.json"), "source_runtime_sha256": "a" * 64}
    return root, c, b


def _fake_snapshot(monkeypatch, source_root, expected):
    from model_collapse.artifacts import gradient_sources
    class Snapshot:
        def __init__(self, ref, root): assert ref == expected
        def get(self, relative): return source_root / relative
    monkeypatch.setattr(gradient_sources, "Snapshot", Snapshot)


def test_verified_bank_cache_is_allowlisted_and_rejects_corruption(tmp_path, monkeypatch):
    bank=tmp_path/'bank';bank.mkdir();state=tmp_path/'state';state.mkdir()
    atomic_json(bank/'manifest.json',{'bank':True});(bank/'tokens.npy').write_bytes(b'array')
    atomic_json(bank/'artifact_index.json',{'files':{n:{'sha256':sha256(bank/n),'bytes':(bank/n).stat().st_size} for n in ['manifest.json','tokens.npy']}})
    (bank/'private.env').write_text('excluded')
    c={'bank_local_path':str(bank),'prepared_bank':{'manifest_sha256':sha256(bank/'manifest.json'),'artifact_index_sha256':sha256(bank/'artifact_index.json')}}
    monkeypatch.setattr(sgd_setup,'STATE',state)
    archive=sgd_setup.bank_cache(c)
    with tarfile.open(archive) as tar:assert set(tar.getnames())=={'manifest.json','tokens.npy','artifact_index.json'}
    (bank/'tokens.npy').write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='corrupt'):sgd_setup.bank_cache(c)


def test_pending_full_rejected_before_any_remote_source_read():
    c=yaml.safe_load(Path('comparisons/smollm2_sgd_greedy_effects/full.yaml').read_text())
    c['calibration_required']=True
    with pytest.raises(RuntimeError,match='Freeze'):sgd_setup.verify_calibration_source(c)


def test_full_config_publication_requires_all_acceptance_checks(tmp_path):
    atomic_json(tmp_path/'resources.json',{'code_sha256':'a'*64})
    atomic_json(tmp_path/'validation_results.json',{'checks':{'incomplete':True},'source_code_sha256':'a'*64})
    with pytest.raises(RuntimeError,match='Accept the recovered'):
        sgd_setup.write_full_configuration(tmp_path,tmp_path/'full.yaml')


def test_resolved_handoff_pins_rate_and_snapshot_without_provisioning(tmp_path):
    from model_collapse.workflow.sgd_calibration import calibration_contract
    c=yaml.safe_load(Path('comparisons/smollm2_sgd_greedy_effects/full.yaml').read_text())
    c.update(calibration_required=True,calibration_reference=None)
    target=tmp_path/'full.yaml';target.write_text(yaml.safe_dump(c))
    receipt={'complete':True,'selection_policy':'bounded_real_only_192_updates_v1','selected_learning_rate':.001,'contract_sha256':calibration_contract(c)}
    atomic_json(tmp_path/'calibration/selection.json',receipt)
    atomic_json(tmp_path/'resources.json',{'code_sha256':'a'*64})
    atomic_json(tmp_path/'validation_results.json',{'checks':{k:True for k in sgd_profile.CHECKS},'source_code_sha256':'a'*64})
    resolved = sgd_profile.apply_calibration(c, receipt, sha256(tmp_path/'calibration/selection.json'))
    atomic_json(tmp_path/'benchmark.json', _benchmark(resolved))
    atomic_json(tmp_path/'artifact_index.json', {'files': {'benchmark.json': {'sha256': sha256(tmp_path/'benchmark.json')}}})
    atomic_json(tmp_path/'recovery_snapshot.json',{'run_id':'smoke-test','repos':{'model':'a/m','dataset':'a/d'},'revisions':{'model':'b'*40,'dataset':'c'*40},'artifact_index_sha256':sha256(tmp_path/'artifact_index.json')})
    result=sgd_setup.write_full_configuration(tmp_path,target)
    actual=yaml.safe_load(target.read_text())
    assert actual['learning_rate']==actual['min_learning_rate']==.001
    assert actual['calibration_reference']['snapshot']['run_id']=='smoke-test'
    assert actual['benchmark_reference']['snapshot']==actual['calibration_reference']['snapshot']
    assert actual['benchmark_reference']['sha256']==sha256(tmp_path/'benchmark.json')
    assert not result['full_experiment_launched']
    from model_collapse.workflow.profiles import protocol_fingerprint
    smoke = yaml.safe_load(Path('comparisons/smollm2_sgd_greedy_effects/smoke.yaml').read_text())
    smoke = sgd_profile.apply_calibration(smoke, receipt, sha256(tmp_path/'calibration/selection.json'))
    assert protocol_fingerprint(smoke) == protocol_fingerprint(actual)


def test_fresh_full_root_restores_pinned_benchmark_without_calibration_or_warmup(tmp_path, monkeypatch):
    source, c, b = _source(tmp_path)
    _fake_snapshot(monkeypatch, source, c["benchmark_reference"]["snapshot"])
    from model_collapse.workflow import sgd_calibration
    from model_collapse.science import engine
    def forbidden(*args, **kwargs): raise AssertionError("Full benchmark attempted new model/calibration work")
    monkeypatch.setattr(sgd_calibration, "calibrate", forbidden)
    monkeypatch.setattr(engine, "load_model", forbidden)
    fresh = tmp_path / "full";fresh.mkdir();atomic_json(fresh/'resources.json', {"code_sha256": "a" * 64})
    assert not (fresh/'calibration/.work').exists()
    sgd_runtime.prepare(fresh, c, {"code_sha256": "a" * 64}, tmp_path/'unused-bank')
    assert sha256(fresh/'benchmark.json') == c['benchmark_reference']['sha256']
    assert sgd_runtime.benchmark(fresh, c, tmp_path/'unused-bank') == b
    assert sgd_setup.verify_calibration_source(c) == b


@pytest.mark.parametrize("change", ["source_bytes", "local_bytes", "runtime", "protocol", "missing_reference"])
def test_full_benchmark_rejects_unverified_or_incompatible_timings(tmp_path, monkeypatch, change):
    source, c, _ = _source(tmp_path)
    _fake_snapshot(monkeypatch, source, c["benchmark_reference"]["snapshot"])
    fresh = tmp_path/'full';fresh.mkdir()
    if change == "source_bytes": atomic_json(source/'benchmark.json', {"changed": True})
    if change == "local_bytes": atomic_json(fresh/'benchmark.json', {"changed": True})
    if change == "protocol": c['learning_rate'] = .01
    if change == "missing_reference": c.pop('benchmark_reference')
    with pytest.raises(RuntimeError, match="benchmark"):
        sgd_setup.restore_benchmark(c, fresh, expected_runtime="z" * 64 if change == "runtime" else "a" * 64)


def test_benchmark_provenance_does_not_change_accepted_scientific_fingerprint(tmp_path):
    from model_collapse.workflow.profiles import protocol_fingerprint
    _, c, _ = _source(tmp_path)
    original = protocol_fingerprint(c)
    c.pop('benchmark_reference')
    assert protocol_fingerprint(c) == original
