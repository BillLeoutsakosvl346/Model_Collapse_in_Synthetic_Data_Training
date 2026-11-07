import copy

import numpy as np
import pytest

from model_collapse.common import config
from model_collapse.completion_mixing import select_rows
from model_collapse.cloud.parallel import check_active, reserved_sibling_allowance
from model_collapse.workflow.completion import allocation_id
from model_collapse.workflow.profiles import protocol_fingerprint


def setup_scope(monkeypatch):
    from model_collapse import remote
    c = config('comparisons/fineweb_completion64_mixed85/full.yaml')
    allocations = {}
    for pct in (85, 90, 95):
        other = config(f'comparisons/fineweb_completion64_mixed{pct}/full.yaml')
        allocations[allocation_id(other)] = {'protocol': protocol_fingerprint(other), 'budget_usd': 3.25}
    authorization = {'max_active_pods': 3, 'backup_store': c['backup_store'],
        'allocations': allocations, 'runtime_sha256': 'runtime', 'backup_storage_reserve_usd': 1.30}
    owner = {'run_id': 'other', 'launch_id': 'launch', 'pod_id': 'pod', 'volume_id': 'volume',
        'kind': 'full', 'workflow': 'completion_comparison', 'code_sha256': 'runtime',
        'parallel_campaign': c['parallel_campaign'], 'allocation': 'fineweb_completion64_mixtures_mixed90'}
    state = {'launches': [owner], 'parallel_campaigns': {c['parallel_campaign']: authorization}}
    monkeypatch.setattr(remote, 'local_resources', lambda name: owner)
    pod = {'id': 'pod', 'name': 'iib-other', 'status': 'RUNNING', 'mounts': {'network': [{'volumeId': 'volume'}]}}
    return c, state, pod


@pytest.mark.parametrize('cycle', range(1, 11))
def test_new_fractions_have_exact_nested_selections(cycle):
    previous = np.array([], dtype=np.int64)
    for pct in (20, 50, 80, 85, 90, 95, 99, 100):
        selection, real = select_rows(10000, pct / 100, 42, cycle)
        indices = selection['synthetic_indices']
        assert len(indices) == 100 * pct and real == 10000 - 100 * pct
        assert np.array_equal(indices[:len(previous)], previous)
        previous = indices
    for pct, count in ((85, 272), (90, 288), (95, 304), (99, 317)):
        selection, real = select_rows(320, pct / 100, 42, cycle)
        assert len(selection['synthetic_indices']) == count and real == 320 - count


def test_parallel_admission_checks_all_owners_and_keeps_duplicate_guard(monkeypatch):
    c, state, pod = setup_scope(monkeypatch)
    check_active(c, [], state)
    check_active(c, [pod], state)
    for bad in ({**pod, 'id': 'unknown'}, {**pod, 'name': 'unknown'}, {**pod, 'mounts': {}}):
        with pytest.raises(RuntimeError):
            check_active(c, [bad], state)
    with pytest.raises(RuntimeError, match='occupied'):
        check_active(c, [pod] * 3, state)
    with pytest.raises(RuntimeError, match='protocol'):
        check_active({**c, 'learning_rate': 1e-4}, [], state)
    changed = copy.deepcopy(state)
    changed['launches'][0]['parallel_campaign'] = 'unrelated'
    with pytest.raises(RuntimeError):
        check_active(c, [pod], changed)


def test_remaining_sibling_funds_are_reserved_before_first_launch(monkeypatch):
    c, state, _ = setup_scope(monkeypatch)
    state['launches'] = []
    assert reserved_sibling_allowance(c, state, lambda _: 0) == pytest.approx(7.80)
    state['launches'] = [{'allocation': 'fineweb_completion64_mixtures_mixed90', 'volume_id': 'v'}]
    assert reserved_sibling_allowance(c, state, lambda d: .5 if d['launches'] else 0) == pytest.approx(7.30)


@pytest.mark.parametrize('pct', (85, 90, 95))
def test_full_smoke_protocol_and_matching_bank(pct):
    full = config(f'comparisons/fineweb_completion64_mixed{pct}/full.yaml')
    smoke = config(f'comparisons/fineweb_completion64_mixed{pct}/smoke.yaml')
    assert protocol_fingerprint(full) == protocol_fingerprint(smoke)
    old = config('comparisons/fineweb_completion64_mixed80/full.yaml')
    assert full['prepared_bank'] == old['prepared_bank']
    assert full['required_gpu'] == 'NVIDIA RTX PRO 4500 Blackwell'
    assert full['required_image'] == old['required_image']
    for key in ('learning_rate', 'microbatch', 'effective_batch', 'num_beams', 'optimizer_policy'):
        assert full[key] == old[key]


def test_99_profile_preserves_science_without_inheriting_parallel_authorization():
    full = config('comparisons/fineweb_completion64_mixed99/full.yaml')
    smoke = config('comparisons/fineweb_completion64_mixed99/smoke.yaml')
    old = config('comparisons/fineweb_completion64_mixed95/full.yaml')
    assert protocol_fingerprint(full) == protocol_fingerprint(smoke)
    assert full['completion_arms'] == [{'id': 'mixed99', 'synthetic_fraction': .99}]
    assert 'parallel_campaign' not in full and 'parallel_campaign' not in smoke
    excluded = {'experiment_id', 'completion_arms', 'parallel_campaign', 'backup_store'}
    assert {k:v for k,v in full.items() if k not in excluded} == {k:v for k,v in old.items() if k not in excluded}
    assert full['backup_store']['backend'] == old['backup_store']['backend'] == 'runpod_s3'
    assert full['backup_store']['region'] == old['backup_store']['region']
    from model_collapse.workflow.profiles import require_acceptance
    certificate = {'source_code_sha256': 'runtime', 'protocol_fingerprint': protocol_fingerprint(old)}
    with pytest.raises(RuntimeError, match='scientific protocol'):
        require_acceptance(full, certificate, 'runtime')
