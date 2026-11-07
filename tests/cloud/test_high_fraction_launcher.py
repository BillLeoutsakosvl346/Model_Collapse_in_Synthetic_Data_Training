import pytest


@pytest.mark.parametrize('pct', (85, 90, 95, 99))
def test_host_admission_overrides_are_restored_and_keep_catalog_truth(monkeypatch, pct):
    from model_collapse import remote
    from scripts.operations.launch_high_fraction_completions import launch_one
    monkeypatch.setattr(remote, 'ledger', lambda: {'launches': [], 'setup_budget_usd': 3.5})
    def preflight(**kwargs):
        return {'gpu_candidates': [{'data_center': 'EU-CZ-1', 'hourly_usd': .72},
                                   {'data_center': 'EU-RO-1', 'hourly_usd': .72}]}
    def detached(run, **kwargs):
        assert kwargs['max_wait_seconds'] >= 1200
        return 'observed'
    monkeypatch.setattr(remote, 'preflight', preflight)
    monkeypatch.setattr(remote, 'verify_detached', detached)
    original_budget = remote.SETUP_BUDGET
    def launch(**kwargs):
        assert remote.SETUP_BUDGET == 3.5
        assert remote.preflight()['gpu_candidates'][0]['data_center'] == 'EU-RO-1'
        assert remote.verify_detached('run', max_wait_seconds=600) == 'observed'
        assert kwargs['config_path'].as_posix().endswith(f'mixed{pct}/full.yaml')
        raise RuntimeError('test failure')
    monkeypatch.setattr(remote, 'launch', launch)
    with pytest.raises(RuntimeError, match='test failure'):
        launch_one(pct)
    assert remote.preflight is preflight
    assert remote.verify_detached is detached
    assert remote.SETUP_BUDGET == original_budget


def test_existing_condition_requires_matching_explicit_resume(monkeypatch):
    from model_collapse import remote
    from scripts.operations.launch_high_fraction_completions import launch_one
    monkeypatch.setattr(remote, 'ledger', lambda: {'launches': [{'allocation': 'fineweb_completion64_mixtures_mixed85', 'kind': 'smoke', 'run_id': 'existing'}]})
    with pytest.raises(RuntimeError, match='already has a launch'):
        launch_one(85, smoke=True)
    with pytest.raises(RuntimeError, match='recorded allocation'):
        launch_one(85, smoke=True, resume_id='wrong')


@pytest.mark.parametrize('valid_owner', (True, False))
def test_current_watchdog_receipt_preserves_old_session(tmp_path, valid_owner):
    from model_collapse.common import atomic_json, read_json, sha256
    from scripts.operations.validate_high_fraction_completions import promote_current_watchdog
    old = tmp_path / 'watchdog_exit.json'
    atomic_json(old, {'pod_id': 'old', 'time': 10, 'reason': 'budget_exhausted'})
    old_hash = sha256(old)
    payload = tmp_path / 'payload.json'
    atomic_json(payload, {'pod_id': 'new' if valid_owner else 'other', 'time': 30,
                          'reason': 'validation_budget_guard'})
    checksum = sha256(payload)
    relative = 'fetched_control/' + checksum + '/watchdog_exit.json'
    destination = tmp_path / relative
    destination.parent.mkdir(parents=True)
    destination.write_bytes(payload.read_bytes())
    atomic_json(tmp_path / 'recovery_snapshot.json', {'revisions': {'dataset': 'pinned'},
        'preserved_control_files': {'watchdog_exit.json': {'recovered_path': relative, 'sha256': checksum}}})
    owner = {'run_id': 'run', 'launch_id': 'launch', 'pod_id': 'new', 'created_at': 20}
    if not valid_owner:
        with pytest.raises(RuntimeError, match='not this validation'):
            promote_current_watchdog(tmp_path, owner)
        assert sha256(old) == old_hash
    else:
        promote_current_watchdog(tmp_path, owner)
        assert sha256(old) == checksum
        history = read_json(tmp_path / 'watchdog_control_promotion.json')['old_receipt_path']
        assert sha256(tmp_path / history) == old_hash
