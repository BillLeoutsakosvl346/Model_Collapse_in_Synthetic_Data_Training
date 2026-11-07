import pytest


@pytest.mark.parametrize('fail', (False, True))
def test_startup_gate_protects_deployment_and_releases_for_independent_training(tmp_path, monkeypatch, fail):
    from filelock import FileLock, Timeout
    from model_collapse import common, remote
    from scripts.operations.launch_concurrent_high_fraction import staged_launch
    c = common.config('comparisons/fineweb_completion64_mixed90/full.yaml')
    monkeypatch.setattr(common, 'STATE', tmp_path)
    monkeypatch.setattr(common, 'config', lambda _: c)
    def observe(run, **kwargs):
        with FileLock(str(tmp_path / 'high-fraction-provision.lock'), timeout=0):
            return 'verified'
    monkeypatch.setattr(remote, 'verify_detached', observe)
    def core(**kwargs):
        with pytest.raises(Timeout):
            FileLock(str(tmp_path / 'high-fraction-provision.lock'), timeout=0).acquire()
        if fail:
            raise RuntimeError('deployment failed')
        return remote.verify_detached('run')
    monkeypatch.setattr(remote, '_launch', core)
    if fail:
        with pytest.raises(RuntimeError, match='deployment failed'):
            staged_launch(90, config_path='fixture', generations=10)
    else:
        assert staged_launch(90, config_path='fixture', generations=10) == 'verified'
    assert remote.verify_detached is observe
    with FileLock(str(tmp_path / 'high-fraction-provision.lock'), timeout=0):
        pass
