from types import SimpleNamespace

import pytest
from huggingface_hub.errors import BadRequestError
from requests import Response

from model_collapse import hub, supervisor
from model_collapse.common import atomic_json, read_json


MISSING_LFS = "Your push was rejected because an LFS pointer pointed to a file that does not exist. Offending file: model.safetensors"


def hub_error(status=400, message=MISSING_LFS):
    response = Response()
    response.status_code = status
    return BadRequestError(message, response=response)


@pytest.mark.parametrize("opt_in,expected", [(False, 1), (True, 2)])
def test_only_explicit_backup_missing_lfs_retries(opt_in, expected, monkeypatch):
    calls, verifies = [], []
    monkeypatch.setattr(hub.time, "sleep", lambda _: None)
    def upload_folder():
        calls.append(1)
        if len(calls) == 1: raise hub_error()
        return 'commit'
    if opt_in:
        assert hub.retry_transfer(upload_folder, retry_missing_lfs_commit=True, before_retry=lambda: verifies.append(1)) == 'commit'
        assert verifies == [1]
    else:
        with pytest.raises(BadRequestError): hub.retry_transfer(upload_folder)
    assert len(calls) == expected


@pytest.mark.parametrize("status,message", [(400, 'invalid path'), (401, MISSING_LFS), (403, MISSING_LFS), (404, MISSING_LFS)])
def test_other_client_errors_remain_terminal(status, message, monkeypatch):
    calls = []
    monkeypatch.setattr(hub.time, "sleep", lambda _: None)
    def upload_folder():
        calls.append(1)
        raise hub_error(status, message)
    with pytest.raises(BadRequestError): hub.retry_transfer(upload_folder, retry_missing_lfs_commit=True)
    assert len(calls) == 1


def test_persistent_missing_lfs_retry_is_bounded_and_rechecks_every_attempt(monkeypatch):
    calls, verifies = [], []
    monkeypatch.setattr(hub.time, "sleep", lambda _: None)
    def upload_folder():
        calls.append(1)
        raise hub_error()
    with pytest.raises(BadRequestError):
        hub.retry_transfer(upload_folder, retry_missing_lfs_commit=True, before_retry=lambda: verifies.append(1))
    assert len(calls) == 4 and len(verifies) == 3


@pytest.mark.parametrize("corrupt", [None, 'checkpoint', 'index'])
def test_backup_retries_identical_allowlist_or_rejects_mutation_without_snapshot(tmp_path, monkeypatch, corrupt):
    root = tmp_path / 'run'
    weights = root / 'seed_42/gen_02/checkpoint/model.safetensors'
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b'immutable checkpoint bytes')
    repos = {'model': 'test/models', 'dataset': 'test/data'}
    atomic_json(root / 'resources.json', {'repos': repos})
    calls = []
    class API:
        def __init__(self, **kwargs): pass
        def upload_folder(self, **kwargs):
            calls.append((kwargs['repo_type'], list(kwargs['allow_patterns'])))
            if len(calls) == 1:
                if corrupt == 'checkpoint': weights.write_bytes(b'changed checkpoint')
                if corrupt == 'index': atomic_json(root / 'artifact_index.json', {'changed': True})
                raise hub_error()
    monkeypatch.setattr(hub, 'HfApi', API)
    monkeypatch.setattr(hub, 'get_token', lambda: None)
    monkeypatch.setattr(hub.time, 'sleep', lambda _: None)
    monkeypatch.setattr(hub, 'resolve_snapshot', lambda _: {'model': 'a' * 40, 'dataset': 'b' * 40})
    if corrupt:
        with pytest.raises(RuntimeError, match='corrupt|index changed'):
            hub.backup(root)
        assert len(calls) == 1
        assert not (root / 'backup_snapshot.json').exists()
    else:
        hub.backup(root)
        assert len(calls) == 3 and calls[0] == calls[1]
        assert read_json(root / 'backup_snapshot.json')['revisions']['model'] == 'a' * 40


@pytest.mark.parametrize("requested,now,expected", [('failure', 100., 'failure'), ('complete', 100., 'complete'),
    (None, 100., 'supervisor_died'), ('failure', 4000., 'budget_exhausted')])
def test_watchdog_preserves_explicit_request_when_supervisor_also_exits(tmp_path, monkeypatch, requested, now, expected):
    atomic_json(tmp_path / 'resources.json', {'pod_id': 'pod', 'created_at': 0., 'session_allowance_usd': 1., 'guard_hourly_usd': 1.})
    atomic_json(tmp_path / 'supervisor.json', {'pid': 123})
    if requested: atomic_json(tmp_path / 'stop_request.json', {'reason': requested})
    stopped = []
    monkeypatch.setattr(supervisor, 'Runpod', lambda: SimpleNamespace(stop=lambda pod: stopped.append(pod)))
    monkeypatch.setattr(supervisor, 'now', lambda: now)
    monkeypatch.setattr(supervisor.os, 'kill', lambda *a: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr(supervisor, '_publish_shutdown_intent', lambda *a: None)
    supervisor.watchdog(tmp_path)
    assert stopped == ['pod']
    assert read_json(tmp_path / 'watchdog_exit.json')['reason'] == expected
