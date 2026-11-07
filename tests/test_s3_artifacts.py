from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from model_collapse.common import sha256
from model_collapse.cloud import s3_artifacts as s3

STORE = {'backend': 'runpod_s3', 'bucket': 'mng23iliig', 'region': 'EU-RO-1', 'prefix': 'test'}


class MemoryS3:
    def __init__(self):
        self.objects = {}

    def head_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({'Error': {'Code': '404'}}, 'HeadObject')
        return {'ContentLength': len(self.objects[Key]), 'Metadata': {}}

    def upload_file(self, filename, bucket, key, **kwargs):
        self.objects[key] = Path(filename).read_bytes()

    def download_file(self, bucket, key, filename):
        Path(filename).write_bytes(self.objects[key])


def test_bulk_roundtrip_and_corruption_detection(tmp_path, monkeypatch):
    monkeypatch.setattr(s3, 'STATE', tmp_path / 'state')
    p = tmp_path / 'weights.pt'
    p.write_bytes(b'checkpoint bytes')
    meta = {'sha256': sha256(p), 'bytes': p.stat().st_size}
    api = MemoryS3()
    ref = s3.upload(p, meta, STORE, connection=api)
    assert s3.upload(p, meta, STORE, connection=api) == ref
    meta['storage'] = ref
    restored = s3.download(meta, connection=api)
    assert restored.read_bytes() == p.read_bytes()
    restored.unlink()
    api.objects[ref['key']] = b'corrupted bytes!'
    with pytest.raises(RuntimeError, match='checksum'):
        s3.download(meta, connection=api)
    assert not restored.exists()


@pytest.mark.parametrize('change', ({'bucket': '../bad'}, {'prefix': '../escape'}, {'region': 'attacker.example'}, {'backend': 'public'}))
def test_references_cannot_redirect_to_arbitrary_destinations(change):
    with pytest.raises(ValueError):
        s3.reference({**STORE, **change}, 'a' * 64)


def test_bulk_reference_is_bound_to_sha_and_historical_metadata_still_works():
    from model_collapse.hub import _check_meta
    meta = {'sha256': 'a' * 64, 'bytes': 10, 'repo_type': 'model'}
    _check_meta(meta)
    meta['storage'] = s3.reference(STORE, meta['sha256'])
    _check_meta(meta)
    meta['storage']['key'] = 'test/objects/' + 'b' * 64
    with pytest.raises(ValueError, match='checksum'):
        _check_meta(meta)


def test_hub_backup_routes_bulk_and_pins_its_index(tmp_path, monkeypatch):
    from model_collapse import hub
    from model_collapse.common import atomic_json, read_json
    atomic_json(tmp_path / 'config.json', {'backup_store': STORE})
    atomic_json(tmp_path / 'status.json', {'state': 'complete'})
    (tmp_path / 'events.jsonl').write_text('')
    checkpoint = tmp_path / 'seed_42/gen_01/checkpoint/weights.safetensors'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'weight tensors')
    api = MemoryS3()
    uploads = []
    class Hub:
        def upload_folder(self, **kwargs):
            uploads.extend(kwargs['allow_patterns'])
    monkeypatch.setattr(hub, 'HfApi', lambda **kw: Hub())
    monkeypatch.setattr(hub, 'resolve_snapshot', lambda _: {'model': 'a'*40, 'dataset': 'b'*40})
    monkeypatch.setattr(s3, 'client', lambda _: api)
    index = hub.backup(tmp_path, {'model': 'org/model', 'dataset': 'org/data'})
    rel = checkpoint.relative_to(tmp_path).as_posix()
    assert rel not in uploads and 'artifact_index.json' in uploads
    meta = index['files'][rel]
    assert meta['storage']['key'].endswith(sha256(checkpoint))
    assert read_json(tmp_path / 'backup_snapshot.json')['artifact_index_sha256'] == sha256(tmp_path / 'artifact_index.json')
    monkeypatch.setattr(s3, 'STATE', tmp_path / 'cold')
    restored = hub.download_snapshot_file(tmp_path.name, index['repos'], rel, meta,
                                        {'model': 'a'*40, 'dataset': 'b'*40})
    assert restored.read_bytes() == checkpoint.read_bytes()
