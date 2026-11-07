from pathlib import Path

from model_collapse.common import atomic_json, read_json, sha256


def test_cloud_staging_recovers_pinned_completion_status_and_events(tmp_path, monkeypatch):
    from model_collapse import common, remote
    from model_collapse.cloud import s3_artifacts
    from scripts.operations import migrate_high_fraction_validation as migration
    import huggingface_hub
    root = tmp_path / 'runs' / migration.RUN
    child = root.with_name(root.name + '-mixed90')
    objects, cache = {}, {}
    repos = {'model': 'org/models', 'dataset': 'org/data'}
    refs = []
    for folder in (root, child):
        folder.mkdir(parents=True)
        payload = folder / 'manifest.json'
        atomic_json(payload, {'source': folder.name})
        meta = {'sha256': sha256(payload), 'bytes': payload.stat().st_size, 'repo_type': 'dataset'}
        atomic_json(folder / 'artifact_index.json', {'run_id': folder.name, 'repos': repos,
                                                   'files': {'manifest.json': meta}})
        ref = {'run_id': folder.name, 'repos': repos, 'revisions': {'model': 'a'*40, 'dataset': 'b'*40},
               'artifact_index_sha256': sha256(folder / 'artifact_index.json')}
        refs.append(ref)
        for name, value in (('status.json', {'state': 'complete'}), ('events.jsonl', None)):
            source = folder / ('cached-' + name)
            if value is None:
                source.write_text('{"event":"completion_handoff_interruption"}\n')
            else:
                atomic_json(source, value)
            cache[folder.name + '/' + name] = str(source)
    atomic_json(root / 'recovery_snapshot.json', refs[0])
    atomic_json(root / 'child_snapshots.json', {'mixed90': refs[1]})
    store = {'backend': 'runpod_s3', 'bucket': 'backup', 'region': 'EU-RO-1', 'prefix': 'test'}
    atomic_json(root / 'config.json', {'backup_store': store})
    owner = {'run_id': migration.RUN, 'pod_id': 'oldpod', 'volume_id': 'oldvol',
             'data_center': 'EU-CZ-1', 'repos': repos}
    class Provider:
        def pod(self, pod):
            return {'status': 'EXITED'}
        def request(self, method, route):
            assert method == 'GET'
            return ({'dataCenter': 'EU-CZ-1'} if route.endswith('oldvol') else
                {'id': 'newvolume', 'dataCenter': 'EU-RO-1', 'size': 60,
                 'name': 'iib-mixed90-validation-recovery-20261107'})
    class S3:
        def upload_file(self, filename, bucket, key):
            objects[key] = Path(filename).read_bytes()
        def put_object(self, *, Bucket, Key, Body):
            objects[Key] = Body
        def head_object(self, *, Bucket, Key):
            return {'ContentLength': len(objects[Key])}
    state = tmp_path / 'state'
    state.mkdir()
    monkeypatch.setattr(common, 'PROJECT', tmp_path)
    monkeypatch.setattr(common, 'STATE', state)
    monkeypatch.setattr(remote, 'local_resources', lambda _: owner)
    monkeypatch.setattr(remote, 'Runpod', Provider)
    monkeypatch.setattr(remote, 'ledger', lambda: {'launches': []})
    monkeypatch.setattr(s3_artifacts, 'client', lambda _: S3())
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', lambda repo, path, **kwargs: cache[path])
    import model_collapse.hub as hub
    monkeypatch.setattr(hub, 'download_snapshot_file', lambda run, repos, rel, meta, revisions: tmp_path / 'runs' / run / rel)
    receipt = migration.stage('newvolume')
    import json
    assert json.loads(objects['runs/' + child.name + '/status.json'])['state'] == 'complete'
    assert b'completion_handoff_interruption' in objects['runs/' + child.name + '/events.jsonl']
    assert objects['runs/' + root.name + '/.handoff_injected'] == b''
    assert len(receipt['operational_state_recovery']) == 4
    assert receipt['source_owner'] == owner
