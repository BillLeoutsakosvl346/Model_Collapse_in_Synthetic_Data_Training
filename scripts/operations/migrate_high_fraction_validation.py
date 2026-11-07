"""Cloud-side recovery of a stopped acceptance workspace into live GPU placement."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath


RUN = '20261107T141453Z-smoke-completion64-mixed90'


def stage(volume_id):
    from filelock import FileLock
    from model_collapse import remote
    from model_collapse.common import PROJECT, STATE, atomic_json, read_json, sha256, now
    from model_collapse.cloud.s3_artifacts import client, validate_reference
    from model_collapse.hub import download_snapshot_file
    from huggingface_hub import hf_hub_download, get_token
    root = PROJECT / 'runs' / RUN
    owner = remote.local_resources(RUN)
    api = remote.Runpod()
    if api.pod(owner['pod_id'])['status'] != 'EXITED':
        raise RuntimeError('Migration requires the exact stopped source owner')
    source_volume = api.request('GET', '/network-volumes/' + owner['volume_id'])
    volume = api.request('GET', '/network-volumes/' + volume_id)
    if (source_volume['dataCenter'] != owner['data_center'] or volume['dataCenter'] != 'EU-RO-1'
            or volume['size'] != 60 or volume['name'] != 'iib-mixed90-validation-recovery-20261107'):
        raise RuntimeError('Migration source/destination placement association differs')
    created = now()
    with FileLock(str(STATE / 'ledger.lock'), timeout=60):
        d = remote.ledger()
        d.setdefault('volumes', {}).setdefault(volume_id, {'run_id': RUN, 'volume_id': volume_id,
            'created_at': created, 'charge_started_at': created, 'hourly_usd': 60 * .07 / 730,
            'purpose': 'Verified acceptance placement recovery; original volume preserved until full recovery passes.'})
        atomic_json(STATE / 'ledger.json', d)
    parent_ref = read_json(root / 'recovery_snapshot.json')
    child_ref = read_json(root / 'child_snapshots.json')['mixed90']
    store = read_json(root / 'config.json')['backup_store']
    connection = client(store)
    records = []

    def transfer(job):
        run_id, relative, meta, revisions = job
        if PurePosixPath(relative).is_absolute() or '..' in PurePosixPath(relative).parts:
            raise ValueError('Unsafe indexed recovery path')
        target = 'runs/' + run_id + '/' + relative
        if 'storage' in meta:
            ref = meta['storage']
            validate_reference(ref, meta['sha256'])
            if ref['bucket'] != store['bucket'] or ref['region'] != store['region']:
                raise RuntimeError('Bulk recovery source is outside the pinned campaign store')
            connection.copy_object(Bucket=volume_id, Key=target,
                CopySource={'Bucket': ref['bucket'], 'Key': ref['key']})
        else:
            path = download_snapshot_file(run_id, owner['repos'], relative, meta, revisions)
            connection.upload_file(str(path), volume_id, target)
        if connection.head_object(Bucket=volume_id, Key=target)['ContentLength'] != meta['bytes']:
            raise RuntimeError('Staged recovery object size differs')
        return {'path': target, 'sha256': meta['sha256'], 'bytes': meta['bytes']}

    jobs, controls = [], []
    for run_id, ref in ((RUN, parent_ref), (child_ref['run_id'], child_ref)):
        folder = PROJECT / 'runs' / run_id
        index_path = folder / 'artifact_index.json'
        if sha256(index_path) != ref['artifact_index_sha256']:
            raise RuntimeError('Pinned recovery index changed')
        index = read_json(index_path)
        jobs.extend((run_id, rel, meta, ref['revisions']) for rel, meta in index['files'].items())
        connection.upload_file(str(index_path), volume_id, 'runs/' + run_id + '/artifact_index.json')
        snapshot = {k: ref[k] for k in ('run_id', 'repos', 'revisions', 'artifact_index_sha256')}
        connection.put_object(Bucket=volume_id, Key='runs/' + run_id + '/backup_snapshot.json',
                              Body=(json.dumps(snapshot, indent=2) + '\n').encode())
        # Status/events are deliberately outside immutable indexes. Restore
        # their original pinned bytes too: the coordinator needs a completed
        # child's status to advance, and acceptance audits its actual events.
        for name in ('status.json', 'events.jsonl'):
            path = hf_hub_download(owner['repos']['dataset'], run_id + '/' + name,
                repo_type='dataset', revision=ref['revisions']['dataset'], token=get_token())
            connection.upload_file(path, volume_id, 'runs/' + run_id + '/' + name)
            controls.append({'run_id': run_id, 'path': name, 'sha256': sha256(path),
                             'source_revision': ref['revisions']['dataset']})
            if name == 'events.jsonl' and run_id == RUN:
                from pathlib import Path
                if 'completion_handoff_interruption' in Path(path).read_text():
                    connection.put_object(Bucket=volume_id, Key='runs/' + RUN + '/.handoff_injected', Body=b'')
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(transfer, jobs))
    receipt = {'run_id': RUN, 'source_owner': owner, 'source_volume': source_volume,
        'destination_volume': volume, 'parent_snapshot': parent_ref, 'child_snapshot': child_ref,
        'operational_state_recovery': controls,
        'files': records, 'staged_bytes': sum(x['bytes'] for x in records), 'time': now(),
        'scope': 'Bulk server-side copies from SHA-addressed private backup; indexed metadata bytes verified locally. Complete byte hashes are verified on the destination GPU before acceptance.'}
    atomic_json(STATE / 'mixed90-placement-recovery.json', receipt)
    atomic_json(PROJECT / 'docs/validation/2026-11-07_completion-mixed90-placement-recovery.json', receipt)
    connection.put_object(Bucket=volume_id, Key='runs/' + RUN + '/placement_recovery.json',
                          Body=(json.dumps(receipt, indent=2) + '\n').encode())
    print(json.dumps({'staged_files': len(records), 'staged_bytes': receipt['staged_bytes'], 'destination_volume': volume_id}), flush=True)
    return receipt


def resume():
    from model_collapse import remote
    from model_collapse.common import STATE, read_json
    from scripts.operations.resume_high_fraction_validation import recovery_progress
    from scripts.operations.validate_high_fraction_completions import validate_one
    receipt = read_json(STATE / 'mixed90-placement-recovery.json')
    original_lookup, original_observer = remote.local_resources, remote.verify_detached
    source = receipt['source_owner']
    destination = receipt['destination_volume']
    current = original_lookup(RUN)
    if not remote.same_owner(current, source):
        raise RuntimeError('Authoritative owner changed after migration staging')
    staged = {**source, 'volume_id': destination['id'], 'data_center': destination['dataCenter'],
        'volume_created_at': receipt['time'], 'staged_recovery_source_owner': source}
    def lookup(run_id):
        actual = original_lookup(run_id)
        # This is a bootstrap placement descriptor only. The registry/receipts
        # retain the original tuple until the provider creates and registers a
        # real replacement Pod associated with this verified destination.
        if run_id == RUN and remote.same_owner(actual, source):
            return staged
        return actual
    remote.local_resources = lookup
    remote.verify_detached = recovery_progress
    try:
        return validate_one(90, RUN)
    finally:
        remote.local_resources, remote.verify_detached = original_lookup, original_observer


def cleanup_source():
    from model_collapse import remote
    from model_collapse.common import PROJECT, STATE, atomic_json, read_json, now
    migration = read_json(STATE / 'mixed90-placement-recovery.json')
    setup = read_json(PROJECT / 'docs/validation/2026-11-07_completion-mixed90-setup.json', {})
    restoration = setup.get('remote_validation', {}).get('restorations', {}).get('mixed90', {})
    if (not setup.get('checks') or not all(setup['checks'].values()) or
            restoration.get('scope') != 'full' or
            restoration.get('artifact_index_sha256') != migration['child_snapshot']['artifact_index_sha256']):
        raise RuntimeError('Original placement requires complete restoration of its exact pinned source')
    source = migration['source_owner']
    entry = next(x for x in remote.ledger()['launches'] if x['launch_id'] == source['launch_id'])
    if not remote.same_owner(source, entry):
        raise RuntimeError('Original source ledger association differs')
    api = remote.Runpod()
    pod = api.pod(source['pod_id'])
    mounts = {x['volumeId'] for x in pod.get('mounts', {}).get('network', [])}
    if pod['status'] != 'EXITED' or pod['name'] != 'iib-' + RUN or source['volume_id'] not in mounts:
        raise RuntimeError('Original source is not the exact stopped temporary resource')
    volume = api.request('GET', '/network-volumes/' + source['volume_id'])
    if volume['dataCenter'] != source['data_center'] or volume['name'] != 'iib-' + RUN:
        raise RuntimeError('Original source volume ownership differs')
    api.request('DELETE', '/pods/' + source['pod_id'])
    api.request('DELETE', '/network-volumes/' + source['volume_id'])
    if (any(p['id'] == source['pod_id'] for p in api.request('GET', '/pods')['pods']) or
            any(v['id'] == source['volume_id'] for v in api.request('GET', '/network-volumes')['networkVolumes'])):
        raise RuntimeError('Original temporary resource removal is not verified')
    remote.update_ledger(source, state='removed', volume_deleted_at=now())
    result = {'run_id': RUN, 'original_pod_removed': source['pod_id'],
        'original_volume_removed': source['volume_id'], 'time': now(),
        'recovery_index_sha256': restoration['artifact_index_sha256'], 'verified_bytes': restoration['verified_bytes']}
    atomic_json(PROJECT / 'docs/validation/2026-11-07_completion-mixed90-original-placement-cleanup.json', result)
    print(json.dumps(result), flush=True)
    return result


def main():
    os.environ.setdefault('HF_HUB_CACHE', r'C:\hf')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', metavar='VOLUME_ID')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cleanup-source', action='store_true')
    args = parser.parse_args()
    if args.stage:
        stage(args.stage)
    elif args.resume:
        resume()
    elif args.cleanup_source:
        cleanup_source()
    else:
        parser.error('Choose --stage or --resume')


if __name__ == '__main__':
    main()
