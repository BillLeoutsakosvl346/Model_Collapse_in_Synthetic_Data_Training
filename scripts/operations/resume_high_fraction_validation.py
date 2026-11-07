"""Recover acceptance evidence after a funded stop, preserving trained stages."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import time


def recovery_progress(run_id, *, max_wait_seconds=None):
    from model_collapse import remote
    from model_collapse.common import PROJECT, atomic_json, digest, now, read_json, sha256
    from model_collapse.workflow.completion import child_roots
    root = PROJECT / 'runs' / run_id
    r = remote.local_resources(run_id)
    c = read_json(root / 'config.json')
    child = next(iter(child_roots(root, c).values())).name
    old_path = root / 'original_training_detached_verification.json'
    if not old_path.exists():
        prior = read_json(root / 'detached_verification.json')
        if prior.get('progress_after_disconnect') is not True:
            raise RuntimeError('Original actual-training detached proof is missing')
        atomic_json(old_path, prior)
    def observe():
        api = remote.Runpod()
        connection = remote.ssh_connect(api.pod(r['pod_id']), remote.STATE / 'runpod_ed25519')
        try:
            script = ("import pathlib,json,hashlib,time; p=pathlib.Path(" + repr(r['remote_root']) + "); "
                "q=p.parent.parent/'completion_acceptance_restore'/" + repr(child) + "; "
                "read=lambda f:json.loads(f.read_text()) if f.exists() else {}; "
                "stages=p.parent/" + repr(child) + "; v=p/'remote_validation.json'; "
                "print(json.dumps({'time':time.time(),'status':read(p/'status.json'),"
                "'watchdog':read(p/'watchdog.json'),'supervisor':read(p/'supervisor.json'),"
                "'restored_files':sum(x.is_file() for x in q.rglob('*')),'restored_bytes':sum(x.stat().st_size for x in q.rglob('*') if x.is_file()),"
                "'validation_mtime':v.stat().st_mtime if v.exists() else 0,"
                "'stage_seals':{str(x.relative_to(stages)):hashlib.sha256(x.read_bytes()).hexdigest() for x in stages.glob('seed_42/gen_*/stage_completion.json')}}))")
            return json.loads(remote.command(connection, 'python3 -c ' + shlex.quote(script), timeout=30))
        finally:
            connection.close()
    first = observe()
    if remote._startup_bad_states({'status': first['status']}, r)[1]:
        raise RuntimeError('Acceptance recovery failed or paused before detached verification')
    cutoff = now() + min(max_wait_seconds or 600, 600)
    def progress(x):
        return (x['status'].get('phase'), x['restored_files'], x['restored_bytes'], x['validation_mtime'], x['stage_seals'])
    while now() < cutoff:
        time.sleep(3)
        second = observe()
        if remote._startup_bad_states({'status': second['status']}, r)[1]:
            raise RuntimeError('Acceptance recovery failed or paused before detached verification')
        if any(second['stage_seals'].get(k) != v for k, v in first['stage_seals'].items()):
            raise RuntimeError('Completed stages changed during acceptance recovery')
        if (progress(second) != progress(first) and all(second[k].get('time', 0) > first[k].get('time', 0)
                for k in ('watchdog', 'supervisor'))):
            proof = {'run_id': run_id, 'pod_id': r['pod_id'], 'launch_id': r['launch_id'],
                'progress_after_disconnect': True, 'observations': [first, second],
                'observations_sha256': digest([first, second]),
                'original_training_proof_sha256': sha256(old_path),
                'scope': 'Verified recovery progress across closed SSH sessions; original GPU training and its detached proof are preserved, rather than claimed as newly executed.'}
            atomic_json(root / 'detached_verification.json', proof)
            connection = remote.ssh_connect(remote.Runpod().pod(r['pod_id']), remote.STATE / 'runpod_ed25519')
            try:
                sftp = connection.open_sftp()
                sftp.put(str(old_path), r['remote_root'] + '/original_training_detached_verification.json')
                target = r['remote_root'] + '/detached_verification.json'
                remote.upload_json(sftp, target + '.partial', proof)
                sftp.rename(target + '.partial', target + '.new')
                remote.command(connection, 'python3 -c ' + shlex.quote('from pathlib import Path; Path(' + repr(target + '.new') + ').replace(' + repr(target) + ')'))
            finally:
                connection.close()
            return proof
    raise RuntimeError('No actual recovery progress observed across SSH disconnection')


def main():
    os.environ.setdefault('HF_HUB_CACHE', r'C:\hf')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--percent', type=int, choices=(85, 90, 95), required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    from model_collapse import remote
    from scripts.operations.validate_high_fraction_completions import validate_one
    original = remote.verify_detached
    remote.verify_detached = recovery_progress
    try:
        validate_one(args.percent, args.run_id)
    finally:
        remote.verify_detached = original


if __name__ == '__main__':
    main()
