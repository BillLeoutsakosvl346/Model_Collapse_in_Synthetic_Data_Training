"""Run and recover the small GPU checks before authorized full launches."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from scripts.operations.launch_high_fraction_completions import launch_one


def validate_one(percent, resume_id=None):
    from model_collapse import remote
    from model_collapse.common import PROJECT, atomic_json, read_json, now, sha256
    from model_collapse.hub import download_snapshot_file, verify_recovered_files
    from model_collapse.workflow.completion import fetch_comparison
    from model_collapse.workflow.completion_validation import validate_local

    resources = launch_one(percent, smoke=True, resume_id=resume_id)
    run_id = resources['run_id']
    root = PROJECT / 'runs' / run_id
    api = remote.Runpod()
    cutoff = now() + 1500
    while now() < cutoff:
        pod = api.pod(resources['pod_id'])
        if pod['status'] in ('EXITED', 'TERMINATED'):
            break
        print(json.dumps({'validation': percent, 'run_id': run_id, 'provider_state': pod['status']}), flush=True)
        time.sleep(30)
    else:
        raise RuntimeError('Validation did not stop in its funded window; inspect its independent watchdog')
    return finalize_validation(percent, resources)


def promote_current_watchdog(root, resources):
    """Select the new owned shutdown receipt while retaining the old session."""
    from model_collapse.common import atomic_json, read_json, sha256
    import shutil
    root = Path(root)
    recovered = read_json(root / 'recovery_snapshot.json', {}).get('preserved_control_files', {}).get('watchdog_exit.json')
    if not recovered:
        return
    if recovered['recovered_path'] != 'fetched_control/' + recovered['sha256'] + '/watchdog_exit.json':
        raise RuntimeError('Unsafe recovered watchdog receipt path')
    source = root / recovered['recovered_path']
    guard = read_json(source)
    if sha256(source) != recovered['sha256']:
        raise RuntimeError('Recovered watchdog receipt checksum differs')
    if (guard.get('pod_id') != resources['pod_id'] or guard.get('time', 0) < resources['created_at'] or
            guard.get('reason') != 'validation_budget_guard'):
        raise RuntimeError('Recovered shutdown is not this validation session')
    previous = root / 'watchdog_exit.json'
    history = root / 'operational_history' / (sha256(previous) + '-watchdog_exit.json')
    history.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(previous, history)
    shutil.copyfile(source, previous)
    atomic_json(root / 'watchdog_control_promotion.json', {'run_id': resources['run_id'],
        'launch_id': resources['launch_id'], 'pod_id': resources['pod_id'],
        'old_receipt_path': history.relative_to(root).as_posix(), 'new_receipt_sha256': sha256(previous),
        'source_recovery_snapshot': read_json(root / 'recovery_snapshot.json')['revisions']})


def finalize_validation(percent, resources):
    from model_collapse import remote
    from model_collapse.common import PROJECT, atomic_json, read_json, now, sha256
    from model_collapse.hub import download_snapshot_file, verify_recovered_files
    from model_collapse.workflow.completion import fetch_comparison
    from model_collapse.workflow.completion_validation import validate_local
    run_id = resources['run_id']
    root = PROJECT / 'runs' / run_id
    if not remote.same_owner(resources, remote.local_resources(run_id)):
        raise RuntimeError('Validation resource ownership association changed')
    if remote.Runpod().pod(resources['pod_id'])['status'] != 'EXITED':
        raise RuntimeError('Validation finalization requires verified stopped compute')
    remote.status(run_id)
    fetch_comparison(root, resources, metadata_only=True)
    promote_current_watchdog(root, resources)
    # The parent contains only its unchanged baseline as bulk. Reuse a verified
    # local copy when present; trained smoke checkpoints were cold-restored and
    # re-evaluated privately by the production acceptance worker.
    index = read_json(root / 'artifact_index.json')
    revisions = read_json(root / 'recovery_snapshot.json')['revisions']
    for relative, meta in index['files'].items():
        path = root / relative
        try:
            verify_recovered_files(root, {relative: meta})
        except RuntimeError:
            pass
        else:
            continue
        candidates = list((PROJECT / 'runs').glob('*/' + relative))
        matching = next((p for p in candidates if p.is_file() and p.stat().st_size == meta['bytes'] and sha256(p) == meta['sha256']), None)
        source = matching or download_snapshot_file(run_id, resources['repos'], relative, meta, revisions)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise RuntimeError('Existing parent artifact differs from its pinned index')
        try:
            os.link(source, path)
        except OSError:
            import shutil
            shutil.copyfile(source, path)
    verify_recovered_files(root, index['files'])
    certificate = validate_local(root)
    from model_collapse.analysis.completion import report
    report(root)
    cleanup = remote.cleanup(run_id)
    receipt = {'condition': percent, 'run_id': run_id, 'runtime_sha256': resources['code_sha256'],
        'checks': certificate['checks'], 'remote_validation': read_json(root / 'remote_validation.json'),
        'cleanup': cleanup, 'time': now(),
        'cost_usd_conservative': remote.campaign_accrued(remote.ledger(), resources['acceptance_campaign'])}
    atomic_json(PROJECT / f'docs/validation/2026-11-07_completion-mixed{percent}-setup.json', receipt)
    print(json.dumps({'validated': percent, 'run_id': run_id, 'cost': receipt['cost_usd_conservative'], 'cleanup': cleanup}), flush=True)
    return receipt


def main():
    import argparse
    os.environ.setdefault('HF_HUB_CACHE', r'C:\hf' if os.name == 'nt' else '/workspace/hf-cache/hub')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--percent', type=int, choices=(85, 90, 95))
    parser.add_argument('--resume', metavar='RUN_ID')
    args = parser.parse_args()
    for percent in ([args.percent] if args.percent else [85, 90, 95]):
        if args.resume and args.percent is None:
            parser.error('--resume requires --percent')
        validate_one(percent, args.resume)


if __name__ == '__main__':
    main()
