"""Serialize campaign admission/deployment, overlap independent startup checks."""
from __future__ import annotations

import argparse
import os


def staged_launch(percent, **kwargs):
    from filelock import FileLock
    from model_collapse import remote
    from model_collapse.common import STATE, config
    c = config(kwargs['config_path'])
    if (kwargs.get('smoke') or kwargs.get('resume_id') or c['kind'] != 'full' or
            c.get('parallel_campaign') != 'fineweb-high-fractions-20261107' or
            c['completion_arms'] != [{'id': f'mixed{percent}', 'synthetic_fraction': percent / 100}]):
        raise RuntimeError('Concurrent startup is restricted to the authorized new full conditions')
    # The ordinary global launch lock includes the entire first-epoch health
    # observation. Keep admission/deployment serialized under a campaign lock,
    # then allow the next independent GPU to deploy while this one trains.
    gate = FileLock(str(STATE / 'high-fraction-provision.lock'), timeout=600)
    gate.acquire()
    observer = remote.verify_detached
    def observe(run_id, **options):
        gate.release()
        return observer(run_id, **options)
    remote.verify_detached = observe
    try:
        return remote._launch(**kwargs)
    finally:
        remote.verify_detached = observer
        if gate.is_locked:
            gate.release()


def main():
    os.environ.setdefault('HF_HUB_CACHE', r'C:\hf')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--percent', type=int, choices=(85, 90, 95), required=True)
    args = parser.parse_args()
    from filelock import FileLock
    from model_collapse import remote
    from model_collapse.common import STATE
    from scripts.operations.launch_high_fraction_completions import launch_one
    original = remote.launch
    with FileLock(str(STATE / f'high-fraction-{args.percent}.lock'), timeout=0):
        remote.launch = lambda **kwargs: staged_launch(args.percent, **kwargs)
        try:
            launch_one(args.percent)
        finally:
            remote.launch = original


if __name__ == '__main__':
    main()
