"""Read-only first-stage and pinned-backup audit for the three full jobs."""
from __future__ import annotations

import json
import math
import os
import shlex


CODE = r'''
import json, math, pathlib, numpy as np
from model_collapse.common import read_json, sha256
from model_collapse.workflow.completion import child_roots, bank_path, verify_stage, forecast
from model_collapse.completion_data import load_pairs
from model_collapse.completion_mixing import select_rows
root = pathlib.Path(ROOT)
c = read_json(root/'config.json')
child = next(iter(child_roots(root,c).values()))
stage = child/'seed_42/gen_01'
snapshot = read_json(child/'backup_snapshot.json',{})
index = read_json(child/'artifact_index.json',{})
if not (stage/'stage_completion.json').exists() or not snapshot or not index:
    print(json.dumps({'ready':False,'reason':'first complete stage/backup pending'}))
else:
    if sha256(child/'artifact_index.json') != snapshot['artifact_index_sha256']:
        print(json.dumps({'ready':False,'reason':'backup transaction pending'}))
        raise SystemExit(0)
    seal = 'seed_42/gen_01/stage_completion.json'
    if index['files'].get(seal,{}).get('sha256') != sha256(stage/'stage_completion.json'):
        print(json.dumps({'ready':False,'reason':'first stage not yet in pinned backup'}))
    else:
        arm = read_json(child/'config.json')
        training = read_json(stage/'training.json')
        metrics = verify_stage(stage,arm,training['lineage'])
        baseline = read_json(root/'seed_42/gen_00/metrics.json')
        initializer = baseline['weights_sha256']
        assert training['initial_weights_sha256'] == initializer
        assert training['selected_weights_sha256'] != initializer
        assert training['lineage']['generator_weights_sha256'] == initializer
        assert training['lineage']['optimizer_parent_sha256'] is None
        assert training['lineage']['bank_manifest_sha256'] == c['prepared_bank']['manifest_sha256']
        assert training['global_step'] == 79 and training['selected_epoch'] == 1
        assert training['executed_epochs'] == 1
        values = np.load(stage/'training_input/tokens.npy',allow_pickle=False)
        attention = np.load(stage/'training_input/attention.npy',allow_pickle=False)
        loss = np.load(stage/'training_input/loss_mask.npy',allow_pickle=False)
        prompt, real, _ = load_pairs(bank_path(root,c),'cycle_01',10000)
        assert values.shape == (10000,129) and np.all(attention==1)
        assert np.array_equal(values[:,1:65],prompt)
        assert np.all(loss[:,:65]==0) and np.all(loss[:,65:]==1) and int(loss.sum())==640000
        fraction = arm['synthetic_fraction']
        selection, real_count = select_rows(10000,fraction,c['seed'],1)
        for key,array in selection.items():
            assert np.array_equal(np.load(stage/'mixture'/(key+'.npy')),array)
        indices = selection['synthetic_indices']
        expected = real.copy()
        generated = np.load(child/'synthetic_for_gen_01/tokens.npy')
        assert generated.shape == (len(indices),64)
        expected[indices] = generated
        assert np.array_equal(values[:,65:],expected)
        assert metrics['validation']['scored_tokens']==468608 and metrics['test']['scored_tokens']==457664
        assert all(math.isfinite(metrics[k]['perplexity']) for k in ('validation','test'))
        print(json.dumps({'ready':True,'run_id':root.name,'child_run_id':child.name,'synthetic_fraction':fraction,
          'training_pairs':10000,'real_pairs':real_count,'synthetic_pairs':len(indices),'scored_training_tokens':640000,
          'global_step':79,'executed_epochs':1,'selected_epoch':1,'first_stage_sha256':sha256(stage/'stage_completion.json'),
          'baseline_weights_sha256':initializer,'selected_weights_sha256':training['selected_weights_sha256'],
          'baseline_test_perplexity':baseline['test']['perplexity'],'first_test_perplexity':metrics['test']['perplexity'],
          'backup_snapshot':snapshot,'full_benchmark_forecast':forecast(read_json(root/'benchmark.json'),c,.72+60*.07/730+20*.1/730),
          'checks':{'actual_weight_update':True,'own_generator_and_initializer':True,'correct_frozen_mixture':True,
             'continuation_only_loss':True,'fixed_evaluation_counts':True,'selected_checkpoint_and_stage_hashes':True,
             'first_stage_in_pinned_backup_index':True}}))
'''


def collect(run_id):
    from model_collapse import remote
    from model_collapse.common import PROJECT, atomic_json, read_json
    from model_collapse.hub import retry_transfer
    from huggingface_hub import hf_hub_download
    from model_collapse.common import sha256
    resources = remote.local_resources(run_id)
    api = remote.Runpod()
    pod = api.pod(resources['pod_id'])
    if pod['status'] != 'RUNNING':
        raise RuntimeError('Full startup audit requires its running owned GPU')
    client = remote.ssh_connect(pod,remote.STATE/'runpod_ed25519')
    try:
        code = CODE.replace('ROOT',repr(resources['remote_root']),1)
        command = 'cd /workspace/project && . /run/iib-week1-secrets && /opt/iib-venv/bin/python -c ' + shlex.quote(code)
        result = json.loads(remote.command(client,command,timeout=120))
    finally:
        client.close()
    if not result['ready']:
        return result
    snapshot = result['backup_snapshot']
    src = retry_transfer(hf_hub_download,resources['repos']['dataset'],result['child_run_id']+'/artifact_index.json',
        repo_type='dataset',revision=snapshot['revisions']['dataset'])
    if sha256(src) != snapshot['artifact_index_sha256']:
        raise RuntimeError('Pinned private first-stage index round-trip differs')
    proof = read_json(PROJECT/'runs'/run_id/'detached_verification.json',{})
    if proof.get('pod_id') != resources['pod_id'] or not proof.get('progress_after_disconnect'):
        return {'ready':False,'reason':'native detached startup verification pending'}
    result['detached_proof'] = proof
    result['checks']['pinned_backup_index_download_verified'] = True
    result['provider_state'] = pod['status']
    result['resources'] = resources
    result['scope'] = 'First full-stage file hashes and pinned private index verified; complete full-run cold restoration is separate future work.'
    atomic_json(PROJECT/'runs'/run_id/'full_startup_audit.json',result)
    return result


def main():
    os.environ.setdefault('HF_HUB_CACHE',r'C:\hf')
    import argparse
    from model_collapse import remote
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id',nargs='?')
    args=parser.parse_args()
    jobs=[args.run_id] if args.run_id else [x['run_id'] for x in remote.ledger()['launches']
        if x.get('parallel_campaign')=='fineweb-high-fractions-20261107' and x['kind']=='full']
    for run in jobs:
        result=collect(run)
        print(json.dumps({'run_id':run,'ready':result['ready'],
              **({'fraction':result['synthetic_fraction'],'checks':result['checks']} if result['ready'] else {'reason':result['reason']})}),flush=True)


if __name__=='__main__':
    main()
