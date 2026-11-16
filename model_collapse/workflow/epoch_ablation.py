"""Detached, source-sealed one-epoch counterfactuals from completed SGD stages."""
from __future__ import annotations

import gc
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from ..common import atomic_json, digest, now, read_json, sha256, verify_files

ALLOCATION = 'smollm2_sgd_epoch_ablation'
BRANCHES = ('both', 'real', 'synthetic')


def validate_config(c):
    expected = {'workflow':'epoch_ablation', 'source_arm':'mixed99', 'cycles':[3,6,10],
                'branches':list(BRANCHES), 'probe_pairs':128, 'replay_tolerance':1e-5,
                'required_gpu':'NVIDIA RTX PRO 4000 Blackwell', 'seed':42,
                'max_compute_seconds':5400, 'validation_max_seconds':900,
                'network_volume_gb':30, 'backup_capacity_gb':10}
    if any(c.get(k) != v for k,v in expected.items()):
        raise ValueError('Unsupported whole-epoch ablation protocol')
    if c.get('source_parent',{}).get('run_id') != '20261114T001352Z-full-completion64-mixtures':
        raise ValueError('Epoch ablation must use the completed SGD control')
    if (len(c.get('source_config_sha256','')) != 64 or c.get('allocation') != ALLOCATION
            or c.get('task_budget_usd') != 4 or c.get('reserve_usd') != 2
            or c.get('validation_cap_usd') != .15):
        raise ValueError('Missing pinned identity/approved budget')
    return c


def batch_selection(case, miniature=False):
    order = np.random.default_rng(42 + case['cycle']).permutation(10000)
    if not miniature:
        return np.arange(10000), order
    # Actual cycle-6 slots 1/5/79, with 5/0/1 real examples respectively.
    rows = np.r_[order[:128], order[512:640], order[9984:]]
    if case['cycle'] != 6 or [int((case['labels'][r] == 0).sum()) for r in (rows[:128],rows[128:256],rows[256:])] != [5,0,1]:
        raise RuntimeError('Miniature source batches changed')
    return rows, np.arange(len(rows))


def _evaluate(checkpoint, native, arrays):
    from ..science import engine
    model = engine.load_model(native, checkpoint)
    out = {split:engine.evaluate(model, x[0], native, masks=x[1], loss_masks=x[2]) for split,x in arrays.items()}
    del model; gc.collect()
    return out


def tensor_distance(left, right):
    from safetensors import safe_open
    total = denom = maximum = 0.
    with safe_open(str(Path(left)/'model.safetensors'), framework='pt', device='cpu') as a, safe_open(str(Path(right)/'model.safetensors'), framework='pt', device='cpu') as b:
        if set(a.keys()) != set(b.keys()):
            raise RuntimeError('Replay tensor names differ')
        for name in a.keys():
            x,y = a.get_tensor(name).double(),b.get_tensor(name).double()
            total += float((x-y).square().sum()); denom += float(y.square().sum())
            maximum = max(maximum, float((x-y).abs().max()))
    return {'relative_l2':math.sqrt(total/max(denom,1e-30)), 'max_absolute':maximum}


def run_branch(root, case, source, evaluations, probe, *, miniature=False):
    from ..science import engine
    from ..science.epoch_ablation import EpochObserver
    root = Path(root)
    branch = root/('acceptance' if miniature else 'cases')/f"cycle_{case['cycle']:02d}"/source
    stage = branch/f"seed_42/gen_{case['cycle']:02d}"
    rows, order = batch_selection(case, miniature)
    binding = {'cycle':case['cycle'], 'source':source, 'initializer':case['initial_hash'],
               'historical_lineage':case['historical_lineage'], 'order':digest(order.tolist()),
               'row_mapping':digest(rows.tolist()), 'scope':'miniature' if miniature else 'full_epoch',
               'probe_identity':probe['identity'], 'protocol':'epoch_ablation_original_batch_v1',
               'learning_rate':.01, 'precision':'fp32', 'microbatch':32, 'effective_batch':128}
    native = {**case['native'], 'purpose':'epoch_ablation', 'update_diagnostics':{'enabled':False},
              'epoch_ablation':binding, 'epochs':1, 'checkpoint_selection':'final_epoch'}
    old = read_json(branch/'branch_completion.json')
    if old:
        if old['binding'] != binding:
            raise RuntimeError('Completed branch source/settings changed')
        verify_files(branch, old['files'])
        old_seal=read_json(stage/'epoch_diagnostics_manifest.json')
        old_rows=[json.loads(line) for line in (root/old['records']).read_text().splitlines()]
        from .selection import training_contract
        count=3 if miniature else 79
        if (old_seal['binding']!=binding or len(old_rows)!=count
                or old['training']['training_contract_sha256']!=training_contract(native)
                or [r['local_step'] for r in old_rows]!=list(range(1,count+1))):
            raise RuntimeError('Completed branch probe/timeline changed')
        return old
    branch.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free<3_000_000_000:
        raise RuntimeError('Working volume has less than 3 GB free before a new branch')
    atomic_json(branch/'config.json', native)
    observer = EpochObserver(stage, native, probe, binding, rows)
    result = engine.train(branch, case['cycle'], case['tokens'][rows], evaluations['validation'][0], native,
        expected_base_hash=case['initial_hash'], initial_checkpoint=case['checkpoint'],
        masks=case['attention'][rows], loss_masks=case['scoring'][rows],
        val_masks=evaluations['validation'][1], val_loss_masks=evaluations['validation'][2],
        optimizer_checkpoint=case['state'], total_steps=790, lineage=binding,
        source_labels=case['labels'][rows], training_source=source, row_order=order, step_observer=observer,
        progress_callback=lambda step,**kw:atomic_json(root/'status.json', {'state':'running',
            'phase':'acceptance' if miniature else 'full', 'cycle':case['cycle'], 'source':source,
            'global_step':step, 'local_step':step-(case['cycle']-1)*79, 'time':now()}))
    metrics = _evaluate(stage/'checkpoint', native, {'test':evaluations['test']})
    metrics['validation']=read_json(stage/'training_history.json')[-1]['validation_metrics']
    seal = observer.verify(result['selected_weights_sha256'])
    records = [json.loads(line) for line in (stage/seal['records_path']).read_text().splitlines()]
    expected_steps = 3 if miniature else 79
    if len(records) != expected_steps:
        raise RuntimeError('Branch has an incorrect update count')
    replay = None
    if source == 'both' and not miniature:
        replay = {**tensor_distance(stage/'checkpoint', case['expected_checkpoint']),
                  'bitwise_tensor_hash_equal':result['selected_weights_sha256']==case['terminal_hash'],
                  'maximum_probe_nll_discrepancy':max(abs(r['probe_nll']-h['states']['mixed']['loss']) for r,h in zip(records,case['historical_records'])),
                  'validation_nll_discrepancy':abs(metrics['validation']['loss']-case['after_metrics']['validation']['loss']),
                  'test_nll_discrepancy':abs(metrics['test']['loss']-case['after_metrics']['test']['loss'])}
        if any(replay[k] > 1e-5 for k in ('relative_l2','maximum_probe_nll_discrepancy','validation_nll_discrepancy','test_nll_discrepancy')):
            atomic_json(branch/'replay_failed.json', replay)
            raise RuntimeError('Both-source replay differs from the original epoch beyond tolerance')
    movement = tensor_distance(stage/'checkpoint', case['checkpoint'])
    atomic_json(branch/'metrics.json', metrics)
    files = {p.relative_to(branch).as_posix():sha256(p) for p in branch.rglob('*') if p.is_file()
             and 'resume' not in p.parts and '.partial' not in p.name}
    done = {'binding':binding, 'training':result, 'metrics':metrics, 'replay':replay, 'movement':movement,
            'updates':len(records), 'initial_probe':seal['initial_probe'], 'final_probe_nll':records[-1]['probe_nll'],
            'checkpoint':str((stage/'checkpoint').relative_to(root)), 'records':str((stage/seal['records_path']).relative_to(root)),
            'files':files, 'complete':True}
    atomic_json(branch/'branch_completion.json', done)
    return done


def verify_acceptance(root, results, case, started, evaluations, probe):
    from ..cloud.s3_artifacts import client
    from ..science import engine
    root = Path(root)
    checks = {'three_independent_initializers':all(x['training']['initial_weights_sha256']==case['initial_hash'] for x in results),
              'nine_scheduled_updates':sum(x['updates'] for x in results)==9,
              'changed_and_reloaded_weights':all(x['training']['terminal_weights_sha256']==x['training']['selected_weights_sha256'] and x['movement']['relative_l2']>0 for x in results),
              'finite_losses':all(math.isfinite(x['final_probe_nll']) for x in results)}
    real = results[1]
    record = [json.loads(l) for l in (root/real['records']).read_text().splitlines()][1]
    checks['empty_source_zero_movement'] = record['gradient_norm']==0 and record['probe_nll']==record['probe_nll_before']
    # Force a cold S3 read to a new directory, bypassing the shared bulk cache.
    index = read_json(root/'artifact_index.json')
    path = results[0]['checkpoint']+'/model.safetensors'
    meta = index['files'][path]
    cold = root/'.cold_acceptance'
    cold.mkdir(exist_ok=True)
    for name in ('config.json','generation_config.json'):
        src = root/results[0]['checkpoint']/name
        if src.exists():
            shutil.copyfile(src, cold/name)
    obj = meta['storage']; store = {k:v for k,v in obj.items() if k!='key'}
    client(store).download_file(obj['bucket'],obj['key'],str(cold/'model.safetensors'))
    checks['private_checkpoint_cold_hash'] = sha256(cold/'model.safetensors')==meta['sha256']
    model = engine.load_model(case['native'],cold)
    checks['cold_tensor_identity'] = engine.weights_hash(model)==results[0]['training']['selected_weights_sha256']
    restored=engine.evaluate(model,evaluations['validation'][0][:32],case['native'],
                            masks=evaluations['validation'][1][:32],loss_masks=evaluations['validation'][2][:32])
    checks['cold_evaluation_reproduced']=abs(restored['loss']-results[0]['metrics']['validation']['loss'])<=1e-12
    production_probe=engine.evaluate(model,probe['tokens'],case['native'],masks=probe['attention'],loss_masks=probe['scoring'])
    del model; gc.collect()
    if not all(checks.values()):
        atomic_json(root/'acceptance_failed.json',checks)
        raise RuntimeError('Miniature epoch-ablation acceptance failed')
    return {'checks':checks, 'time':now(), 'seconds':now()-started, 'updates':9,
            'production_128_probe_seconds':production_probe['seconds'],
            'scope':'Original cycle-6 slots 1,5,79; 272 pairs, 6 real/266 synthetic. Not a full scientific epoch.'}


def run(root, validation=False):
    from filelock import FileLock
    from ..artifacts.epoch_sources import restore
    from ..hub import backup
    from ..science import engine
    root = Path(root); c=validate_config(read_json(root/'config.json')); r=read_json(root/'resources.json')
    with FileLock(str(root/'worker.lock'), timeout=0):
        import torch
        torch.set_num_threads(4)
        try:
            accepted=read_json(root/'acceptance.json')
            if accepted and (accepted.get('runtime_sha256')!=r['code_sha256'] or accepted.get('configuration_sha256')!=digest(c)):
                raise RuntimeError('Saved acceptance does not bind this runtime/protocol')
            if not read_json(root/'acceptance.json'):
                atomic_json(root/'coordinator.json',{'phase':'acceptance','time':now()})
                atomic_json(root/'status.json',{'state':'running','phase':'acceptance_restore','time':now()})
                cases, ev, probe, native = restore(root,c,cycles=[6],replay_endpoint=False)
                import importlib.metadata
                versions={name:importlib.metadata.version(name) for name in ('torch','transformers','datasets')}
                if versions!=native['required_environment'] or torch.cuda.get_device_name()!=c['required_gpu']:
                    raise RuntimeError('Acceptance environment differs from the historical run')
                atomic_json(root/'environment.json',{'packages':versions,'gpu':torch.cuda.get_device_name(),'fp32':True,'tf32':False})
                mini_ev={k:tuple(x[:32] for x in v) for k,v in ev.items()}
                mini_probe={**probe, **{k:probe[k][:32] for k in ('tokens','attention','scoring','documents')},
                            'identity':digest({'parent':probe['identity'],'pairs':32})}
                t=now(); results=[run_branch(root,cases[0],s,mini_ev,mini_probe,miniature=True) for s in BRANCHES]
                backup(root)
                acceptance=verify_acceptance(root,results,cases[0],t,ev,probe)
                if now()>=r['acceptance_deadline']:
                    raise RuntimeError('GPU acceptance exceeded its combined deadline')
                acceptance.update(runtime_sha256=r['code_sha256'],configuration_sha256=digest(c),
                                  paid_seconds=now()-r['created_at'])
                # Forecast complete 711-slot matrix, endpoint evaluations and backups.
                journals=[[json.loads(line) for line in (root/x['records']).read_text().splitlines()] for x in results]
                train_rate=max(row['seconds'] for journal in journals for row in journal[:2])
                eval_seconds=sum(v['seconds'] for v in results[0]['metrics'].values())
                fixed=sum(x['training']['seconds']-read_json(root/x['checkpoint'].replace('/checkpoint','/training_history.json'))[-1]['train_seconds'] for x in results)/3
                forecast=711*(train_rate+acceptance['production_128_probe_seconds']) + 12*eval_seconds*(len(ev['test'][0])+len(ev['validation'][0]))/64 + 9*fixed + 300
                acceptance['forecast_seconds']=forecast*1.3
                if now()+acceptance['forecast_seconds']>r['absolute_deadline']-30:
                    atomic_json(root/'acceptance_forecast_blocker.json',acceptance)
                    raise RuntimeError('Measured full matrix cannot fit the remaining paid deadline')
                atomic_json(root/'acceptance.json',acceptance)
                (root/'validation_guard_deadline.json').unlink(missing_ok=True)
                backup(root)
            atomic_json(root/'status.json',{'state':'running','phase':'full_restore','time':now()})
            atomic_json(root/'coordinator.json',{'phase':'full','time':now()})
            cases, evaluations, probe, native = restore(root,c)
            results=[]
            for case in cases:
                baseline=_evaluate(case['checkpoint'],native,evaluations)
                atomic_json(root/f"baseline_cycle_{case['cycle']:02d}.json",baseline)
                for source in BRANCHES:
                    if now()>r['absolute_deadline']-60:
                        raise RuntimeError('Insufficient remaining deadline for another branch')
                    results.append(run_branch(root,case,source,evaluations,probe))
                    backup(root)
            atomic_json(root/'training_matrix_complete.json',{'complete':True,'branches':len(results),'updates':sum(x['updates'] for x in results),'time':now()})
            atomic_json(root/'coordinator.json',{'phase':'analysis','time':now()})
            from ..analysis.epoch_ablation import report
            if not read_json(root/'generated/results/analysis_manifest.json'):
                if (root/'generated/results').exists():
                    (root/'generated/results').rename(root/'generated'/('incomplete-results-'+str(int(now()))))
                report(root,root/'generated/results')
            else:
                verify_files(root/'generated/results',read_json(root/'generated/results/analysis_manifest.json')['outputs'])
            atomic_json(root/'status.json',{'state':'complete','branches':9,'time':now()})
            backup(root)
            atomic_json(root/'completion.json',{'complete':True,'branches':9,'updates':711,'time':now()})
            backup(root)
            atomic_json(root/'coordinator.json',{'phase':'complete','time':now()})
        except BaseException as exc:
            atomic_json(root/'status.json',{'state':'failed','error':str(exc),'time':now()})
            raise


def validate_local(root):
    root=Path(root); done=read_json(root/'completion.json')
    if not done or done.get('updates')!=711 or done.get('branches')!=9:
        raise RuntimeError('Whole-epoch matrix is incomplete')
    for cycle in (3,6,10):
        for source in BRANCHES:
            p=root/f'cases/cycle_{cycle:02d}'/source/'branch_completion.json';b=read_json(p)
            if not b or b.get('updates')!=79 or b['binding']['source']!=source or b['binding']['cycle']!=cycle:
                raise RuntimeError('Missing or inconsistent completed branch')
            verify_files(p.parent,b['files'])
            records=[json.loads(line) for line in (root/b['records']).read_text().splitlines()]
            if ([x['step'] for x in records]!=list(range((cycle-1)*79+1,cycle*79+1))
                    or any(x['source']!=source for x in records)
                    or sorted(i for x in records for i in x['original_rows'])!=list(range(10000))):
                raise RuntimeError('Completed branch has a different historical timeline')
    return {'complete':True,'updates':711,'branches':9}
