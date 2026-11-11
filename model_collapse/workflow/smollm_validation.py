"""Bounded model-portability acceptance: two updates, one mixed condition."""
from __future__ import annotations

import gc
import math
from pathlib import Path
import numpy as np

from ..common import STATE, atomic_json, read_json, sha256, verify_files
from ..completion_data import assemble, load_pairs
from ..completion_mixing import select_rows
from . import smollm


def fraction_checks():
    audits=[];previous=set()
    for spec in smollm.ARMS:
        f=spec['synthetic_fraction'];selection,real_count=select_rows(10000,f,42,1)
        idx=selection['synthetic_indices'];current=set(idx.tolist())
        expected=[0,5000,8000,9500,9900,9990,10000][len(audits)]
        if len(idx)!=expected or real_count!=10000-expected or not previous.issubset(current):
            raise RuntimeError('Seven-fraction count/nesting check failed')
        original=np.arange(10000*64,dtype=np.int32).reshape(10000,64)
        mixed=original.copy();mixed[idx]=-1
        if not np.array_equal(mixed[selection['source_labels']==0],original[selection['source_labels']==0]):
            raise RuntimeError('Real target placement changed')
        audits.append({'fraction':f,'synthetic_pairs':len(idx),'real_pairs':real_count,'nested':True})
        previous=current
    return audits


def validate_remote(root):
    import torch
    from .. import engine
    from .completion import child_roots, bank_path, verify_completed_arm
    from .completion_restore import verify_terminal_restore
    from .profiles import protocol_fingerprint
    root=Path(root);c=read_json(root/'config.json');r=read_json(root/'resources.json')
    smollm.validate(c)
    if c['kind']!='smoke':raise RuntimeError('Fast validation is smoke-only')
    children=child_roots(root,c)
    if list(children)!=['mixed50']:raise RuntimeError('Fast GPU coverage must be exactly mixed50')
    child=children['mixed50'];arm=read_json(child/'config.json');bank=bank_path(child,arm)
    verify_completed_arm(child,arm)
    previous=read_json(root/'seed_42/gen_00/metrics.json')['weights_sha256'];optimizer_hash=None;stages=[]
    for cycle in (1,2):
        stage=child/f'seed_42/gen_{cycle:02d}';m=read_json(stage/'metrics.json');t=m['training']
        history=read_json(stage/'training_history.json')
        if (len(history)!=1 or history[0]['global_step']!=cycle or
            history[0]['scored_training_tokens']!=8192 or
            not all(math.isfinite(history[0][k]) for k in ('train_loss','validation_loss','learning_rate_end')) or
            not math.isfinite(m['training_evaluation']['loss'])):
            raise RuntimeError('Fast training did not record one finite optimizer update')
        prompt,real,_=load_pairs(bank,f'cycle_{cycle:02d}',128)
        selection,n_real=select_rows(128,.5,42,cycle);idx=selection['synthetic_indices']
        generated=np.load(child/f'synthetic_for_gen_{cycle:02d}/tokens.npy')
        if generated.shape!=(64,64):raise RuntimeError('Fast generated pool shape differs')
        expected=real.copy();expected[idx]=generated
        tokens,attention,scoring=assemble(prompt,expected,arm['bos_token_id'])
        for filename,values in [('tokens.npy',tokens),('attention.npy',attention),('loss_mask.npy',scoring)]:
            if not np.array_equal(np.load(stage/'training_input'/filename),values):raise RuntimeError('Native inputs/masks differ')
        if not np.all(scoring[:,:65]==0) or not np.all(scoring[:,65:]==1) or int(scoring.sum())!=8192:
            raise RuntimeError('Prompt/continuation loss boundary differs')
        lineage=t['lineage']
        if (t['initial_weights_sha256']!=previous or lineage['generator_weights_sha256']!=previous or
            t['optimizer_parent_sha256']!=optimizer_hash or t['global_step']!=cycle or
            t['selected_weights_sha256']==previous or t['selected_epoch']!=1 or t['executed_epochs']!=1):
            raise RuntimeError('Native continued-state handoff differs')
        state=torch.load(stage/'optimizer_state/cycle.pt',map_location='cpu',weights_only=False)
        if state['global_step']!=cycle or state['scheduler']['last_epoch']!=cycle or any(int(x['step'])!=cycle for x in state['optimizer']['state'].values()):
            raise RuntimeError('AdamW/scheduler state did not continue')
        if not all(math.isfinite(m[s]['loss']) and m[s]['scored_tokens']==2048 for s in ('validation','test')):
            raise RuntimeError('Fast held-out loss is invalid')
        stages.append({'cycle':cycle,'initializer':previous,'selected':t['selected_weights_sha256'],
                       'generator':lineage['generator_weights_sha256'],'global_step':cycle,'real_pairs':n_real,'synthetic_pairs':len(idx)})
        previous=t['selected_weights_sha256'];optimizer_hash=sha256(stage/'optimizer_state/cycle.pt')
        del state
    final=child/'seed_42/gen_02';model=engine.load_model(arm,final/'checkpoint')
    if engine.weights_hash(model)!=previous:raise RuntimeError('Terminal checkpoint reload failed')
    p,t,_=load_pairs(bank,'test',1);ids,attention,mask=assemble(p,t,arm['bos_token_id'])
    model.eval()
    with torch.inference_mode(),engine.autocast(arm):
        logits=model(input_ids=engine.tensor(ids),attention_mask=engine.tensor(attention),use_cache=False).logits
        actual=engine.shifted_nll(logits,engine.tensor(ids),engine.tensor(mask)).sum().item()/64
    direct=float((-torch.log_softmax(logits[0,64:128].float(),dim=-1)[torch.arange(64,device=logits.device),engine.tensor(t[0])]).mean())
    if abs(actual-direct)>1e-5:raise RuntimeError('Direct conditional NLL differs')
    del model,logits;gc.collect();torch.cuda.empty_cache()
    restore=verify_terminal_restore(root,child,arm)
    checks={k:True for k in smollm.CHECKS-{'detached_progress_verified','independent_watchdog_and_provider_stop_verified'}}
    atomic_json(root/'remote_validation.json',{'workflow':'completion_comparison','acceptance_scope':'smollm2_fast_model_portability_v1',
        'source_code_sha256':r['code_sha256'],'protocol_fingerprint':protocol_fingerprint(c),'checks':checks,
        'gpu_tested_fractions':[.5],'offline_fractions':fraction_checks(),'optimizer_updates':2,'stages':stages,
        'direct_nll_absolute_error':abs(actual-direct),'terminal_restore':restore,
        'coverage_limit':'One representative mixed arm; no claim of seven-condition GPU execution, full-data training or every-stage cold restoration.'})
    benchmark=read_json(root/'benchmark.json');benchmark['measured_stage_bytes']=sum(p.stat().st_size for p in final.rglob('*') if p.is_file())
    atomic_json(root/'benchmark.json',benchmark)


def validate_local(root):
    from ..remote import local_resources, source_archive, Runpod
    from ..mixing import certificate_name
    from .completion import verify_recovery, benchmark_filename
    from .profiles import require_acceptance
    root=Path(root);verify_recovery(root)
    c=read_json(root/'config.json');r=local_resources(root.name);result=read_json(root/'remote_validation.json')
    guard=read_json(root/'watchdog_exit.json',{});pod=Runpod().pod(r['pod_id'])
    if pod['status']!='EXITED' or guard.get('pod_id')!=r['pod_id'] or guard.get('reason')!='complete':
        raise RuntimeError('Ordinary independent watchdog completion/shutdown unverified')
    detached=read_json(root/'detached_verification.json',{})
    if detached.get('pod_id')!=r['pod_id'] or detached.get('progress_after_disconnect') is not True:
        raise RuntimeError('Detached progress not verified')
    runtime=source_archive()[1]
    result['checks'].update(detached_progress_verified=True,independent_watchdog_and_provider_stop_verified=True)
    require_acceptance(c,result,runtime)
    result.update(validated_run_id=root.name,provider_pod_state=pod['status'],gpu=r['gpu'],
                  validation_storage_backend=r.get('storage_backend','network'),full_working_storage_backend='network')
    atomic_json(root/'validation_results.json',result);atomic_json(STATE/certificate_name(c,runtime),result)
    atomic_json(STATE/benchmark_filename(c),read_json(root/'benchmark.json'))
    return result
