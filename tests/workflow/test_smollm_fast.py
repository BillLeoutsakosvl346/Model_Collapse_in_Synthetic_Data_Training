"""Focused portability tests, using tiny tensors and no cloud/private caches."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from model_collapse.common import atomic_json, read_json, sha256
from model_collapse.workflow import completion, smollm
from model_collapse.workflow.profiles import protocol_fingerprint, require_acceptance
from model_collapse.workflow.smollm_validation import fraction_checks
from model_collapse.completion_data import assemble, audit_bank

PROJECT=Path(__file__).resolve().parents[2]


def profile(kind='full'):
    return yaml.safe_load((PROJECT/f'comparisons/smollm2_135m_fineweb_completion64_fractions/{kind}.yaml').read_text(encoding='utf-8'))


def test_seven_profiles_and_fast_certificate_binding():
    full,smoke=profile(),profile('smoke')
    completion.validate_config(full);completion.validate_config(smoke)
    assert len(completion.arm_specs(full))==7
    assert completion.arm_specs(smoke)==[smollm.ARMS[1]]
    assert protocol_fingerprint(full)==protocol_fingerprint(smoke)
    for name,kind in [('real','real'),('synthetic','synthetic'),('mixed99p9','mixed')]:
        child=completion.arm_config(full,name);completion.validate_config(child);assert child['data_kind']==kind
    certificate={'workflow':'completion_comparison','source_code_sha256':'runtime','protocol_fingerprint':protocol_fingerprint(smoke),'checks':dict.fromkeys(smollm.CHECKS,True)}
    require_acceptance(full,certificate,'runtime')
    wrong=copy.deepcopy(full);wrong['model_revision']='a'*40
    with pytest.raises(RuntimeError,match='protocol'):require_acceptance(wrong,certificate,'runtime')
    with pytest.raises(RuntimeError,match='source version'):require_acceptance(full,certificate,'new runtime')
    certificate['checks']['private_terminal_restore']=False
    with pytest.raises(RuntimeError,match='acceptance'):require_acceptance(full,certificate,'runtime')


@pytest.mark.parametrize('field,value',[('model','facebook/opt-125m'),('learning_rate',1e-5),('num_beams',1),('new_tokens',63),('microbatch',16),('generations',9),('network_volume_gb',60),('do_sample',True),('temperature',.7),('generation_decoder','sampling')])
def test_smollm_rejects_protocol_drift(field,value):
    c=profile();c[field]=value
    with pytest.raises(ValueError):completion.validate_config(c)


def test_seven_mixture_counts_and_nested_indices():
    actual=fraction_checks()
    assert [x['synthetic_pairs'] for x in actual]==[0,5000,8000,9500,9900,9990,10000]
    assert actual[-2]['real_pairs']==10


def test_native_bos_zero_and_conditional_loss_oracle():
    from transformers import LlamaConfig,LlamaForCausalLM
    torch.set_num_threads(1);torch.manual_seed(42)
    model=LlamaForCausalLM(LlamaConfig(vocab_size=64,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,bos_token_id=0,eos_token_id=0,pad_token_id=0)).eval()
    prompt=np.arange(1,65,dtype=np.int32)[None,:]%63+1;target=prompt[:,::-1].copy()
    ids,attention,scoring=assemble(prompt,target,0)
    assert ids.shape==(1,129) and ids[0,0]==0 and np.all(attention==1)
    assert scoring.sum()==64 and not scoring[:,:65].any()
    from model_collapse.science.engine import shifted_nll
    with torch.no_grad():logits=model(torch.tensor(ids).long(),attention_mask=torch.tensor(attention),use_cache=False).logits
    loss=shifted_nll(logits,torch.tensor(ids).long(),torch.tensor(scoring)).sum()/64
    oracle=-torch.log_softmax(logits[0,64:128].float(),dim=-1)[torch.arange(64),torch.tensor(target[0]).long()].mean()
    torch.testing.assert_close(loss,oracle)
    changed=logits.clone();changed[:,10,:]=100
    torch.testing.assert_close(shifted_nll(changed,torch.tensor(ids).long(),torch.tensor(scoring)).sum()/64,loss)
    changed=logits.clone();changed[0,64,target[0,0]]+=2
    assert shifted_nll(changed,torch.tensor(ids).long(),torch.tensor(scoring)).sum()/64<loss


def test_native_newline_diagnostic_has_no_opt_label():
    values=np.full((2,64),7,dtype=np.int32);values[0,:32]=4
    d=smollm.probe_diagnostics(values,[4])
    assert d['newline_token_count']==32 and d['newline_token_share']==.25
    assert not any('50118' in k for k in d)


def test_absolute_watchdog_deadline_never_extends_budget():
    from model_collapse.cloud.supervisor import deadline
    r={'created_at':100.,'guard_hourly_usd':1.,'session_allowance_usd':1.,'absolute_deadline':1000.}
    assert deadline(r)==1000.
    r['absolute_deadline']=10000.
    assert deadline(r)==3700.


def test_historical_opt_profiles_and_frozen_sources_unchanged():
    for kind in ('full','smoke'):
        c=yaml.safe_load((PROJECT/f'comparisons/fineweb_completion64_mixtures/{kind}.yaml').read_text(encoding='utf-8'))
        completion.validate_config(c);assert len(completion.arm_specs(c))==2
    assert sha256(PROJECT/'model_collapse/completion_data.py')=='0d0594d48957ffca5db9e1d941761daba04bba2fd1ee46fc01809f0f27fee61d'
    assert sha256(PROJECT/'model_collapse/fresh_data.py')=='990e84dac5291762d214e58d685fbdf1a7be563642dff39405d6215642fff890'


class Tokenizer:
    bos_token_id=eos_token_id=pad_token_id=0
    def encode(self,text,add_special_tokens=False):return list(map(int,text.split()))
    def decode(self,ids,**kwargs):return ' '.join(map(str,ids))
    def __len__(self):return 64


def test_native_bank_retains_allocations_and_extends_without_leakage(tmp_path,monkeypatch):
    from model_collapse.science.completion_bank_v2 import prepare_bank
    tok=Tokenizer();source=tmp_path/'source';original=tmp_path/'original'
    def row(name,seed,count=128):
        ids=np.random.default_rng(seed).integers(1,64,count).tolist()
        return {'id':name,'url':'https://example.test/'+name,'text':tok.decode(ids),'used_source_tokens':count,'window_start':0,'genuine_document_end':True}
    def rows(path,values):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(x)+'\n' for x in values),encoding='utf-8')
    for split,start in [('validation',100),('test',1000)]:rows(source/f'fineweb_edu/{split}/documents.jsonl',[row(split+str(i),start+i) for i in range(256)])
    a,b,extra=row('original1',10,256),row('original2',20,512),row('extension1',30,256)
    rows(original/'cycle_01/documents.jsonl',[a]);rows(original/'cycle_02/documents.jsonl',[b])
    atomic_json(original/'manifest.json',{'source':'original mock allocation','files':{
        f'cycle_{i:02d}/documents.jsonl':sha256(original/f'cycle_{i:02d}/documents.jsonl') for i in (1,2)}})
    c=profile();c.update(full_cycles=2,full_train_pairs=4,secondary_corpora=['fineweb_edu'],source_opt_bank_manifest_sha256=sha256(original/'manifest.json'))
    out=tmp_path/'native'
    m=prepare_bank(out,c,tok,source=source,original=original,iterator=[b,extra],source_tokenizer=tok)
    audit=audit_bank(out,tok)
    assert audit['all_pairs_reconstructed'] and audit['disjoint_documents_and_passages']
    assert m['bos_token_id']==0 and m['collections']['cycle_01']['pairs']==4
    first=[json.loads(s)['id'] for s in (out/'cycle_01/documents.jsonl').read_text().splitlines()]
    second=[json.loads(s)['id'] for s in (out/'cycle_02/documents.jsonl').read_text().splitlines()]
    assert first==['original1','extension1'] and second==['original2']
    before=sha256(out/'manifest.json')
    assert prepare_bank(out,c,tok,source=source,original=original,iterator=[],source_tokenizer=tok)==m
    assert sha256(out/'manifest.json')==before
    import shutil
    from model_collapse import completion_data
    stored=tmp_path/'pinned_original';shutil.copytree(original,stored);shutil.rmtree(original)
    c['source_opt_bank_ref']={'immutable':'test-original'}
    def cold_restore(ref,destination):
        assert ref==c['source_opt_bank_ref'] and destination==original
        shutil.copytree(stored,destination)
    monkeypatch.setattr(completion_data,'restore_bank',cold_restore)
    assert prepare_bank(out,c,tok,source=source,original=original,iterator=[],source_tokenizer=tok)==m
    (original/'cycle_01/documents.jsonl').write_text('{}\n')
    with pytest.raises(RuntimeError):
        prepare_bank(out,c,tok,source=source,original=original,iterator=[],source_tokenizer=tok)


def test_full_backup_capacity_is_enforced_before_provision(monkeypatch,tmp_path):
    from model_collapse.workflow import smollm
    monkeypatch.setattr(smollm,'STATE',tmp_path)
    c=profile()
    class Api:
        size=200
        def request(self,method,path):return {'size':self.size,'dataCenter':'EU-RO-1'}
    api=Api()
    with pytest.raises(RuntimeError,match='200 GB'):smollm.resolve_backup(c,api)
    store={'backend':'runpod_s3','bucket':'smoltesting','region':'EU-RO-1','prefix':'smollm2'}
    atomic_json(tmp_path/'smollm2_backup_store.json',store)
    full=smollm.resolve_backup(c,api)
    assert full['backup_store']==store and protocol_fingerprint(full)==protocol_fingerprint(c)
    api.size=30
    with pytest.raises(RuntimeError,match='capacity'):smollm.resolve_backup(c,api)


@pytest.mark.parametrize('fail_at',[None,'mixed95'])
def test_seven_arm_coordinator_order_isolation_and_failure(tmp_path,monkeypatch,fail_at):
    from model_collapse import hub,supervisor,pipeline
    from model_collapse.analysis import completion as analysis
    c=profile();c['prepared_bank']={'run_id':'pinned-test-bank'}
    root=tmp_path/'parent';atomic_json(root/'config.json',c)
    atomic_json(root/'resources.json',{'code_sha256':'runtime','run_id':'parent','pod_id':'one-pod','volume_id':'one-volume'})
    baseline=root/'seed_42/gen_00';(baseline/'checkpoint').mkdir(parents=True)
    atomic_json(baseline/'metrics.json',{'checkpoint_files':{},'weights_sha256':'initial'})
    monkeypatch.setattr(completion,'restore_bank',lambda *_:{'bos_token_id':0,'newline_token_ids':[7]})
    monkeypatch.setattr(pipeline,'environment',lambda:{'simulated':True})
    monkeypatch.setattr(completion,'release_coordinator_gpu',lambda:None)
    monkeypatch.setattr(hub,'backup',lambda *_:None)
    monkeypatch.setattr(completion,'verify_completed_arm',lambda *_:{'pinned':True})
    monkeypatch.setattr(analysis,'report',lambda *a,**kw:None)
    calls=[]
    def worker(child,**kwargs):
        arm=read_json(child/'config.json');resources=read_json(child/'resources.json')
        calls.append(arm['arm_id'])
        assert resources['owner_completion_id']=='parent' and resources['pod_id']=='one-pod'
        assert arm['initialization']=='continued' and arm['model']==smollm.MODEL and arm['bos_token_id']==0
        assert not (child/'seed_42/gen_01/optimizer_state').exists()
        atomic_json(child/'status.json',{'state':'budget_paused' if arm['arm_id']==fail_at else 'complete'})
        atomic_json(child/'backup_snapshot.json',{'pinned':True})
    monkeypatch.setattr(supervisor,'run_worker',worker)
    completion.run(root)
    names=[x['id'] for x in smollm.ARMS]
    assert calls==(names if fail_at is None else names[:names.index(fail_at)+1])
    assert read_json(root/'completion.json',{}).get('complete',False)==(fail_at is None)
    paths=completion.child_roots(root,c)
    assert len(set(paths.values()))==7 and all(p.parent==root.parent for p in paths.values())


def test_pre_pod_storage_is_counted_without_inventing_compute(monkeypatch):
    from model_collapse.cloud.budget import accrued_cost
    from model_collapse.cloud import execution
    failed={'run_id':'rejected','charge_started_at':0.,'deleted_at':1800.,'hourly_usd':1.,'kind':'smoke','acceptance_campaign':'native_acceptance'}
    bank={'run_id':'native-bank','charge_started_at':0.,'hourly_usd':2.,'kind':'smoke'}
    d={'launches':[],'volumes':{'rejected-volume':failed,'native-bank-volume':bank}}
    assert accrued_cost(d,3600.,kind='smoke')==2.5
    assert accrued_cost(d,3600.,kind='full')==0.
    monkeypatch.setattr(execution,'now',lambda:3600.)
    assert execution.campaign_accrued(d,'native_acceptance')==.5


def test_explicit_graphql_transport_preserves_pinned_request():
    from model_collapse.cloud.runpod import Runpod
    api=Runpod(key='test-key');calls=[]
    payload={'name':'unique','image':'pinned','disk':20,'cloud':'SECURE',
        'gpu':{'id':'matching','count':1,'allowedCudaVersions':['13.2']},
        'dataCenterIds':['EU-RO-1'],'mounts':{'network':[{'volumeId':'owned','path':'/workspace'}]},
        'ports':['22/tcp'],'startSsh':True,'startJupyter':False}
    def request(method,path,body=None,**kw):
        calls.append((method,path,body,kw))
        return {'data':{'podFindAndDeployOnDemand':{'id':'new'}}} if kw else {'id':'new','status':'RUNNING'}
    api.request=request
    assert api.create_pod_graphql(payload)['id']=='new'
    inputs=calls[0][2]['variables']['input']
    assert inputs['dataCenterIds']==['EU-RO-1'] and inputs['networkVolumeId']=='owned'
    assert inputs['imageName']=='pinned' and inputs['allowedCudaVersions']==['13.2']
    assert inputs['containerDiskInGb']==20 and inputs['volumeInGb']==0 and inputs['gpuCount']==1
    assert len(calls)==1  # Never lose a known acquisition to an immediate REST read.
    api.request=lambda *a,**k:{'errors':[{'message':'no capacity'}]}
    with pytest.raises(RuntimeError,match='no capacity'):api.create_pod_graphql(payload)


def test_full_retention_forecast_covers_both_volumes():
    r=smollm.retention_forecast(profile(),200_000_000_000,2.)
    assert r['seven_day_combined_storage_usd']==pytest.approx(400*.07/730*168)
    assert r['retention_above_existing_reserve_usd']==pytest.approx(r['seven_day_combined_storage_usd']-2.)
    larger=smollm.retention_forecast(profile(),300_000_000_000,2.)
    assert larger['seven_day_backup_storage_usd']>r['seven_day_backup_storage_usd']
