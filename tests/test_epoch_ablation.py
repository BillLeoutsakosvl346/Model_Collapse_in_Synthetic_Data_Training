import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from model_collapse.common import read_json, sha256
from model_collapse.science import engine
from model_collapse.science.epoch_ablation import SourceStep, EpochObserver, validate_training_source
from model_collapse.science.sgd_effects import SourceAccumulator
from model_collapse.workflow.epoch_ablation import batch_selection, validate_config


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.emb=torch.nn.Embedding(13,5);self.head=torch.nn.Linear(5,13)
    def forward(self,input_ids,attention_mask,use_cache=False):
        return SimpleNamespace(logits=self.head((self.emb(input_ids)*attention_mask[...,None]).cumsum(1)))


def gradients(model, source, labels, denominator=12, old=False):
    ids=torch.tensor([[1,2,3,4,5],[1,3,4,5,6],[1,7,8,9,10]])
    mask=torch.tensor([[0,0,0,1,1]]*3)
    model.zero_grad(set_to_none=True)
    acc=SourceAccumulator(model,6) if old else SourceStep(model,denominator,source)
    for start in (0,2):
        ix=slice(start,start+2)
        sums=engine.shifted_nll(model(ids[ix],torch.ones_like(ids[ix])).logits,ids[ix],mask[ix]).sum(1)
        acc.backward(sums,mask[ix,1:].sum(1),np.asarray(labels)[ix])
    return [p.grad.clone() for p in model.parameters()]


def test_source_gradients_reconstruct_without_renormalizing():
    torch.manual_seed(42);base=Tiny()
    both=gradients(copy.deepcopy(base),'both',[0,1,1])
    real=gradients(copy.deepcopy(base),'real',[0,1,1])
    synthetic=gradients(copy.deepcopy(base),'synthetic',[0,1,1])
    for a,b,c in zip(both,real,synthetic):torch.testing.assert_close(a,b+c,atol=2e-7,rtol=2e-6)
    legacy=gradients(copy.deepcopy(base),'both',[0,1,1],old=True)
    for a,b in zip(both,legacy):torch.testing.assert_close(a*2,b,atol=0,rtol=0)


def test_empty_real_source_has_exactly_zero_gradient_and_update():
    base=Tiny();before=[p.clone() for p in base.parameters()]
    gs=gradients(base,'real',[1,1,1])
    opt=torch.optim.SGD(base.parameters(),lr=.01,momentum=0,weight_decay=0,foreach=False,fused=False)
    opt.step()
    assert all(torch.count_nonzero(x)==0 for x in gs)
    assert all(torch.equal(a,b) for a,b in zip(before,base.parameters()))


def test_tail_denominator_preserves_eightfold_weight():
    base=Tiny()
    full=gradients(copy.deepcopy(base),'real',[0,1,1],denominator=128*64)
    tail=gradients(copy.deepcopy(base),'real',[0,1,1],denominator=16*64)
    for a,b in zip(full,tail):torch.testing.assert_close(a*8,b,atol=0,rtol=0)


def test_actual_original_acceptance_batches_and_order():
    from model_collapse.completion_mixing import select_rows
    case={'cycle':6,'labels':select_rows(10000,.99,42,6)[0]['source_labels']}
    rows,order=batch_selection(case,True)
    assert len(rows)==272 and sum(case['labels'][rows]==0)==6
    assert np.array_equal(order,np.arange(272))
    full,order=batch_selection(case)
    assert np.array_equal(full,np.arange(10000))
    assert np.array_equal(order,np.random.default_rng(48).permutation(10000))


@pytest.mark.parametrize('field,value',[('cycles',[1,5,10]),('required_gpu','NVIDIA A40'),('replay_tolerance',.01),('validation_cap_usd',1),('source_arm','synthetic')])
def test_protocol_rejects_changed_settings(field,value):
    from model_collapse.common import PROJECT,config
    c=config(PROJECT/'comparisons/smollm2_sgd_epoch_ablation/diagnostic.yaml')
    c[field]=value
    with pytest.raises(ValueError):validate_config(c)


def test_observer_preserves_rng_mode_and_gradients(tmp_path,monkeypatch):
    monkeypatch.setattr(engine,'device',lambda:'cpu')
    base=Tiny().train();tokens=np.array([[1,2,3,4,5],[1,3,4,5,6]])
    att=np.ones_like(tokens);score=np.array([[0,0,0,1,1]]*2)
    probe={'tokens':tokens,'attention':att,'scoring':score,'documents':np.arange(2),'identity':'fixed'}
    c={'eval_batch':1,'precision':'fp32','effective_batch':2}
    obs=EpochObserver(tmp_path,c,probe,{'source':'real'},np.arange(2))
    before=torch.get_rng_state().clone();obs.start(base,10)
    assert base.training and torch.equal(before,torch.get_rng_state())
    assert all(p.grad is None for p in base.parameters())
    obs.after_step(base,step=11,batch_indices=np.array([0,1]),gradient_norm=0.,learning_rate=.01,seconds=.1,
                   denominator=4,real_pairs=0,synthetic_pairs=2,real_weighted_loss=0.,synthetic_weighted_loss=1.)
    obs.seal(engine.weights_hash(base));obs.verify(engine.weights_hash(base))
    with (obs.attempt/'updates.jsonl').open('a') as f:f.write('corruption')
    with pytest.raises(RuntimeError):obs.verify(engine.weights_hash(base))


def test_shared_engine_branch_replay_and_completed_integrity(tmp_path,monkeypatch):
    from transformers import GPT2Config,GPT2LMHeadModel
    torch.set_num_threads(1);engine.seed_all(42)
    base=tmp_path/'base';model=GPT2LMHeadModel(GPT2Config(vocab_size=17,n_positions=129,n_embd=16,n_layer=1,n_head=2,
                resid_pdrop=0,embd_pdrop=0,attn_pdrop=0));model.save_pretrained(base);initial=engine.weights_hash(model)
    def load(c,checkpoint=None):
        engine.seed_all(42);return GPT2LMHeadModel.from_pretrained(checkpoint or base).cpu()
    monkeypatch.setattr(engine,'load_model',load);monkeypatch.setattr(engine,'device',lambda:'cpu')
    tokens=np.random.default_rng(42).integers(0,17,(144,129),dtype=np.int32)
    att=np.ones_like(tokens);mask=np.zeros_like(tokens);mask[:,65:]=1
    labels=np.ones(144,dtype=np.uint8);labels[[0,139]]=0
    probe={'tokens':tokens[:4],'attention':att[:4],'scoring':mask[:4],'documents':np.arange(4),'identity':'pinned'}
    c=dict(workflow='completion_arm',optimizer='sgd',learning_rate_policy='constant',momentum=0,
           learning_rate=.01,min_learning_rate=.01,weight_decay=0,precision='fp32',epochs=1,seed=42,
           microbatch=32,effective_batch=128,eval_batch=4,checkpoint_selection='final_epoch',initialization='continued',
           prepared_bank={'manifest_sha256':'a'*64},data_kind='mixed',synthetic_fraction=.99,generation_decoder='greedy',
           num_beams=1,do_sample=False,block_size=64,new_tokens=64,repetition_penalty=1.,update_diagnostics={'enabled':False},
           epoch_ablation={'source':'both'})
    kwargs=dict(initial_checkpoint=base,masks=att,val_masks=att[:4],loss_masks=mask,val_loss_masks=mask[:4],
                total_steps=790,source_labels=labels,training_source='both',row_order=np.arange(144))
    stage=tmp_path/'new/seed_42/gen_03'
    obs=lambda:EpochObserver(stage,c,probe,{'source':'both'},np.arange(144))
    first=engine.train(tmp_path/'new',3,tokens,tokens[:4],c,initial,step_observer=obs(),**kwargs)
    checksum=sha256(stage/'epoch_diagnostics_manifest.json')
    again=engine.train(tmp_path/'new',3,tokens,tokens[:4],c,initial,step_observer=obs(),**kwargs)
    assert again==first and sha256(stage/'epoch_diagnostics_manifest.json')==checksum
    oldc={**c,'epoch_ablation':None,'update_diagnostics':{'enabled':True}}
    legacy=engine.train(tmp_path/'old',3,tokens,tokens[:4],oldc,initial,
        initial_checkpoint=base,masks=att,val_masks=att[:4],loss_masks=mask,val_loss_masks=mask[:4],total_steps=790,
        source_labels=labels,update_probe=probe)
    # Historical order differs unless explicitly replayed in the branch.
    kwargs['row_order']=np.random.default_rng(45).permutation(144)
    stage2=tmp_path/'matched/seed_42/gen_03'
    matched=engine.train(tmp_path/'matched',3,tokens,tokens[:4],c,initial,
        step_observer=EpochObserver(stage2,c,probe,{'source':'both'},np.arange(144)),**kwargs)
    assert matched['selected_weights_sha256']==legacy['selected_weights_sha256']
    replay_stage=tmp_path/'interrupted/seed_42/gen_03'
    interrupted=EpochObserver(replay_stage,c,probe,{'source':'both'},np.arange(144))
    original_after=interrupted.after_step
    def exit_after_update(*args,**kw):
        original_after(*args,**kw)
        raise RuntimeError('injected branch exit')
    interrupted.after_step=exit_after_update
    with pytest.raises(RuntimeError,match='injected branch exit'):
        engine.train(tmp_path/'interrupted',3,tokens,tokens[:4],c,initial,step_observer=interrupted,**kwargs)
    old_journal=interrupted.attempt/'updates.jsonl';old_bytes=old_journal.read_bytes()
    replay=engine.train(tmp_path/'interrupted',3,tokens,tokens[:4],c,initial,
        step_observer=EpochObserver(replay_stage,c,probe,{'source':'both'},np.arange(144)),**kwargs)
    assert replay['selected_weights_sha256']==matched['selected_weights_sha256']
    assert old_journal.read_bytes()==old_bytes
    changed=labels.copy();changed[0]=1
    with pytest.raises(RuntimeError,match='provenance'):
        engine.train(tmp_path/'new',3,tokens,tokens[:4],c,initial,step_observer=obs(),**{**kwargs,'source_labels':changed})
