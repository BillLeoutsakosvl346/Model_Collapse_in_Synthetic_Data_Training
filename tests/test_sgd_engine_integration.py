"""Exercise the actual shared engine with a tiny locally initialized transformer."""
import json
import numpy as np
import pytest
import torch

from model_collapse.common import read_json, sha256
from model_collapse.science import engine
from model_collapse.science.sgd_effects import verify_seal


def test_actual_engine_sgd_continuity_and_cycle_replay(tmp_path, monkeypatch):
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.set_num_threads(1)
    engine.seed_all(42)
    base = tmp_path / "pretrained"
    config = GPT2Config(vocab_size=17, n_positions=129, n_embd=16, n_layer=1, n_head=2,
                        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
    initial = GPT2LMHeadModel(config); initial.save_pretrained(base)
    expected = engine.weights_hash(initial)
    def load(c, checkpoint=None):
        engine.seed_all(c["seed"])
        return GPT2LMHeadModel.from_pretrained(checkpoint or base).cpu()
    monkeypatch.setattr(engine, "load_model", load)
    monkeypatch.setattr(engine, "device", lambda: "cpu")
    c = dict(workflow="completion_arm", optimizer="sgd", learning_rate_policy="constant", momentum=0,
             learning_rate=.01, min_learning_rate=.01, weight_decay=0, precision="fp32", epochs=1,
             seed=42, microbatch=32, effective_batch=128, eval_batch=4, checkpoint_selection="final_epoch",
             initialization="continued", prepared_bank={"manifest_sha256":"a"*64}, data_kind="mixed",
             synthetic_fraction=.99, generation_decoder="greedy", num_beams=1, do_sample=False,
             block_size=64, new_tokens=64, repetition_penalty=1., update_diagnostics={"enabled":True})
    tokens=np.random.default_rng(3).integers(0,17,(144,129),dtype=np.int32)
    attention=np.ones_like(tokens);scoring=np.zeros_like(tokens);scoring[:,65:]=1
    sources=np.ones(144,dtype=np.uint8);sources[[2,33,139]]=0
    probe=dict(tokens=tokens[:4].copy(),attention=attention[:4].copy(),scoring=scoring[:4].copy(),
               documents=np.arange(4),identity="frozen-tiny-probe")
    kwargs=dict(masks=attention,val_masks=attention[:4],loss_masks=scoring,val_loss_masks=scoring[:4],
                total_steps=4,source_labels=sources,update_probe=probe)
    first=engine.train(tmp_path,1,tokens,tokens[:4],c,expected,initial_checkpoint=base,**kwargs)
    stage1=tmp_path/'seed_42/gen_01';seal=verify_seal(stage1,expected_steps=[1,2])
    old=sha256(stage1/'diagnostics_manifest.json')
    assert first['global_step']==2 and first['selected_weights_sha256']!=expected
    state=torch.load(stage1/'optimizer_state/cycle.pt',weights_only=False)
    assert state['optimizer']['state']=={} and state['scheduler'] is None
    args=dict(initial_checkpoint=stage1/'checkpoint',optimizer_checkpoint=stage1/'optimizer_state/cycle.pt',**kwargs)
    with pytest.raises(engine.ValidationInterruption):
        engine.train(tmp_path,2,tokens,tokens[:4],c,first['selected_weights_sha256'],inject=True,**args)
    last=engine.train(tmp_path,2,tokens,tokens[:4],c,first['selected_weights_sha256'],inject=True,**args)
    assert last['global_step']==4 and sha256(stage1/'diagnostics_manifest.json')==old
    verify_seal(tmp_path/'seed_42/gen_02',expected_steps=[3,4])
    with pytest.raises(RuntimeError,match='provenance changed'):
        changed=sources.copy();changed[0]=1-changed[0]
        engine.train(tmp_path,1,tokens,tokens[:4],c,expected,initial_checkpoint=base,**{**kwargs,'source_labels':changed})


def test_engine_replays_exit_after_diagnostic_seal_before_training_commit(tmp_path, monkeypatch):
    """A sealed observer is still uncommitted until training.json is written."""
    from pathlib import Path
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.set_num_threads(1);engine.seed_all(42)
    base=tmp_path/'pretrained'
    initial=GPT2LMHeadModel(GPT2Config(vocab_size=17,n_positions=129,n_embd=16,n_layer=1,n_head=2,
                                     resid_pdrop=0,embd_pdrop=0,attn_pdrop=0))
    initial.save_pretrained(base);expected=engine.weights_hash(initial)
    def load(c,checkpoint=None):
        engine.seed_all(c['seed'])
        return GPT2LMHeadModel.from_pretrained(checkpoint or base).cpu()
    monkeypatch.setattr(engine,'load_model',load);monkeypatch.setattr(engine,'device',lambda:'cpu')
    c=dict(workflow='completion_arm',optimizer='sgd',learning_rate_policy='constant',momentum=0,
           learning_rate=.01,min_learning_rate=.01,weight_decay=0,precision='fp32',epochs=1,
           seed=42,microbatch=32,effective_batch=128,eval_batch=4,checkpoint_selection='final_epoch',
           initialization='continued',prepared_bank={'manifest_sha256':'a'*64},data_kind='mixed',
           synthetic_fraction=.99,generation_decoder='greedy',num_beams=1,do_sample=False,
           block_size=64,new_tokens=64,repetition_penalty=1.,update_diagnostics={'enabled':True})
    tokens=np.random.default_rng(3).integers(0,17,(144,129),dtype=np.int32)
    attention=np.ones_like(tokens);scoring=np.zeros_like(tokens);scoring[:,65:]=1
    sources=np.ones(144,dtype=np.uint8);sources[[2,33,139]]=0
    probe=dict(tokens=tokens[:4].copy(),attention=attention[:4].copy(),scoring=scoring[:4].copy(),
               documents=np.arange(4),identity='frozen-tiny-probe')
    kwargs=dict(initial_checkpoint=base,masks=attention,val_masks=attention[:4],loss_masks=scoring,
                val_loss_masks=scoring[:4],total_steps=2,source_labels=sources,update_probe=probe)
    atomic=engine.atomic_json
    def exit_at_commit(path,value):
        if Path(path).name=='training.json':raise engine.ValidationInterruption('after seal, before training commit')
        atomic(path,value)
    monkeypatch.setattr(engine,'atomic_json',exit_at_commit)
    with pytest.raises(engine.ValidationInterruption,match='after seal'):
        engine.train(tmp_path,1,tokens,tokens[:4],c,expected,**kwargs)
    stage=tmp_path/'seed_42/gen_01'
    original=verify_seal(stage,expected_steps=[1,2]);sealed_bytes=(stage/'diagnostics_manifest.json').read_bytes()
    assert not (stage/'training.json').exists()
    monkeypatch.setattr(engine,'atomic_json',atomic)
    changed=sources.copy();changed[0]=1-changed[0]
    with pytest.raises(RuntimeError,match='binding'):
        engine.train(tmp_path,1,tokens,tokens[:4],c,expected,**{**kwargs,'source_labels':changed})
    assert (stage/'diagnostics_manifest.json').read_bytes()==sealed_bytes
    result=engine.train(tmp_path,1,tokens,tokens[:4],c,expected,**kwargs)
    recovered=verify_seal(stage,expected_steps=[1,2])
    archived=stage/'update_diagnostics/attempts'/original['attempt_id']/'uncommitted_seal.json'
    assert archived.read_bytes()==sealed_bytes
    assert recovered['attempt_id']!=original['attempt_id']
    assert result['selected_weights_sha256']==original['terminal_weights_sha256']
    assert result['global_step']==2
    final_hash=sha256(stage/'diagnostics_manifest.json')
    engine.train(tmp_path,1,tokens,tokens[:4],c,expected,**kwargs)
    assert sha256(stage/'diagnostics_manifest.json')==final_hash
    events=[json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    archived_events=[e for e in events if e['event']=='sgd_uncommitted_seal_archived']
    assert len(archived_events)==1 and archived_events[0]['attempt_id']==original['attempt_id']
