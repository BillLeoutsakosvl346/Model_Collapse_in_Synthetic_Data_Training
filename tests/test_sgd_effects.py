import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from model_collapse.common import read_json, atomic_json, sha256
from model_collapse.science.engine import shifted_nll, weights_hash
from model_collapse.science.gradients import sequence_losses
from model_collapse.science.sgd_effects import SourceAccumulator, UpdateObserver, verify_seal, preserve_rng, prepare_replay


class TinyLM(torch.nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.embedding = torch.nn.Embedding(13, 5)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(5, 13)

    def forward(self, input_ids, attention_mask, use_cache=False):
        h = self.dropout(self.embedding(input_ids)) * attention_mask[..., None]
        return SimpleNamespace(logits=self.head(h.cumsum(1)))


@pytest.fixture
def case():
    torch.manual_seed(42)
    model = TinyLM()
    ids = np.array([[1, 2, 3, 4, 5], [1, 6, 7, 8, 9], [1, 5, 6, 7, 8]])
    attention = np.ones_like(ids)
    scoring = np.array([[0, 0, 0, 1, 1], [0, 0, 0, 1, 1], [0, 0, 0, 0, 1]])
    return model, ids, attention, scoring


def backward(model, arrays, labels, *, microbatch=2):
    ids, attention, scoring = arrays
    model.zero_grad(set_to_none=True)
    acc = SourceAccumulator(model, int(scoring[:, 1:].sum()))
    for offset in range(0, len(ids), microbatch):
        ix = slice(offset, offset + microbatch)
        tokens, mask = torch.tensor(ids[ix]), torch.tensor(scoring[ix])
        logits = model(tokens, torch.tensor(attention[ix])).logits
        losses = shifted_nll(logits, tokens, mask).sum(1)
        acc.backward(losses, mask[:, 1:].sum(1), np.asarray(labels)[ix])
    return acc


def sgd(model, lr=.001):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0, weight_decay=0,
                           foreach=False, fused=False)


def test_direct_token_weighted_source_gradients_and_authoritative_total(case):
    model, *arrays = case
    acc = backward(model, arrays, [0, 1, 1])
    actual = {n: p.grad.clone() for n, p in model.named_parameters()}
    source_real = {n: g.clone() for n, g in acc.real.items()}
    meta = acc.metrics()
    assert meta['total_scored_targets'] == 5
    assert meta['real_scored_targets'] == 2
    assert meta['synthetic_scored_targets'] == 3
    assert meta['gradient']['reconstruction_relative_error'] < 1e-6
    model.zero_grad()
    ids, attention, scoring = (torch.tensor(a) for a in arrays)
    losses = shifted_nll(model(ids, attention).logits, ids, scoring)
    (losses.sum()/5).backward()
    for n, p in model.named_parameters():
        torch.testing.assert_close(actual[n], p.grad, rtol=1e-6, atol=1e-7)
    model.zero_grad()
    (shifted_nll(model(ids[:1], attention[:1]).logits, ids[:1], scoring[:1]).sum()/5).backward()
    for n, p in model.named_parameters():
        torch.testing.assert_close(source_real[n], p.grad)


@pytest.mark.parametrize('label,source,absent',[(0,'real','synthetic'),(1,'synthetic','real')])
def test_endpoints_reuse_total_and_absent_source_is_undefined(case,label,source,absent):
    model,*arrays=case;acc=backward(model,arrays,[label]*3)
    meta=acc.metrics()
    assert meta[absent+'_source_mean_nll'] is None
    assert meta[absent+'_pairs']==0
    assert meta['gradient']['real_synthetic_cosine'] is None
    assert meta['gradient'][absent+'_norm']==0.
    assert meta['gradient'][source+'_norm']==pytest.approx(meta['gradient']['total_norm'])
    for n,p in model.named_parameters():torch.testing.assert_close(getattr(acc,source)[n],p.grad,rtol=0,atol=0)


def test_production_continuation_mask_and_16_example_tail():
    torch.manual_seed(42);model=TinyLM()
    ids=np.tile(np.arange(129)%13,(16,1));attention=np.ones_like(ids)
    scoring=np.zeros_like(ids);scoring[:,65:]=1
    acc=backward(model,(ids,attention,scoring),[0]*3+[1]*13,microbatch=7)
    meta=acc.metrics()
    assert meta['total_scored_targets']==1024
    assert meta['real_scored_targets']==192 and meta['synthetic_scored_targets']==832
    assert meta['gradient']['reconstruction_relative_error']<1e-6
    logits=model(torch.tensor(ids),torch.tensor(attention)).logits
    direct=-torch.log_softmax(logits[:,64:128].float(),dim=-1).gather(-1,torch.tensor(ids[:,65:])[...,None]).squeeze(-1)
    assert shifted_nll(logits,torch.tensor(ids),torch.tensor(scoring)).sum().item()==pytest.approx(direct.sum().item())


def test_observer_shadow_equality_weighting_and_source_effects(case,tmp_path):
    model,*arrays=case;model.train()
    observer=UpdateObserver(model,arrays,root=tmp_path,cycle=1,probe_identity='fixed-probe')
    acc=backward(model,arrays,[0,1,1]);original=copy.deepcopy(model.state_dict())
    old_gradients={n:p.grad.clone() for n,p in model.named_parameters()}
    record=observer.before_step(acc,.001,1,[10,11,12])
    assert all(torch.equal(p,original[n]) for n,p in model.state_dict().items())
    assert all(torch.equal(p.grad,old_gradients[n]) for n,p in model.named_parameters())
    actual=copy.deepcopy(model)
    with torch.no_grad():
        for n,p in actual.named_parameters():p.add_(acc.real[n],alpha=-.001)
    losses,counts=sequence_losses(actual,*arrays)
    assert record['states']['real']['loss']==pytest.approx(losses.sum()/counts.sum())
    assert record['effects']['real_conditional_delta_nll']==pytest.approx(
        record['states']['mixed']['loss']-record['states']['synthetic']['loss'])
    sgd(model).step();finished=observer.after_step(step_seconds=.25)
    assert finished['mixed_shadow_exact'] and finished['mixed_shadow_max_absolute_error']==0
    assert finished['step_seconds']>=.25 and finished['after_step_seconds']>=0
    assert model.training
    manifest=observer.seal(weights_hash(model))
    assert manifest['updates']==1
    verify_seal(tmp_path,expected_weights=weights_hash(model),expected_probe_identity='fixed-probe',expected_steps=[1])


def test_cache_and_endpoint_measurement_dedup(case,tmp_path):
    model,*arrays=case;observer=UpdateObserver(model,arrays,root=tmp_path,cycle=2,probe_identity='fixed')
    for step in [80,81]:
        acc=backward(model,arrays,[1]*3)
        before=observer.before_step(acc,.001,step,[0,1,2]);sgd(model).step();observer.after_step()
        assert before['effects']['real_only_delta_nll']==0.
        assert before['effects']['real_conditional_delta_nll']==0.
        assert before['effect_status']['real_conditional_delta_nll']=='exact_shared_measurement'
        assert not before['numerically_unresolved']['real_conditional_delta_nll']
    assert observer.records[0]['unique_probe_evaluations']==2
    assert observer.records[1]['unique_probe_evaluations']==1
    assert observer.records[1]['states']['no_update']['measurement_id']==observer.records[0]['states']['mixed']['measurement_id']
    observer.seal(weights_hash(model));verify_seal(tmp_path,expected_steps=[80,81])


def test_probe_cache_invalidated_after_external_weight_change(case,tmp_path):
    model,*arrays=case;observer=UpdateObserver(model,arrays,root=tmp_path,cycle=1,probe_identity='fixed')
    for step in [1,2]:
        if step==2:
            with torch.no_grad():next(model.parameters()).add_(.01)
        acc=backward(model,arrays,[1]*3)
        observer.before_step(acc,.001,step,[0,1,2]);sgd(model).step();observer.after_step()
    assert observer.records[1]['unique_probe_evaluations']==2


def test_rng_isolation_and_observation_does_not_change_training(tmp_path,case):
    _,*arrays=case;torch.manual_seed(42);model=TinyLM(.3);other=copy.deepcopy(model)
    initial_rng=torch.get_rng_state().clone()
    observer=UpdateObserver(model,arrays,root=tmp_path,cycle=1,probe_identity='fixed')
    assert torch.equal(torch.get_rng_state(),initial_rng)
    for step in [1,2]:
        acc=backward(model,arrays,[0,1,1]);rng=torch.get_rng_state().clone()
        observer.before_step(acc,.001,step,[0,1,2]);sgd(model).step();observer.after_step()
        assert torch.equal(rng,torch.get_rng_state())
    torch.set_rng_state(initial_rng)
    for _ in [1,2]:
        other.zero_grad(set_to_none=True)
        ids,attention,scoring=(torch.tensor(a) for a in arrays)
        # Match the authoritative microbatch accumulation exactly.
        for start in [0,2]:
            rows=slice(start,start+2)
            logits=other(ids[rows],attention[rows]).logits
            (shifted_nll(logits,ids[rows],scoring[rows]).sum()/5).backward()
        sgd(other).step()
    for p,q in zip(model.parameters(),other.parameters()):torch.testing.assert_close(p,q,rtol=1e-6,atol=2e-7)
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    baseline=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    with preserve_rng():random.random();np.random.random();torch.rand(3)
    assert random.getstate()==baseline[0]
    assert np.array_equal(np.random.get_state()[1],baseline[1][1])
    assert torch.equal(torch.get_rng_state(),baseline[2])


def test_interrupted_attempt_replay_seals_only_completed_attempt(case,tmp_path):
    original,*arrays=case
    first=UpdateObserver(original,arrays,root=tmp_path,cycle=1,probe_identity='fixed')
    start=copy.deepcopy(original)
    acc=backward(original,arrays,[0,1,1]);first.before_step(acc,.001,1,[0,1,2]);sgd(original).step();first.after_step()
    replay=UpdateObserver(start,arrays,root=tmp_path,cycle=1,probe_identity='fixed')
    assert replay.attempt_id!=first.attempt_id
    acc=backward(start,arrays,[0,1,1]);replay.before_step(acc,.001,1,[0,1,2]);sgd(start).step();replay.after_step()
    manifest=replay.seal(weights_hash(start))
    assert first.attempt.exists() and manifest['attempt_id']==replay.attempt_id
    assert verify_seal(tmp_path)['updates']==1
    with pytest.raises(RuntimeError):UpdateObserver(start,arrays,root=tmp_path,cycle=1,probe_identity='fixed')


@pytest.mark.parametrize('fault',['probe','weights','steps','file','mean'])
def test_corrupted_or_incompatible_seals_rejected(case,tmp_path,fault):
    model,*arrays=case;observer=UpdateObserver(model,arrays,root=tmp_path,cycle=1,probe_identity='fixed')
    acc=backward(model,arrays,[0,1,1]);observer.before_step(acc,.001,1,[0,1,2]);sgd(model).step();observer.after_step()
    observer.seal(weights_hash(model))
    if fault=='probe':
        with pytest.raises(RuntimeError):verify_seal(tmp_path,expected_probe_identity='changed')
    elif fault=='weights':
        with pytest.raises(RuntimeError):verify_seal(tmp_path,expected_weights='0'*64)
    elif fault=='steps':
        with pytest.raises(RuntimeError):verify_seal(tmp_path,expected_steps=[2])
    elif fault=='file':
        observer.journal.write_text('corrupted')
        with pytest.raises(RuntimeError):verify_seal(tmp_path)
    else:
        row=json.loads(observer.journal.read_text());row['states']['mixed']['loss']+=.1
        observer.journal.write_text(json.dumps(row)+'\n')
        manifest=read_json(tmp_path/'diagnostics_manifest.json');manifest['files'][manifest['records_path']]=sha256(observer.journal)
        atomic_json(tmp_path/'diagnostics_manifest.json',manifest)
        with pytest.raises(RuntimeError,match='per-example'):verify_seal(tmp_path)


def test_invalid_labels_counts_and_incomplete_accumulation(case):
    model,*arrays=case
    ids,attention,scoring=(torch.tensor(a) for a in arrays)
    losses=shifted_nll(model(ids,attention).logits,ids,scoring).sum(1)
    with pytest.raises(ValueError):SourceAccumulator(model,0)
    acc=SourceAccumulator(model,5)
    with pytest.raises(ValueError):acc.backward(losses,[2,2,1],[0,.5,1])
    with pytest.raises(ValueError):acc.backward(losses,[2,0,1],[0,1,1])
    with pytest.raises(ValueError):acc.metrics()


def test_shadow_rejects_non_sgd_actual_update(case,tmp_path):
    model,*arrays=case;observer=UpdateObserver(model,arrays,root=tmp_path,cycle=1,probe_identity='fixed')
    acc=backward(model,arrays,[0,1,1]);observer.before_step(acc,.001,1,[0,1,2])
    sgd(model,lr=.002).step()
    with pytest.raises(RuntimeError,match='ordinary SGD'):observer.after_step()


def test_uncommitted_seal_can_be_archived_only_with_matching_binding(case,tmp_path):
    model,*arrays=case;binding={'input':'frozen','contract':'sgd'}
    observer=UpdateObserver(model,arrays,root=tmp_path,cycle=1,probe_identity='fixed',binding=binding)
    acc=backward(model,arrays,[0,1,1]);observer.before_step(acc,.001,1,[0,1,2]);sgd(model).step();observer.after_step()
    manifest=observer.seal(weights_hash(model));original=(tmp_path/'diagnostics_manifest.json').read_bytes()
    with pytest.raises(RuntimeError,match='binding'):
        prepare_replay(tmp_path,cycle=1,probe_identity='fixed',binding={'input':'changed'},expected_steps=[1])
    assert (tmp_path/'diagnostics_manifest.json').read_bytes()==original
    with pytest.raises(RuntimeError,match='Completed diagnostics'):
        UpdateObserver(model,arrays,root=tmp_path,cycle=1,probe_identity='fixed',binding={'input':'changed'})
    result=prepare_replay(tmp_path,cycle=1,probe_identity='fixed',binding=binding,expected_steps=[1])
    assert result['attempt_id']==manifest['attempt_id']
    assert (tmp_path/result['archived_path']).read_bytes()==original
    assert not (tmp_path/'diagnostics_manifest.json').exists()
    with pytest.raises(RuntimeError,match='binding'):
        UpdateObserver(model,arrays,root=tmp_path,cycle=1,probe_identity='fixed',binding={'input':'changed'})
    UpdateObserver(model,arrays,root=tmp_path,cycle=1,probe_identity='fixed',binding=binding)


def test_completed_training_forbids_uncommitted_seal_archival(case,tmp_path):
    model,*arrays=case;observer=UpdateObserver(model,arrays,root=tmp_path,cycle=1,probe_identity='fixed')
    acc=backward(model,arrays,[0,1,1]);observer.before_step(acc,.001,1,[0,1,2]);sgd(model).step();observer.after_step()
    observer.seal(weights_hash(model));old=sha256(tmp_path/'diagnostics_manifest.json')
    atomic_json(tmp_path/'training.json',{'complete':True})
    with pytest.raises(RuntimeError,match='Completed training'):
        prepare_replay(tmp_path,cycle=1,probe_identity='fixed',binding={},expected_steps=[1])
    assert sha256(tmp_path/'diagnostics_manifest.json')==old
