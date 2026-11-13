import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from model_collapse.science.gradients import (mean_gradient, combine, geometry, relative_error,
    apply_gradient, optimizer_for, sequence_losses, stratified_indices, paired_change)


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(13, 5)
        self.head = torch.nn.Linear(5, 13)

    def forward(self, input_ids, attention_mask, use_cache=False):
        h = self.embedding(input_ids) * attention_mask[..., None]
        return SimpleNamespace(logits=self.head(h.cumsum(1)))


@pytest.fixture
def case():
    torch.manual_seed(42)
    model = TinyLM()
    ids = np.array([[1, 2, 3, 4, 5], [1, 6, 7, 8, 9], [1, 5, 6, 7, 8]], dtype=np.int64)
    attention = np.ones_like(ids)
    scoring = np.array([[0, 0, 0, 1, 1], [0, 0, 0, 1, 1], [0, 0, 0, 0, 1]])
    return model, ids, attention, scoring


def test_token_weighted_gradient_and_direct_loss(case):
    model, ids, att, scoring = case
    original = copy.deepcopy(model.state_dict())
    g, meta = mean_gradient(model, ids, att, scoring, microbatch=1)
    from model_collapse.science.engine import shifted_nll
    logits = model(torch.tensor(ids), torch.tensor(att)).logits
    loss = shifted_nll(logits, torch.tensor(ids), torch.tensor(scoring)).sum() / 5
    model.zero_grad(); loss.backward()
    for n, p in model.named_parameters():
        torch.testing.assert_close(g[n], p.grad)
    assert meta['scored_tokens'] == 5 and meta['loss'] == pytest.approx(loss.item())
    assert all(torch.equal(v, original[n]) for n, v in model.state_dict().items())


def test_component_reconstruction_uses_tokens_not_rows(case):
    model, ids, att, mask = case
    left, _ = mean_gradient(model, ids, att, mask, rows=[0])
    right, _ = mean_gradient(model, ids, att, mask, rows=[1, 2])
    direct, _ = mean_gradient(model, ids, att, mask)
    assert relative_error(direct, combine(left, right, 2/5)) < 1e-6
    assert relative_error(direct, combine(left, right, 1/3)) > 1e-3


def test_prompt_labels_masked_but_prompt_visible(case):
    model, ids, att, mask = case
    loss, counts = sequence_losses(model, ids, att, mask)
    altered = ids.copy(); altered[:, 1] = 12
    changed, _ = sequence_losses(model, altered, att, mask)
    assert counts.tolist() == [2, 2, 1]
    assert not np.allclose(changed, loss)


def test_identical_targets_give_identical_intervention(case):
    model, ids, att, mask = case
    g, _ = mean_gradient(model, ids, att, mask)
    a, b = copy.deepcopy(model), copy.deepcopy(model)
    apply_gradient(a, g, None, 3e-6)
    apply_gradient(b, combine(g, g, .01), None, 3e-6)
    for x, y in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(x, y, rtol=0, atol=1e-7)


def test_optimizer_history_explicit_zero_and_copy_isolation(case):
    model, ids, att, mask = case
    g, _ = mean_gradient(model, ids, att, mask)
    optimizer = apply_gradient(model, g, None, 3e-6)
    saved = {'optimizer': copy.deepcopy(optimizer.state_dict())}
    snapshot = copy.deepcopy(saved)
    a, b = copy.deepcopy(model), copy.deepcopy(model)
    apply_gradient(a, None, saved, 3e-6)
    apply_gradient(b, g, saved, 3e-6)
    assert any(not torch.equal(x, y) for x, y in zip(a.parameters(), model.parameters()))
    for k in saved['optimizer']['state']:
        for name in ['step', 'exp_avg', 'exp_avg_sq']:
            assert torch.equal(saved['optimizer']['state'][k][name], snapshot['optimizer']['state'][k][name])
    assert all(p.grad is None for p in model.parameters())


def test_geometry_alignment_and_zero():
    a = {'x': torch.tensor([1., 0.])}
    assert geometry(a, {'x': -a['x']})['cosine'] == -1
    assert geometry(a, {'x': torch.zeros(2)})['cosine'] is None


def test_invalid_mask_and_rows_rejected(case):
    model, ids, att, mask = case
    bad = mask.copy(); bad[0, 0] = 2
    with pytest.raises(ValueError): mean_gradient(model, ids, att, bad)
    with pytest.raises(ValueError): mean_gradient(model, ids, att, mask, rows=[0, 0])


def test_document_probe_and_paired_intervals():
    windows = np.array([[d, j] for d in range(12) for j in range(d + 1)])
    ix = stratified_indices(windows, 16)
    assert np.array_equal(ix, stratified_indices(windows, 16))
    assert len(set(windows[ix[:12], 0])) == 12
    before = np.ones(16); after = before - .01; counts = np.ones(16)
    r = paired_change(before, after, counts, windows[ix, 0])
    assert r['delta_nll'] == pytest.approx(-.01)
    assert r['delta_nll_document_bootstrap_95'] == pytest.approx([-.01, -.01])


def test_source_input_reconstruction_and_corruption():
    from model_collapse.artifacts.gradient_sources import check_input
    from model_collapse.completion_data import assemble
    from model_collapse.science.identity import token_hash
    p, r, g = (np.full((4, 64), n, dtype=np.int32) for n in (1, 2, 3))
    targets = r.copy(); targets[[1, 3]] = g[:2]
    arrays = assemble(p, targets, 0)
    manifest = {'lineage': {'generator_weights_sha256':'w','initializer_weights_sha256':'w',
                           'bank_manifest_sha256':'b','chunk_manifest_sha256':'c'},
                'prompt_content_sha256':token_hash(p),'target_content_sha256':token_hash(targets),'scored_targets':256}
    check_input(*arrays, p, r, g[:2], [1, 3], manifest, 'w', 'b', 'c')
    with pytest.raises(RuntimeError):
        check_input(*arrays, p, r, g[:2], [0, 3], manifest, 'w', 'b', 'c')


def test_cli_and_workflow_are_lazy():
    from model_collapse.workflow.registry import known_workflow, handler
    assert known_workflow('gradient_diagnostic')
    assert callable(handler({'workflow':'gradient_diagnostic'}, 'report'))


def test_diagnostic_configuration_rejects_changed_source_and_protocol():
    from model_collapse.common import config
    from model_collapse.paths import PROJECT
    from model_collapse.workflow.gradient_diagnostic import validate_config
    c = config(PROJECT/'comparisons/smollm2_gradient_diagnostic/diagnostic.yaml')
    for change in [{'cycles':[1,4,10]}, {'common_learning_rate':1e-4},
                   {'arms':['mixed50','synthetic']}, {'precision':'bf16'}, {'max_compute_seconds':7200}]:
        with pytest.raises(ValueError): validate_config({**c, **change})


def test_restoration_rejects_changed_lineage_and_masks():
    from model_collapse.artifacts.gradient_sources import check_input
    from model_collapse.completion_data import assemble
    from model_collapse.science.identity import token_hash
    p, r = (np.full((4,64),i,dtype=np.int32) for i in (1,2))
    arrays = assemble(p,r,0)
    manifest={'lineage':{'generator_weights_sha256':'w','initializer_weights_sha256':'w',
                        'bank_manifest_sha256':'b','chunk_manifest_sha256':'c'},
              'prompt_content_sha256':token_hash(p),'target_content_sha256':token_hash(r),'scored_targets':256}
    with pytest.raises(RuntimeError): check_input(*arrays,p,r,r,np.arange(4),manifest,'wrong','b','c')
    corrupted=arrays[2].copy();corrupted[:,64]=1
    with pytest.raises(RuntimeError): check_input(arrays[0],arrays[1],corrupted,p,r,r,np.arange(4),manifest,'w','b','c')


def test_setup_attribution_is_not_double_charged():
    from model_collapse.cloud.budget import accrued_cost
    d={'launches':[{'run_id':'diagnostic','kind':'diagnostic','created_at':0.,'compute_started_at':0.,
                   'compute_stopped_at':3600.,'compute_hourly_usd':.6,'validation_estimated_usd':.05}],
       'volumes':{}}
    assert accrued_cost(d,7200)==pytest.approx(.6)
    assert accrued_cost(d,7200,'smoke')==pytest.approx(.05)


def test_layer_supplement_retains_same_scientific_cases():
    from model_collapse.common import config
    from model_collapse.paths import PROJECT
    from model_collapse.workflow.gradient_diagnostic import validate_config
    c=config(PROJECT/'comparisons/smollm2_gradient_diagnostic/diagnostic.yaml')
    validate_config({**c,'phase':'layer_norms','max_compute_seconds':5400})
    with pytest.raises(ValueError):validate_config({**c,'phase':'layer_norms','max_compute_seconds':7200})


def test_heldout_directional_derivative_without_heldout_update(case):
    model,ids,att,mask=case
    training,_=mean_gradient(model,ids,att,mask,rows=[0,1])
    heldout,_=mean_gradient(model,ids,att,mask,rows=[2])
    before,counts=sequence_losses(model,ids[2:],att[2:],mask[2:])
    original={n:p.detach().clone() for n,p in model.named_parameters()}
    apply_gradient(model,training,None,1e-6)
    predicted=sum(float(((p.detach()-original[n]).double()*heldout[n].double()).sum()) for n,p in model.named_parameters())
    after,_=sequence_losses(model,ids[2:],att[2:],mask[2:])
    observed=float((after-before).sum()/counts.sum())
    assert abs(observed-predicted)<1e-6


def test_per_parameter_norms_reconstruct_global_norm():
    g={'a':torch.tensor([3.,4.]),'b':torch.tensor([0.,12.])}
    value=geometry(g)
    assert value['norm']==13.
    assert sum(n*n for n in value['parameters'].values())==pytest.approx(value['norm']**2)


@pytest.mark.parametrize('cycle',[1,5])
def test_layer_followup_end_to_end_without_cloud(tmp_path,monkeypatch,cycle):
    import importlib.metadata
    from model_collapse.science import engine
    from model_collapse.completion_data import assemble
    from model_collapse.common import read_json,atomic_json
    from model_collapse.workflow.gradient_layers import run_layers
    import model_collapse.artifacts.gradient_sources as sources
    import model_collapse.workflow.gradient_diagnostic as diagnostic
    import model_collapse.hub as hub
    torch.manual_seed(42); model=TinyLM(); rng=np.random.default_rng(42)
    prompt=rng.integers(1,13,(16,64)); real=rng.integers(1,13,(16,64))
    target=real.copy();target[8:]=rng.integers(1,13,(8,64));arrays=assemble(prompt,target,0)
    val=assemble(prompt,real,0);syn=np.arange(8,16)
    parent=None
    if cycle>1:
        initial,_=mean_gradient(model,*arrays)
        optimizer=apply_gradient(model,initial,None,5e-6)
        for item in optimizer.state.values():item['step'].fill_(316)
        parent={'optimizer':copy.deepcopy(optimizer.state_dict()),'model_weights_sha256':engine.weights_hash(model),'global_step':316,'total_steps':790}
        torch.save(parent,tmp_path/'parent.pt')
    gs,sm=mean_gradient(model,*arrays,rows=syn)
    gr,_=mean_gradient(model,*arrays,rows=np.arange(8))
    before,counts=sequence_losses(model,*val)
    updates=[]
    for action,g in [('synthetic_only',gs),('mixed',combine(gr,gs,.01))]:
        clone=copy.deepcopy(model);apply_gradient(clone,g,parent,5e-6)
        after,_=sequence_losses(clone,*val)
        updates.append({'action':action,'learning_rate_policy':'historical','delta_nll':float((after-before).sum()/counts.sum()),'weights_after_sha256':engine.weights_hash(clone)})
    prior={'real_vs_synthetic':geometry(gr,gs),'matched_real_vs_synthetic':geometry(gr,gs),'synthetic_source':sm}
    c={'source_parent':{'run_id':'source'},'main_measurements':{'run_id':'main'},'microbatch':8}
    class FakeSnapshot:
        def __init__(self,*args):pass
        def json(self,path):
            if path=='config.json':return {'source_parent':c['source_parent']}
            if path=='completion.json':return {'complete':True}
            if path.endswith('gradients.json'):return prior
            if path.endswith('interventions.json'):return updates
            raise AssertionError(path)
        def array(self,path):return np.arange(16)
        def checkpoint(self,prefix):return tmp_path/'checkpoint'
        def get(self,path):return tmp_path/'parent.pt'
    class FakeBank:
        def array(self,path):return prompt if path.endswith('prompts.npy') else real
    case={'arm':'mixed99','cycle':cycle,'weights':engine.weights_hash(model),'tokens':arrays[0],
          'attention':arrays[1],'scoring':arrays[2],'synthetic_rows':syn,'child':FakeSnapshot()}
    scientific={'required_environment':{n:importlib.metadata.version(n) for n in ('torch','transformers','datasets')},
                'bos_token_id':0,'learning_rate':5e-6,'weight_decay':.01}
    monkeypatch.setattr(sources,'Snapshot',FakeSnapshot)
    monkeypatch.setattr(sources,'restore_sources',lambda *a,**k:([case],FakeBank(),scientific))
    monkeypatch.setattr(engine,'load_model',lambda *a,**k:copy.deepcopy(model))
    monkeypatch.setattr(diagnostic,'_accept',lambda m,case,val,root,r:atomic_json(root/'acceptance.json',{'test_only':True}))
    monkeypatch.setattr(hub,'backup',lambda *a,**k:None)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'test CPU')
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    import time
    resources={'created_at':time.time(),'session_allowance_usd':10.,'guard_hourly_usd':1.,'absolute_deadline':time.time()+3600}
    run_layers(tmp_path,c,resources)
    result=read_json(tmp_path/f'layers/mixed99-cycle{cycle:02d}.json')
    assert result['validation_gradient_applied_to_weights'] is False
    assert result['global_norm_relative_error']<1e-7
    assert len(result['directional_interventions'])==(2 if cycle==1 else 3)
    assert all(r['primary_update_tensor_hash_matches'] for r in result['directional_interventions'] if r['action']!='synthetic_without_momentum')
    if parent:
        unchanged=torch.load(tmp_path/'parent.pt',weights_only=False)
        for key in parent['optimizer']['state']:
            assert torch.equal(parent['optimizer']['state'][key]['exp_avg'],unchanged['optimizer']['state'][key]['exp_avg'])
