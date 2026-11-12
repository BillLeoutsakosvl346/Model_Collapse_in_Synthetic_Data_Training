"""Offline SmolLM2 fraction analysis; never trains, provisions or reads credentials."""
from __future__ import annotations
import argparse,csv,hashlib,io,json,math,platform,subprocess
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from model_collapse.analysis.completion import _collect,comparison_contract
from model_collapse.common import read_json,atomic_json,sha256
from model_collapse.completion_mixing import select_rows
from model_collapse.completion_data import assemble
from model_collapse.science.metrics import diversity
from model_collapse.workflow.smollm import probe_diagnostics
from scripts.analysis.publication import new_destination

RID='20261111T210854Z-full-completion64-mixtures'
ARMS=['real','mixed50','mixed80','mixed95','mixed99','mixed99p9','synthetic']
LABELS=dict(zip(ARMS,['0%','50%','80%','95%','99%','99.9%','100%']))
COLORS=dict(zip(ARMS,['#167C80','#3265A8','#6550A0','#B08313','#D06C24','#D34256','#77213C']))
def th(a):return hashlib.sha256(np.asarray(a,dtype=np.int32).tobytes()).hexdigest()
def nh(a):
 # The shared save_array contract stores all token/selection arrays as int32.
 b=io.BytesIO();np.save(b,np.asarray(a,dtype=np.int32));return hashlib.sha256(b.getvalue()).hexdigest()
def near(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-9),(a,b)

def audit(records,bank=None,bulk=None):
 records=Path(records);root=records/RID;c=read_json(root/'config.json');children=read_json(root/'children.json')
 assert list(children)==ARMS and read_json(root/'completion.json')['unique_training_stages']==70
 baseline=read_json(root/'seed_42/gen_00/metrics.json');checks=[];probes=[];unique=set();target_stats=[];seals=0
 for arm,rid in children.items():
  child=records/rid;cc=read_json(child/'config.json');index=read_json(child/'artifact_index.json')['files'];snap=read_json(child/'recovery_snapshot.json')
  assert sha256(child/'artifact_index.json')==snap['artifact_index_sha256'] and all(len(x)==40 for x in snap['revisions'].values())
  assert read_json(child/'completion.json')['cycles']==10
  for p in child.rglob('*.json'):
   rel=p.relative_to(child).as_posix()
   if rel in index:assert sha256(p)==index[rel]['sha256'],str(p)
  previous=baseline['weights_sha256'];optimizer=None
  for cycle in range(11):
   stage=child/f'seed_42/gen_{cycle:02d}';m=read_json(stage/'metrics.json')
   seq=read_json((stage if cycle else root/'seed_42/gen_00')/'test_sequences.json')
   x=np.asarray(seq['sequence_mean_nll']);n=np.asarray(seq['scored_tokens_per_sequence']);assert len(x)==7303 and np.all(n==64)
   near(np.dot(x,n)/n.sum(),m['test']['loss']);near(np.exp(m['test']['loss']),m['test']['perplexity'])
   near(np.median(x),m['test']['median_example_nll']);near(np.percentile(x,95),m['test']['p95_example_nll'])
   assert m['validation']['scored_tokens']==476608 and m['test']['scored_tokens']==467392
   if not cycle:assert m['weights_sha256']==baseline['weights_sha256']
   else:
    t=m['training'];inp=read_json(stage/'training_input/manifest.json');line=inp['lineage'];history=read_json(stage/'training_history.json')
    assert t['lineage']==line and t['initial_weights_sha256']==previous==line['initializer_weights_sha256']
    assert t['optimizer_parent_sha256']==optimizer==line['optimizer_parent_sha256']
    assert t['global_step']==79*cycle and t['total_optimizer_steps']==790 and t['executed_epochs']==t['selected_epoch']==1
    assert t['checkpoint_selection']=='final_epoch' and t['selected_weights_sha256']==t['terminal_weights_sha256']==m['weights_sha256']!=previous
    assert inp['pairs']==10000 and inp['scored_targets']==640000 and len(history)==1 and history[0]['scored_training_tokens']==640000
    near(history[0]['learning_rate_end'],1e-6+2e-6*(1+math.cos(math.pi*cycle/10)))
    assert sha256(stage/'training_input/manifest.json')==m['input_manifest_sha256']
    seal=read_json(stage/'stage_completion.json')
    for rel,h in seal['files'].items():
     full=stage.relative_to(child).as_posix()+'/'+rel
     assert full in index and index[full]['sha256']==h,full
     if (stage/rel).exists():assert sha256(stage/rel)==h,full
     seals+=1
    frac=cc['synthetic_fraction'];selection,nreal=select_rows(10000,frac,42,cycle);idx=selection['synthetic_indices']
    if bank:
     bp=Path(bank)/f'cycle_{cycle:02d}';prompt=np.load(bp/'prompts.npy');real=np.load(bp/'targets.npy')
     assert th(prompt)==inp['prompt_content_sha256'] and sha256(bp/'manifest.json')==line['chunk_manifest_sha256']
     assert line['bank_manifest_sha256']==sha256(Path(bank)/'manifest.json')
    if 0<frac<1:
     sm=read_json(stage/'mixture/selection.json');mm=read_json(stage/'mixture/manifest.json')
     assert sm['synthetic_pairs']==len(idx) and sm['real_pairs']==nreal and sm['generator_weights_sha256']==previous
     assert sha256(stage/'mixture/manifest.json')==line['mixture_manifest_sha256']==m['mixture']['mixture_manifest_sha256']
     for name,arr in selection.items():assert nh(arr)==sm['files'][name+'.npy']
     assert mm['mixed_targets_sha256']==inp['target_content_sha256']
    if frac:
     gm=read_json(child/f'synthetic_for_gen_{cycle:02d}/manifest.json');g=gm['provenance']
     assert g['parent_weights_sha256']==previous==line['generator_weights_sha256']
     assert g['prompt_count']==len(idx) and g['new_tokens']==64 and g['num_beams']==5 and g['repetition_penalty']==1
     assert sha256(child/f'synthetic_for_gen_{cycle:02d}/manifest.json')==line['generation_manifest_sha256']
     if bank:
      selected=prompt[idx] if frac<1 else prompt
      visible=np.column_stack([np.zeros(len(selected),dtype=np.int32),selected])
      assert th(visible)==g['prompt_tokens_sha256']
    else:
     assert line['generator_weights_sha256'] is None
     if bank:assert th(real)==inp['target_content_sha256']
    previous=m['weights_sha256'];optimizer=t['state_files']['optimizer_state/cycle.pt']
    checks.append({'arm':arm,'cycle':cycle,'weights':previous,'model_optimizer_generator_chain':True,'exact_counts':True,'index_seals_match':True})
   if bulk:
    br=Path(bulk)/(rid if cycle else RID);bi=read_json(br/'artifact_index.json')['files']
    candidates=[k for k in bi if k.startswith(f'seed_42/gen_{cycle:02d}/probe/') and k.endswith('/tokens.npy')];assert len(candidates)==1
    p=br/candidates[0];assert sha256(p)==bi[candidates[0]]['sha256'];a=np.load(p);assert a.shape==(256,64)
    result={**diversity(a),**probe_diagnostics(a,cc['newline_token_ids'])}
    for key,value in result.items():
     if isinstance(value,(int,float)):near(value,m['probe'][key])
     else:assert value==m['probe'][key]
    assert th(a)==m['probe']['continuation_tokens_sha256']
    probes.append({'arm':arm,'cycle':cycle,'sha256':sha256(p)});unique.add(str(p))
  if bank and bulk:
   bp=Path(bank)/'cycle_10';prompt=np.load(bp/'prompts.npy');real=np.load(bp/'targets.npy');frac=cc['synthetic_fraction'];sel,nreal=select_rows(10000,frac,42,10)
   target=real.copy()
   if frac:
    generated=np.load(Path(bulk)/rid/'synthetic_for_gen_10/tokens.npy');assert generated.shape==(len(sel['synthetic_indices']),64)
    if frac==1:target=generated
    else:target[sel['synthetic_indices']]=generated
   stage=Path(bulk)/rid/'seed_42/gen_10';inp=read_json(records/rid/'seed_42/gen_10/training_input/manifest.json')
   assert th(target)==inp['target_content_sha256']
   arrays=assemble(prompt,target,0)
   for name,a in zip(['tokens.npy','attention.npy','loss_mask.npy'],arrays):assert np.array_equal(np.load(stage/'training_input'/name),a)
   target_stats.append({'arm':arm,'synthetic_fraction':frac,'pairs':10000,'real_pairs':nreal,'synthetic_pairs':len(sel['synthetic_indices']),
    'target_content_sha256':th(target),'generator':'Original real text' if frac==0 else 'This arm checkpoint M9',**diversity(target),**probe_diagnostics(target,cc['newline_token_ids'])})
 inputs={p.relative_to(records).as_posix():sha256(p) for p in records.rglob('*.json')}
 return {'training_stages_verified':len(checks),'stage_checks':checks,'sealed_file_references_verified':seals,
  'logical_probe_rows_recomputed':len(probes),'unique_probe_arrays':len(unique),'probe_checks':probes,'inputs':inputs,
  'scope':'Pinned-index/file and saved-loss audit. No full checkpoint/optimizer restoration or new inference.'},target_stats

def build(records,out,bank=None,bulk=None,opt_metrics=None):
 records=Path(records).resolve();out=new_destination(records,out);out.mkdir(parents=True)
 checked,targets=audit(records,bank,bulk);rows,_,examples=_collect(records/RID);assert len(rows)==77
 atomic_json(out/'audit.json',checked);atomic_json(out/'training_target_statistics.json',targets)
 atomic_json(out/'metrics.json',rows);atomic_json(out/'examples.json',examples)
 with (out/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 series={arm:sorted([r for r in rows if r['arm']==arm],key=lambda r:r['cycle']) for arm in ARMS};ends={arm:r[-1] for arm,r in series.items()};baseline=series['real'][0]
 plt.rcParams.update({'font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white','font.family':'DejaVu Sans'})
 figures=[]
 def finish(fig,name,caption):
  # Fixed padding avoids title clipping when square axes and narrow data ranges
  # interact with automatic layout; every exported canvas remains square.
  fig.subplots_adjust(left=.15,right=.96,bottom=.14,top=.90)
  for axes in fig.axes:axes.title.set_fontsize(11.5);axes.title.set_wrap(True)
  fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.pdf'),metadata={'CreationDate':None,'ModDate':None});plt.close(fig);figures.append({'name':name,'caption':caption,'width_inches':7.5,'height_inches':7.5})
 def curve(name,field,title,ylabel,arms=ARMS,percent=False):
  fig,ax=plt.subplots(figsize=(7.5,7.5));ax.set_box_aspect(1)
  for arm in arms:
   rr=[r for r in series[arm] if r.get(field) is not None]
   ax.plot([r['cycle'] for r in rr],[r[field]*(100 if percent else 1) for r in rr],'-o',lw=2,ms=4,color=COLORS[arm],label=LABELS[arm]+' synthetic')
  ax.set(title=title,xlabel='Continued training cycle',ylabel=ylabel,xticks=range(11));ax.grid(alpha=.2);ax.legend(fontsize=9,loc='best');finish(fig,name,title)
 curve('01_test_perplexity','test_perplexity','SmolLM2: fixed real-test continuations','Corpus continuation perplexity')
 curve('02_test_perplexity_zoom','test_perplexity','Real-test perplexity: 0–99% synthetic','Corpus continuation perplexity',ARMS[:5])
 curve('03_validation_perplexity','validation_perplexity','Fixed real-validation continuations','Corpus continuation perplexity')
 curve('04_pooled_entropy','token_entropy_nats','Vocabulary spread across the fixed probe','Pooled token entropy (nats)')
 curve('05_within_entropy','mean_within_entropy_nats','Token diversity inside individual outputs','Mean within-output token entropy (nats)')
 curve('06_distinct2','distinct_2','Unique two-token phrases across outputs','Pooled distinct-2 (%)',percent=True)
 curve('07_distinct3','distinct_3','Unique three-token phrases across outputs','Pooled distinct-3 (%)',percent=True)
 curve('08_repeated4','mean_within_repeated_4gram_rate','Repeated phrases inside individual outputs','Mean within-output repeated 4-grams (%)',percent=True)
 curve('09_training_nll','training_loss','Fitting each cycle’s actual training targets','Post-epoch training NLL (nats / target)')
 curve('10_nll_p95','test_nll_p95','More surprising held-out real passages','P95 example NLL (nats / target)')
 curve('11_learning_rate','learning_rate_end','Common decreasing learning-rate schedule','Learning rate')
 curve('12_newlines','newline_token_share','Formatting in the fixed probe','Tokens containing a newline (%)',percent=True)
 for number,field,title,ylabel,percent in [
  (13,'test_perplexity','Cycle-10 perplexity versus synthetic fraction','Corpus continuation perplexity',False),
  (14,'token_entropy_nats','Cycle-10 pooled entropy versus synthetic fraction','Pooled token entropy (nats)',False),
  (15,'mean_within_entropy_nats','Cycle-10 local entropy versus synthetic fraction','Mean within-output entropy (nats)',False),
  (16,'distinct_2','Cycle-10 distinct-2 versus synthetic fraction','Pooled distinct-2 (%)',True),
  (17,'distinct_3','Cycle-10 distinct-3 versus synthetic fraction','Pooled distinct-3 (%)',True),
  (18,'mean_within_repeated_4gram_rate','Cycle-10 repetition versus synthetic fraction','Mean within-output repeated 4-grams (%)',True)]:
  fig,ax=plt.subplots(figsize=(7.5,7.5));ax.set_box_aspect(1)
  for arm in ARMS:
   r=ends[arm];ax.scatter(100*r['synthetic_fraction'],r[field]*(100 if percent else 1),s=65,color=COLORS[arm],label=LABELS[arm]+' synthetic')
  ax.set(title=title,xlabel='Synthetic training continuations (%)',ylabel=ylabel,xlim=(-3,104));ax.grid(alpha=.2);ax.legend(fontsize=9);finish(fig,f'{number:02d}_endpoint_'+field,title)
 corpora=['fineweb_edu','wikitext103','c4_en','pg19','ag_news'];matrix=np.array([[ends[a]['secondary_'+n+'_perplexity']/series[a][0]['secondary_'+n+'_perplexity'] for n in corpora] for a in ARMS])
 fig,ax=plt.subplots(figsize=(7.5,7.5));im=ax.imshow(np.log2(matrix),cmap='RdBu_r',vmin=-.3,vmax=4,aspect='auto')
 for i in range(7):
  for j in range(5):ax.text(j,i,f'{matrix[i,j]:.2f}×',ha='center',va='center',fontsize=9.5,color='white' if matrix[i,j]>3 else 'black')
 ax.set(xticks=range(5),xticklabels=['FineWeb','WikiText\n103','C4','PG-19','AG News'],yticks=range(7),yticklabels=[LABELS[a] for a in ARMS],ylabel='Synthetic training continuations',title='Full-text perplexity / model’s pretrained baseline');fig.colorbar(im,ax=ax,label='log₂ perplexity ratio',shrink=.6);finish(fig,'19_secondary_corpora','Secondary real-text evaluations; distinct from completion perplexity')
 cross=[]
 if opt_metrics:
  old=read_json(opt_metrics)
  for arm in ARMS:
   f=ends[arm]['synthetic_fraction'];a=[x for x in old if x['synthetic_fraction']==f];initial=next(x for x in a if int(x.get('cycle',x.get('generation',-1)))==0);terminal=next(x for x in a if int(x.get('cycle',x.get('generation',-1)))==10)
   cross.append({'fraction':f,'opt_relative_ppl':terminal['test_perplexity']/initial['test_perplexity'],'smollm_relative_ppl':ends[arm]['test_perplexity']/baseline['test_perplexity']})
  fig,ax=plt.subplots(figsize=(7.5,7.5));ax.set_box_aspect(1);xx=np.arange(7)
  ax.plot(xx,[x['opt_relative_ppl'] for x in cross],'o--',color='#53637C',label='OPT-125M (historical)');ax.plot(xx,[x['smollm_relative_ppl'] for x in cross],'o-',color='#A3314F',label='SmolLM2-135M')
  ax.set(xticks=xx,xticklabels=[LABELS[a] for a in ARMS],xlabel='Synthetic fraction (categorical spacing)',ylabel='Final test perplexity / own baseline',title='Cross-model context: baseline-relative deterioration');ax.grid(alpha=.2);ax.legend(fontsize=9);finish(fig,'20_cross_model_relative','Tokenizer, native bank and pretraining differences prevent a clean causal model-only comparison')
 summary={'baseline_test_perplexity':baseline['test_perplexity'],'final':ends,'relative_test_ppl':{a:ends[a]['test_perplexity']/baseline['test_perplexity'] for a in ARMS},'cross_model':cross,'secondary_relative_matrix':matrix.tolist()}
 atomic_json(out/'summary.json',summary);atomic_json(out/'figure_manifest.json',figures)
 try:revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 except subprocess.CalledProcessError:revision=None
 atomic_json(out/'analysis_manifest.json',{'inputs':checked['inputs'],'script_sha256':sha256(__file__),'revision':revision,'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__,'contract':comparison_contract(records/RID),'bank_manifest_sha256':sha256(Path(bank)/'manifest.json') if bank else None,'opt_metrics_sha256':sha256(opt_metrics) if opt_metrics else None,'outputs':{p.name:sha256(p) for p in out.iterdir() if p.is_file()}})
 print(json.dumps({'rows':len(rows),'training_stages':len(checked['stage_checks']),'logical_probes':len(checked['probe_checks']),'unique_probe_arrays':len(checked['unique_probe_arrays']) if isinstance(checked['unique_probe_arrays'],list) else checked['unique_probe_arrays'],'final_ppl':{a:ends[a]['test_perplexity'] for a in ARMS}}))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--records',required=True);p.add_argument('--out',required=True);p.add_argument('--bank');p.add_argument('--bulk');p.add_argument('--opt-metrics');build(**vars(p.parse_args()))
