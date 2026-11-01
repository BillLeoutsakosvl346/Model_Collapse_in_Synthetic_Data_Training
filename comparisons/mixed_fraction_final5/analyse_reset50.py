"""Audit the sealed matched 50/50 pair without contacting Runpod or loading a model.

--recover reads immutable Hugging Face objects only. Analysis uses saved JSON and
256x64 output probes; full weights/training arrays remain in private backups.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import shutil
import sys
import tarfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from model_collapse.common import PROJECT, STATE, atomic_json, digest, read_json, sha256

sys.path.insert(0, str(PROJECT))
from comparisons.continued_mixed50_vs_synthetic_final5.analyse import probe_statistics
from scripts.analysis.publication import new_destination, verified_publication, write_provenance

PARENT = '20261101T030306Z-full-mixed-campaign'
RUN = PARENT + '-reset-mixed50'
OLD_RUN = '20261031T163428Z-full-continued'
PUBLIC = PROJECT / 'reports' / RUN
RECORDS = PUBLIC / 'records'
OLD = PROJECT / 'reports' / OLD_RUN / 'records'
COLORS = {'continued':'#178b87', 'reset':'#7755a0', 'shared':'#626b73'}
LABELS = {'continued':'50/50 continued', 'reset':'50/50 reset to S0'}


def copy_verified(src, dest, checksum):
    src, dest = Path(src), Path(dest)
    assert sha256(src) == checksum, str(src)
    if dest.exists():
        assert sha256(dest) == checksum, str(dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)


def recover():
    from model_collapse.hub import download_snapshot_file
    proof = read_json(STATE / 'mixed_reset_analysis_20261101/sealed_source.json')
    assert proof and proof['run_id'] == RUN
    snapshot = proof['seal']['snapshot']
    local = PROJECT / 'runs' / RUN
    index_path = local / 'artifact_index.json'
    assert sha256(index_path) == snapshot['artifact_index_sha256']
    index = read_json(index_path)
    names = ['config.json','manifest.json','environment.json','data/manifest.json','completion.json',
             'shared_prefix.json','shared_mixture_verification.json','benchmark.json']
    for g in range(11):
        names += [f'seed_42/gen_{g:02d}/{n}' for n in
                  ('metrics.json','training.json','samples.json','test_sequences.json',
                   'reference_sequences.json','stage_completion.json','import.json')]
        if g:
            names += [f'seed_42/gen_{g:02d}/training_history.json']
        if g >= 2:
            names += [f'mixed_for_gen_{g:02d}/manifest.json',f'synthetic_for_gen_{g:02d}/manifest.json',
                      f'synthetic_for_gen_{g+1:02d}/manifest.json']
    names = sorted(set(n for n in names if n in index['files']))
    def one(rel):
        meta = index['files'][rel]
        source = local / rel
        if not source.exists() or sha256(source) != meta['sha256']:
            source = download_snapshot_file(RUN, snapshot['repos'], rel, meta, snapshot['revisions'])
        copy_verified(source, RECORDS/rel, meta['sha256'])
        return rel, meta
    with ThreadPoolExecutor(max_workers=4) as pool:
        selected = dict(pool.map(one, names))
    # Shared probes are exact historical observations, not generated again.
    old_manifest = read_json(OLD / 'recovery_manifest.json')
    for g in (0,1):
        rel = f'probes/gen_{g:02d}.npy'
        expected = old_manifest['selected_files'][rel]
        copy_verified(OLD/rel, RECORDS/rel, expected['sha256'])
        selected[rel] = {**expected,'source_run':OLD_RUN,'shared_observation':True}
    def probe(g):
        rel = f'synthetic_for_gen_{g+1:02d}/shard_0000000.npy'
        meta = index['files'][rel]
        assert meta['bytes'] == 65664
        src = download_snapshot_file(RUN,snapshot['repos'],rel,meta,snapshot['revisions'])
        key = f'probes/gen_{g:02d}.npy'
        copy_verified(src,RECORDS/key,meta['sha256'])
        return key,{**meta,'source_relative_path':rel,'source_run':RUN}
    with ThreadPoolExecutor(max_workers=4) as pool:
        selected.update(dict(pool.map(probe,range(2,11))))
    copy_verified(index_path,RECORDS/'artifact_index.json',snapshot['artifact_index_sha256'])
    selected['artifact_index.json']={'sha256':sha256(index_path),'bytes':index_path.stat().st_size}
    source_path=STATE/'mixed_reset_analysis_20261101/sealed_source.json'
    copy_verified(source_path,RECORDS/'sealed_source.json',sha256(source_path))
    selected['sealed_source.json']={'sha256':sha256(source_path),'bytes':source_path.stat().st_size}
    core={}
    for code in ('266afa984a2a51b63f850390232a12b3ec98a93975ad27f9401983d1afc060c5',
                 '35c89c3b9fb78c7967c4f62f07972739eabbe57b353a0360749eba71d1af2171'):
        with tarfile.open(STATE/f'code-{code}.tar.gz') as bundle:
            core[code]={'files':{},'mixing_functions':{}}
            for name in ('engine.py','data.py','workflow/selection.py'):
                raw=bundle.extractfile('model_collapse/'+name).read().replace(b'\r\n',b'\n')
                core[code]['files'][name]=hashlib.sha256(raw).hexdigest()
            tree=ast.parse(bundle.extractfile('model_collapse/mixing.py').read().decode())
            for node in tree.body:
                if isinstance(node,ast.FunctionDef) and node.name in ('synthetic_fraction','mixture_counts','reconstruct','sample_mixture','mixture_input'):
                    core[code]['mixing_functions'][node.name]=digest(ast.dump(node,include_attributes=False))
    assert len({digest(v['files']) for v in core.values()})==1
    assert len({digest(v['mixing_functions']) for v in core.values()})==1
    atomic_json(RECORDS/'runtime_core_identity.json',{'source_bundles':core,'scientific_core_identical':True,
        'note':'Complete runtime hashes differ for workflow/cloud support; training, data, selection and mixer functions are identical.'})
    selected['runtime_core_identity.json']={'sha256':sha256(RECORDS/'runtime_core_identity.json'),
                                          'bytes':(RECORDS/'runtime_core_identity.json').stat().st_size}
    receipt={'run_id':RUN,'snapshot':snapshot,'selected_files':selected,'selected_files_count':len(selected),
             'selected_files_bytes':sum(m['bytes'] for m in selected.values()),'new_probe_arrays':9,
             'new_probe_bytes':9*65664,'shared_probes_reused':2,
             'scope':'Read-only sealed Hub recovery; no Runpod API, SSH, model execution, full-array/weight downloads or resource changes.'}
    atomic_json(RECORDS/'recovery_manifest.json',receipt)
    return receipt


def csv_write(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def analyse(out):
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerBase
    out=new_destination(PUBLIC,out)
    verified_publication(PROJECT/'reports'/OLD_RUN)
    recovery=read_json(RECORDS/'recovery_manifest.json');assert recovery
    inputs=set()
    def track(p):inputs.add(Path(p));return Path(p)
    def read(p):return read_json(track(p))
    for rel,meta in recovery['selected_files'].items():assert sha256(track(RECORDS/rel))==meta['sha256'],rel
    seal=read(RECORDS/'sealed_source.json')['seal']
    assert sha256(track(RECORDS/'completion.json'))==seal['completion_sha256']
    assert sha256(track(RECORDS/'config.json'))==seal['configuration_sha256']
    assert sha256(track(RECORDS/'artifact_index.json'))==seal['snapshot']['artifact_index_sha256']
    assert read(RECORDS/'completion.json')['complete'] and read(RECORDS/'completion.json')['generations']==10
    configs={arm:read(root/'config.json') for arm,root in [('continued',OLD),('reset',RECORDS)]}
    a,b=configs['continued'],configs['reset']
    keys=('seed','model','model_revision','dataset','dataset_revision','dataset_config','block_size','new_tokens',
          'num_beams','repetition_penalty','epochs','checkpoint_selection','learning_rate','min_learning_rate',
          'weight_decay','microbatch','effective_batch','generation_batch','eval_batch','precision','probe_count',
          'start_checkpoint','synthetic_fraction')
    assert all(a[k]==b[k] for k in keys)
    assert a['initialization']=='continued' and b['initialization']=='reset_s0' and b['synthetic_fraction']==.5
    assert read(OLD/'environment.json')==read(RECORDS/'environment.json')
    assert read(OLD/'data/manifest.json')['files']==read(RECORDS/'data/manifest.json')['files']
    assert read(RECORDS/'runtime_core_identity.json')['scientific_core_identical']
    tokenizer_path=PROJECT/'reports/20261030T180755Z-full-paired-final5/records/continued/seed_42/gen_00/tokenizer/tokenizer.json'
    tokenizer=Tokenizer.from_file(str(track(tokenizer_path)))
    assert tokenizer.decode([50118],skip_special_tokens=False)=='\n' and tokenizer.decode([1437],skip_special_tokens=False)==' '
    roots={'continued':OLD,'reset':RECORDS};by={};seqs={};refs={};probes={};samples={};within={};histories=[];mixtures=[];prompt_hashes={37946:set(),256:set()}
    for arm,root in roots.items():
        by[arm]=[];seqs[arm]={};refs[arm]={};probes[arm]={};samples[arm]={};within[arm]={}
        for g in range(11):
            stage=root/f'seed_42/gen_{g:02d}';m,t=read(stage/'metrics.json'),read(stage/'training.json');assert m['training']==t
            for key,count in [('test',283122),('validation',247086),('frozen_g0_reference',16128)]:
                v=m[key];assert v['scored_tokens']==count and v['context_tokens']==64
                assert math.isclose(v['nll_sum']/count,v['loss'],abs_tol=1e-10)
                assert math.isclose(math.exp(v['loss']),v['perplexity'],rel_tol=1e-10)
            gp=m['generation_timing']['provenance'];assert gp['prompt_count']==(37946 if g<10 else 256)
            assert gp['new_tokens']==64 and gp['num_beams']==5 and gp['repetition_penalty']==1
            prompt_hashes[gp['prompt_count']].add(gp['prompt_tokens_sha256'])
            block=np.asarray(read(stage/'test_sequences.json')['sequence_mean_nll'])
            reference=np.asarray(read(stage/'reference_sequences.json')['sequence_mean_nll'])
            assert len(block)==4494 and len(reference)==256
            for vals,key in [(block,'test'),(reference,'frozen_g0_reference')]:
                assert np.isfinite(vals).all() and math.isclose(float(vals.mean()),m[key]['loss'],abs_tol=1e-10)
            assert math.isclose(float(np.exp(block).mean()),m['test']['mean_sequence_perplexity'],rel_tol=1e-10)
            assert math.isclose(float(np.exp(block).std()),m['test']['std_sequence_perplexity'],rel_tol=1e-10)
            probe=np.load(track(root/f'probes/gen_{g:02d}.npy'),allow_pickle=False);saved=read(stage/'samples.json')
            assert probe.shape==(256,64) and len(saved)==256
            for row,entry in zip(probe,saved):
                assert PreTrainedTokenizerBase.clean_up_tokenization(tokenizer.decode(row.tolist(),skip_special_tokens=False))==entry['continuation']
            measured,rep=probe_statistics(probe,saved)
            for key,value in {**m['diversity'],**m.get('probe_diagnostics',{})}.items():
                assert math.isclose(measured[key],value,abs_tol=1e-12),(arm,g,key)
            new=g>=(1 if arm=='continued' else 2)
            row={'arm':arm,'cycle':g,'test_nll':float(block.mean()),'test_perplexity':m['test']['perplexity'],
                 'validation_nll':m['validation']['loss'],'validation_perplexity':m['validation']['perplexity'],
                 'test_block_nll_p50':float(np.percentile(block,50)),'test_block_nll_p95':float(np.percentile(block,95)),
                 'frozen_s0_nll_mean':float(reference.mean()),'frozen_s0_nll_p95':float(np.percentile(reference,95)),
                 'selected_epoch':t['selected_epoch'],'executed_epochs':t.get('executed_epochs',0),
                 'best_validation_epoch':t.get('best_validation_epoch'),'selected_validation_loss':t.get('selected_validation_loss'),
                 'minimum_validation_loss':t.get('best_validation_loss'),'train_seconds':t['seconds'] if new else 0,
                 'generation_seconds':m['generation_timing']['seconds'] if new else 0,'new_training_stage':new,
                 'selected_weights_sha256':t['selected_weights_sha256'],**measured}
            by[arm].append(row);seqs[arm][g]=block;refs[arm][g]=reference;probes[arm][g]=probe;samples[arm][g]=saved;within[arm][g]=rep
            if g:
                assert t['selected_epoch']==t['executed_epochs']==5 and t['checkpoint_selection']=='final_epoch'
                assert t['selected_weights_sha256']==t['terminal_weights_sha256']!=t['initial_weights_sha256']
                assert t['optimizer_initialization']=='fresh_each_cycle'
                prev=read(root/f'seed_42/gen_{g-1:02d}/metrics.json')
                expected=prev['training']['selected_weights_sha256'] if arm=='continued' or g==1 else by[arm][0]['selected_weights_sha256']
                assert t['initial_weights_sha256']==expected
                assert t['lineage']['generator_weights_sha256']==prev['training']['selected_weights_sha256']
                mixroot=root if arm=='continued' or g>=2 else OLD
                mix=read(mixroot/f'mixed_for_gen_{g:02d}/manifest.json')
                assert mix['real_blocks']==mix['synthetic_blocks']==18973 and mix['training_blocks']==37946
                assert mix['sources']['real_split']=='train'
                assert mix['sources']['real_file_sha256']==read(root/'data/manifest.json')['files']['train.npy']
                assert mix['sources']['real_manifest_sha256']==sha256(track(root/'data/manifest.json'))
                assert mix['synthetic_fraction']==mix['effective_synthetic_fraction']==.5
                assert mix['real_tokens']==mix['synthetic_tokens']==1214272
                assert mix['training_tokens_sha256']==t['input_tokens_sha256']
                assert mix['sources']['synthetic_tokens_sha256']==prev['generation_timing']['continuation_tokens_sha256']==t['lineage']['synthetic_source_tokens_sha256']
                assert mix['sources']['generator_weights_sha256']==prev['training']['selected_weights_sha256']
                streams=[np.random.default_rng(np.random.SeedSequence([42,g,i])) for i in range(3)]
                selection={'real_indices':streams[0].choice(37946,18973,replace=False).tolist(),
                           'synthetic_indices':streams[1].choice(37946,18973,replace=False).tolist(),
                           'permutation':streams[2].permutation(37946).tolist()}
                selection['source_labels']=[int(i>=18973) for i in selection['permutation']]
                selection_hash=hashlib.sha256((json.dumps(selection,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
                assert selection_hash==mix['files']['selection.json']
                mixtures.append({'arm':arm,'cycle':g,'real_blocks':18973,'synthetic_blocks':18973,'real_tokens':1214272,
                                 'synthetic_tokens':1214272,'training_tokens_sha256':mix['training_tokens_sha256'],
                                 'selection_sha256':selection_hash,'new_execution':new})
                history=read(stage/'training_history.json');assert [v['epoch'] for v in history]==[1,2,3,4,5]
                assert all(math.isfinite(v) for h in history for v in h.values() if isinstance(v,(int,float)))
                assert math.isclose(history[-1]['validation_loss'],m['validation']['loss'],abs_tol=1e-10)
                histories += [{'arm':arm,'cycle':g,'new_execution':new,**h} for h in history]
            if new:
                completed=read(stage/'stage_completion.json')
                assert completed['selected_weights_sha256']==t['selected_weights_sha256']
                idx=read(root/'artifact_index.json')['files']
                for rel,checksum in completed['files'].items():assert idx[f'seed_42/gen_{g:02d}/'+rel]['sha256']==checksum
    for g in (0,1):
        for name in ('metrics.json','training.json','test_sequences.json','reference_sequences.json','samples.json'):
            assert sha256(track(OLD/f'seed_42/gen_{g:02d}'/name))==sha256(track(RECORDS/f'seed_42/gen_{g:02d}'/name))
        for data in (seqs,refs,probes):assert np.array_equal(data['continued'][g],data['reset'][g])
    assert mixtures[1]['training_tokens_sha256']==next(x['training_tokens_sha256'] for x in mixtures if x['arm']=='reset' and x['cycle']==2)
    assert all(len(v)==1 for v in prompt_hashes.values())
    assert len({hashlib.sha256(p.tobytes()).hexdigest() for arm in probes.values() for p in arm.values()})==20
    prompts=[e['prompt'] for e in samples['continued'][0]]
    assert all([e['prompt'] for e in ss]==prompts for arm in samples.values() for ss in arm.values())
    for g in range(1,11):
        lr=[[x['learning_rate'] for x in histories if x['arm']==arm and x['cycle']==g] for arm in roots];assert lr[0]==lr[1]
    gap=seqs['continued'][10]-seqs['reset'][10];rgap=within['continued'][10]-within['reset'][10]
    anchors=[(0,'Fixed historical anchor, retained before inspecting this comparison'),
             (int(np.argmin(abs(rgap-np.median(rgap)))),'First output nearest the median paired within-repetition difference')]
    for arm in roots:
        unusual=[i for i,e in enumerate(samples[arm][10]) if not e['continuation'].strip()]
        if unusual:anchors.append((unusual[0],f'First whitespace-only final output in {arm}'))
    examples=[]
    for i,rule in anchors:
        if any(e['prompt_index']==i for e in examples):continue
        examples.append({'prompt_index':i,'selection_rule':rule,'prompt':prompts[i],
           'shared_s1':samples['continued'][1][i]['continuation'],
           'outputs':{arm:{'text':samples[arm][10][i]['continuation'],'within_repeated4':float(within[arm][10][i])} for arm in roots}})
    endpoints={'s0':by['continued'][0],'shared_s1':by['continued'][1],**{arm:by[arm][10] for arm in roots}}
    summary={'passed':True,'run_id':RUN,'continued_run_id':OLD_RUN,'matched_settings':list(keys),
       'recorded_environment':read(RECORDS/'environment.json'),'shared_s0_s1_d2_m2_verified':True,
       'matched_probe_order':True,'unique_probes':20,'new_probe_arrays':9,'new_probe_bytes':590976,
       'new_reset_stages':9,'new_reset_epochs':45,'unique_pair_training_stages':19,'unique_pair_epochs':95,
       'source_and_reset_epoch_entries_checked':100,'total_metric_observations':22,'endpoint_metrics':endpoints,
       'reset_final_ppl_reduction_percent':100*(1-endpoints['reset']['test_perplexity']/endpoints['continued']['test_perplexity']),
       'reset_ppl_above_s0_percent':100*(endpoints['reset']['test_perplexity']/endpoints['s0']['test_perplexity']-1),
       'paired_test_blocks':{'total':4494,'reset_lower':int(np.sum(gap>0)),'continued_lower':int(np.sum(gap<0)),
                            'ties':int(np.sum(gap==0)),'mean_continued_minus_reset_nll':float(gap.mean())},
       'final_vs_s0_blocks':{arm:{'worse':int(np.sum(seqs[arm][10]>seqs[arm][0])),
                                  'better':int(np.sum(seqs[arm][10]<seqs[arm][0]))} for arm in roots},
       'paired_within_repetition':{'continued_more':int(np.sum(rgap>0)),'reset_more':int(np.sum(rgap<0)),'ties':int(np.sum(rgap==0))},
       'best_validation_epochs':{arm:[x['best_validation_epoch'] for x in by[arm][1:]] for arm in roots},
       'probe_scope':'Same first 256 consecutive real training prompts, 64 continuation tokens; six adjacent articles, not a random corpus sample.',
       'runtime':{'new_reset_training_seconds':sum(x['train_seconds'] for x in by['reset']),
                  'new_reset_generation_seconds':sum(x['generation_seconds'] for x in by['reset'])},
       'scope':'Saved statistics and token-probe audit only; no model forward pass, bulk arrays/weights, Runpod API or SSH. Paired blocks are descriptive, not independent model replicates.'}
    out.mkdir(parents=True,exist_ok=True)
    csv_write(out/'metrics.csv',by['continued']+by['reset']);csv_write(out/'training_history.csv',histories);csv_write(out/'mixture_counts.csv',mixtures)
    csv_write(out/'paired_test_blocks.csv',[{'block':i,'continued_nll':float(seqs['continued'][10][i]),'reset_nll':float(seqs['reset'][10][i]),'continued_minus_reset_nll':float(v)} for i,v in enumerate(gap)])
    atomic_json(out/'summary.json',summary);atomic_json(out/'examples.json',examples)
    plot(out,by,histories,seqs,refs,within,summary)
    write_provenance(PROJECT,out,__file__,inputs|{RECORDS/'recovery_manifest.json',
        PROJECT/'comparisons/continued_mixed50_vs_synthetic_final5/analyse.py'},scope=summary['scope'])
    return summary


def plot(out,by,histories,seqs,refs,within,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':12,'axes.titlesize':13,'axes.labelsize':11,'legend.fontsize':10})
    figdir=out/'figures';figdir.mkdir()
    def curves(ax,key,title,percent=False):
        for arm in by:
            ax.plot(range(1,11),[x[key]*(100 if percent else 1) for x in by[arm][1:]],'o-',color=COLORS[arm],label=LABELS[arm],markersize=3)
        ax.plot([0,1],[x[key]*(100 if percent else 1) for x in by['continued'][:2]],'o-',color=COLORS['shared'],label='Shared S0/S1',markersize=4)
        ax.set_title(title);ax.grid(alpha=.2);ax.set_xticks([0,2,4,6,8,10])
    fig,axs=plt.subplots(3,2,figsize=(10.5,7.8))
    for ax,key,title,pct in zip(axs.flat,['test_perplexity','validation_perplexity','distinct_2','distinct_3','token_entropy_nats','mean_within_repeated_4gram_rate'],
        ['A  Real-test corpus perplexity','B  Real-validation perplexity','C  Pooled distinct-2 (%)','D  Pooled distinct-3 (%)','E  Pooled token entropy (nats)','F  Mean within-output repetition (%)'],[False,False,True,True,False,True]):curves(ax,key,title,pct)
    for ax in axs[-1]:ax.set_xlabel('Synthetic training cycle')
    fig.legend(*axs.flat[0].get_legend_handles_labels(),loc='upper center',ncol=3,bbox_to_anchor=(.5,1.015))
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(figdir/'matched_trajectories.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(10.5,6.8));ax=axs[0,0]
    for arm in by:
        z=np.sort(seqs[arm][10]);ax.plot(z,np.arange(1,len(z)+1)/len(z),color=COLORS[arm],label=LABELS[arm])
    z=np.sort(seqs['continued'][0]);ax.plot(z,np.arange(1,len(z)+1)/len(z),color=COLORS['shared'],ls='--',label='Shared S0')
    ax.set(title='A  Final real-test block NLL',xlabel='Mean NLL per 64-token block',ylabel='Cumulative fraction');ax.legend()
    gap=seqs['continued'][10]-seqs['reset'][10];ax=axs[0,1];ax.hist(gap,bins=45,color=COLORS['reset'],alpha=.8)
    ax.axvline(0,color='#555',ls='--');ax.set(title='B  Paired test-block difference',xlabel='Continued − reset NLL',ylabel='Blocks')
    for ax,metric,title in [(axs[1,0],'validation_loss','C  Cycle 10: validation loss'),(axs[1,1],'train_loss','D  Cycle 10: training loss')]:
        for arm in by:
            h=[x for x in histories if x['arm']==arm and x['cycle']==10]
            ax.plot([x['epoch'] for x in h],[x[metric] for x in h],'o-',color=COLORS[arm],label=LABELS[arm])
        ax.axvline(5,color='#777',ls=':',label='Retained epoch 5');ax.set(title=title,xlabel='Epoch',ylabel='NLL');ax.legend(fontsize=9)
    for ax in axs.flat:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(figdir/'training_and_real_text.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(3,2,figsize=(10.5,7.8))
    curves(axs[0,0],'repeated_4gram_rate','A  Pooled repeated 4-grams (%)',True)
    curves(axs[0,1],'mean_within_entropy_nats','B  Within-output entropy (nats)')
    ax=axs[1,0]
    for j,arm in enumerate(by):ax.bar(np.arange(2)+(j-.5)*.33,[100*by[arm][-1][k] for k in ('newline_token_50118_share','standalone_space_share')],width=.33,color=COLORS[arm],label=LABELS[arm])
    ax.set_xticks([0,1],['Newline token','Standalone-space token']);ax.set(title='C  Final formatting-token share (%)');ax.legend(fontsize=9)
    ax=axs[1,1]
    for arm in by:
        z=np.sort(within[arm][10]);ax.plot(100*z,np.arange(1,len(z)+1)/len(z),color=COLORS[arm],label=LABELS[arm])
    ax.set(title='D  Final within-output repetition',xlabel='Repeated 4-grams (%)',ylabel='Cumulative fraction')
    curves(axs[2,0],'frozen_s0_nll_mean','E  Frozen-S0 NLL: mean')
    curves(axs[2,1],'frozen_s0_nll_p95','F  Frozen-S0 NLL: P95')
    for ax in axs.flat:ax.grid(alpha=.2)
    for ax in axs[-1]:ax.set_xlabel('Synthetic training cycle')
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,bbox_to_anchor=(.5,1.015))
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(figdir/'repetition_formatting_and_s0.png',dpi=180,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--recover',action='store_true');parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.recover:recover()
    s=analyse(args.out)
    print(json.dumps({'passed':s['passed'],'endpoints':{k:{n:v[n] for n in ('test_perplexity','distinct_2','mean_within_repeated_4gram_rate','token_entropy_nats')} for k,v in s['endpoint_metrics'].items()},'paired_blocks':s['paired_test_blocks']},indent=2))
