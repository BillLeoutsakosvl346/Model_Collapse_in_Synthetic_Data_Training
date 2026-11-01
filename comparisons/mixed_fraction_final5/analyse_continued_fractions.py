"""Audit saved continued runs at 100%, 50% and 20% synthetic fractions.

Default execution is offline. --recover reads only the completed child's sealed
Hugging Face backup, never Runpod, SSH or active campaign workers.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from model_collapse.common import PROJECT, STATE, atomic_json, read_json, sha256
sys.path.insert(0,str(PROJECT))
from comparisons.continued_mixed50_vs_synthetic_final5 import analyse as previous
from scripts.analysis.publication import new_destination, verified_publication, write_provenance
from scripts.analysis.square_panel_figures import make_figures

PARENT='20261101T030306Z-full-mixed-campaign'
RUN=PARENT+'-continued-mixed20'
ROOT=PROJECT/'reports'/RUN
RECORDS=ROOT/'records'
FIFTY=PROJECT/'reports/20261031T163428Z-full-continued'
SYNTH=PROJECT/'reports/20261030T180755Z-full-paired-final5'


def recover():
    from model_collapse.hub import download_snapshot_file
    proof=read_json(STATE/'mixed20_analysis_20261101/sealed_source.json')
    assert proof['run_id']==RUN
    snapshot=proof['seal']['snapshot'];local=PROJECT/'runs'/RUN
    index_path=local/'artifact_index.json'
    if not index_path.exists():
        from huggingface_hub import hf_hub_download
        from model_collapse.hub import retry_transfer
        index_path=Path(retry_transfer(hf_hub_download,snapshot['repos']['dataset'],RUN+'/artifact_index.json',
            repo_type='dataset',revision=snapshot['revisions']['dataset']))
    assert sha256(index_path)==snapshot['artifact_index_sha256']
    index=read_json(index_path)
    names=['config.json','manifest.json','environment.json','data/manifest.json','completion.json','initial_dataset_import.json','benchmark.json']
    for g in range(11):
        names += [f'seed_42/gen_{g:02d}/{n}' for n in ('metrics.json','training.json','samples.json','test_sequences.json','reference_sequences.json','stage_completion.json','import.json')]
        names += [f'synthetic_for_gen_{g+1:02d}/manifest.json']
        if g:names += [f'seed_42/gen_{g:02d}/training_history.json',f'mixed_for_gen_{g:02d}/manifest.json']
    def copy(rel):
        meta=index['files'][rel];src=local/rel
        if not src.exists() or sha256(src)!=meta['sha256']:src=download_snapshot_file(RUN,snapshot['repos'],rel,meta,snapshot['revisions'])
        previous.immutable_copy(src,RECORDS/rel,meta['sha256']);return rel,meta
    with ThreadPoolExecutor(max_workers=4) as pool:selected=dict(pool.map(copy,sorted(set(n for n in names if n in index['files']))))
    def probe(g):
        rel=f'synthetic_for_gen_{g+1:02d}/shard_0000000.npy';meta=index['files'][rel];assert meta['bytes']==65664
        src=download_snapshot_file(RUN,snapshot['repos'],rel,meta,snapshot['revisions'])
        dest=f'probes/gen_{g:02d}.npy';previous.immutable_copy(src,RECORDS/dest,meta['sha256'])
        return dest,{**meta,'source_relative_path':rel,'source_run':RUN}
    with ThreadPoolExecutor(max_workers=4) as pool:selected.update(dict(pool.map(probe,range(1,11))))
    rel='probes/gen_00.npy';meta=read_json(FIFTY/'records/recovery_manifest.json')['selected_files'][rel]
    previous.immutable_copy(FIFTY/'records'/rel,RECORDS/rel,meta['sha256']);selected[rel]={**meta,'shared_observation':True}
    for src,name in [(index_path,'artifact_index.json'),(STATE/'mixed20_analysis_20261101/sealed_source.json','sealed_source.json')]:
        previous.immutable_copy(src,RECORDS/name,sha256(src));selected[name]={'sha256':sha256(src),'bytes':src.stat().st_size}
    atomic_json(RECORDS/'recovery_manifest.json',{'run_id':RUN,'snapshot':snapshot,'selected_files':selected,
        'selected_files_count':len(selected),'selected_files_bytes':sum(v['bytes'] for v in selected.values()),
        'new_probes':10,'new_probe_bytes':656640,'shared_s0_probe_reused':True,
        'scope':'Sealed Hub reads only; no Runpod/SSH, weights, full arrays or model evaluation.'})


def read_csv(path):
    with Path(path).open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    for row in rows:
        for key,val in row.items():
            if val in ('True','False'):row[key]=val=='True'
            elif val=='':row[key]=None
            else:
                try:row[key]=float(val)
                except (TypeError,ValueError):pass
    return rows


def csv_write(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(rows)


def analyse(out):
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerBase
    out=new_destination(ROOT,out);out.mkdir(parents=True)
    verified_publication(FIFTY)
    # The older paired publication uses a historical schema; the existing audit
    # below verifies its preserved recovery manifest and selected records.
    historical=previous.analyse(out/'historical_audit')
    old_rows=read_csv(out/'historical_audit/metrics.csv');histories=read_csv(out/'historical_audit/training_history.csv')
    by={arm:[r for r in old_rows if r['arm']==arm] for arm in ('synthetic_100','mixed_50')}
    recovery=read_json(RECORDS/'recovery_manifest.json');inputs=set()
    def read(p):inputs.add(Path(p));return read_json(p)
    for rel,meta in recovery['selected_files'].items():assert sha256(RECORDS/rel)==meta['sha256'];inputs.add(RECORDS/rel)
    seal=read(RECORDS/'sealed_source.json')['seal'];idx=read(RECORDS/'artifact_index.json')['files']
    assert sha256(RECORDS/'artifact_index.json')==seal['snapshot']['artifact_index_sha256']
    assert sha256(RECORDS/'config.json')==seal['configuration_sha256']
    assert sha256(RECORDS/'completion.json')==seal['completion_sha256']
    assert read(RECORDS/'completion.json')['complete'] and read(RECORDS/'completion.json')['generations']==10
    c=read(RECORDS/'config.json');old=read(FIFTY/'records/config.json')
    core=read(PROJECT/'reports/20261101T030306Z-full-mixed-campaign-reset-mixed50/records/runtime_core_identity.json')
    assert core['scientific_core_identical']
    assert c['initialization']=='continued' and c['synthetic_fraction']==.2
    for key in historical['matched_config_keys']:assert c[key]==old[key],key
    assert c['initial_dataset']['tokens_sha256']==old['initial_dataset']['tokens_sha256']
    env=read(RECORDS/'environment.json');assert env==read(FIFTY/'records/environment.json')
    data=read(RECORDS/'data/manifest.json');assert data['files']==read(FIFTY/'records/data/manifest.json')['files']
    tokenizer_path=SYNTH/'records/continued/seed_42/gen_00/tokenizer/tokenizer.json';inputs.add(tokenizer_path)
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    roots={'synthetic_100':SYNTH/'records/continued','mixed_50':FIFTY/'records','mixed_20':RECORDS}
    seqs={};within={};samples={};refs={};probe_hashes=set();prompts=None
    for arm,root in roots.items():
        seqs[arm]={};within[arm]={};samples[arm]={};refs[arm]={}
        for g in range(11):
            stage=root/f'seed_42/gen_{g:02d}'
            probe_path=SYNTH/f'records/probes/continued/gen_{g:02d}.npy' if arm=='synthetic_100' else root/f'probes/gen_{g:02d}.npy'
            inputs.add(probe_path);probe=np.load(probe_path,allow_pickle=False);probe_hashes.add(hashlib.sha256(probe.tobytes()).hexdigest())
            saved=read(stage/'samples.json');stats,rep=previous.probe_statistics(probe,saved)
            assert probe.shape==(256,64)
            if prompts is None:prompts=[s['prompt'] for s in saved]
            assert [s['prompt'] for s in saved]==prompts
            seqs[arm][g]=np.asarray(read(stage/'test_sequences.json')['sequence_mean_nll'])
            refs[arm][g]=np.asarray(read(stage/'reference_sequences.json')['sequence_mean_nll'])
            within[arm][g]=rep;samples[arm][g]=saved
            if arm!='mixed_20':continue
            m=read(stage/'metrics.json');t=read(stage/'training.json');assert m['training']==t
            assert len(seqs[arm][g])==4494 and len(refs[arm][g])==256
            for key,count,vals in [('test',283122,seqs[arm][g]),('validation',247086,None),('frozen_g0_reference',16128,refs[arm][g])]:
                v=m[key];assert v['scored_tokens']==count and v['context_tokens']==64
                assert math.isclose(v['loss'],v['nll_sum']/count,abs_tol=1e-10)
                assert math.isclose(math.exp(v['loss']),v['perplexity'],rel_tol=1e-10)
                if vals is not None:assert np.isfinite(vals).all() and math.isclose(float(vals.mean()),v['loss'],abs_tol=1e-10)
            assert math.isclose(float(np.exp(seqs[arm][g]).mean()),m['test']['mean_sequence_perplexity'],rel_tol=1e-10)
            assert math.isclose(float(np.exp(seqs[arm][g]).std()),m['test']['std_sequence_perplexity'],rel_tol=1e-10)
            for tokens,sample in zip(probe,saved):assert PreTrainedTokenizerBase.clean_up_tokenization(tokenizer.decode(tokens.tolist(),skip_special_tokens=False))==sample['continuation']
            for key,value in {**m['diversity'],**m.get('probe_diagnostics',{})}.items():assert math.isclose(stats[key],value,abs_tol=1e-12),(g,key)
            gp=m['generation_timing']['provenance'];assert gp['prompt_count']==(37946 if g<10 else 256)
            assert gp['new_tokens']==64 and gp['num_beams']==5 and gp['repetition_penalty']==1
            assert gp['prompt_tokens_sha256']==read(FIFTY/f'records/seed_42/gen_{g:02d}/metrics.json')['generation_timing']['provenance']['prompt_tokens_sha256']
            row={'arm':arm,'cycle':g,'test_nll':float(seqs[arm][g].mean()),'test_perplexity':m['test']['perplexity'],
                 'validation_nll':m['validation']['loss'],'validation_perplexity':m['validation']['perplexity'],
                 'test_block_nll_p50':float(np.percentile(seqs[arm][g],50)),'test_block_nll_p95':float(np.percentile(seqs[arm][g],95)),
                 'frozen_s0_nll_mean':float(refs[arm][g].mean()),'frozen_s0_nll_p95':float(np.percentile(refs[arm][g],95)),
                 'selected_epoch':t['selected_epoch'],'executed_epochs':t.get('executed_epochs',0),'best_validation_epoch':t.get('best_validation_epoch'),
                 'selected_validation_loss':t.get('selected_validation_loss'),'minimum_validation_loss':t.get('best_validation_loss'),
                 'train_seconds':t['seconds'] if g else 0,'generation_seconds':m['generation_timing']['seconds'] if g else 0,
                 'selected_weights_sha256':t['selected_weights_sha256'],**stats}
            by.setdefault(arm,[]).append(row)
            if g:
                prior=read(root/f'seed_42/gen_{g-1:02d}/metrics.json')
                assert t['selected_epoch']==t['executed_epochs']==5 and t['checkpoint_selection']=='final_epoch'
                assert t['selected_weights_sha256']==t['terminal_weights_sha256']!=t['initial_weights_sha256']
                assert t['optimizer_initialization']=='fresh_each_cycle'
                assert t['initial_weights_sha256']==prior['training']['selected_weights_sha256']==t['lineage']['generator_weights_sha256']
                mix=read(root/f'mixed_for_gen_{g:02d}/manifest.json')
                assert (mix['real_blocks'],mix['synthetic_blocks'],mix['training_blocks'])==(30357,7589,37946)
                assert mix['sources']['real_split']=='train' and mix['sources']['real_file_sha256']==data['files']['train.npy']
                assert mix['training_tokens_sha256']==t['input_tokens_sha256']
                assert mix['sources']['synthetic_tokens_sha256']==prior['generation_timing']['continuation_tokens_sha256']==t['lineage']['synthetic_source_tokens_sha256']
                assert mix['sources']['generator_weights_sha256']==t['initial_weights_sha256']
                streams=[np.random.default_rng(np.random.SeedSequence([42,g,i])) for i in range(3)]
                selection={'real_indices':streams[0].choice(37946,30357,replace=False).tolist(),'synthetic_indices':streams[1].choice(37946,7589,replace=False).tolist(),'permutation':streams[2].permutation(37946).tolist()}
                selection['source_labels']=[int(i>=30357) for i in selection['permutation']]
                assert hashlib.sha256((json.dumps(selection,indent=2,allow_nan=False)+'\n').encode()).hexdigest()==mix['files']['selection.json']
                h=read(stage/'training_history.json');assert [e['epoch'] for e in h]==[1,2,3,4,5]
                assert all(math.isfinite(v) for e in h for v in e.values() if isinstance(v,(int,float)))
                assert math.isclose(h[-1]['validation_loss'],m['validation']['loss'],abs_tol=1e-10)
                assert [e['learning_rate'] for e in h]==[e['learning_rate'] for e in histories if e['arm']=='mixed_50' and e['cycle']==g]
                histories.extend({'arm':arm,'cycle':g,**e} for e in h)
                complete=read(stage/'stage_completion.json');assert complete['selected_weights_sha256']==t['selected_weights_sha256']
                for rel,checksum in complete['files'].items():assert idx[f'seed_42/gen_{g:02d}/'+rel]['sha256']==checksum
    assert len(probe_hashes)==31 and len(histories)==150
    assert all(np.array_equal(seqs[a][0],seqs['mixed_20'][0]) for a in roots)
    assert all(by[a][0]['selected_weights_sha256']==by['mixed_20'][0]['selected_weights_sha256'] for a in roots)
    endpoints={'s0':by['mixed_20'][0],**{arm:rows[-1] for arm,rows in by.items()}}
    paired={};paired_rep={};mixture_rows=[];real_union=set();real_events=0
    for g in range(1,11):
        mix=read(RECORDS/f'mixed_for_gen_{g:02d}/manifest.json')
        real=np.random.default_rng(np.random.SeedSequence([42,g,0])).choice(37946,30357,replace=False)
        real_union.update(map(int,real));real_events+=len(real)
        mixture_rows.append({'cycle':g,'real_blocks':30357,'synthetic_blocks':7589,'real_tokens':1942848,'synthetic_tokens':485696,
            'training_tokens_sha256':mix['training_tokens_sha256'],'selection_sha256':mix['files']['selection.json']})
    for arm in ('synthetic_100','mixed_50'):
        gap=seqs[arm][10]-seqs['mixed_20'][10];rgap=within[arm][10]-within['mixed_20'][10]
        paired[arm]={'mixed20_lower':int(np.sum(gap>0)),'other_lower':int(np.sum(gap<0)),'ties':int(np.sum(gap==0)),'mean_other_minus_mixed20_nll':float(gap.mean())}
        paired_rep[arm]={'mixed20_more':int(np.sum(rgap<0)),'other_more':int(np.sum(rgap>0)),'ties':int(np.sum(rgap==0))}
    summary={'passed':True,'run_id':RUN,'arms':list(by),'same_s0_d1':True,'matched_environment':env,'metric_rows':33,'unique_probes':31,'epoch_entries':150,
        'endpoint_metrics':endpoints,'first_update_metrics':{arm:rows[1] for arm,rows in by.items()},'paired_test_blocks':paired,'paired_repetition':paired_rep,
        'worse_than_s0_blocks':{arm:int(np.sum(seqs[arm][10]>seqs[arm][0])) for arm in by},
        'ppl_reduction_percent_vs':{arm:100*(1-endpoints['mixed_20']['test_perplexity']/endpoints[arm]['test_perplexity']) for arm in ('synthetic_100','mixed_50')},
        'mixed20_ppl_above_s0_percent':100*(endpoints['mixed_20']['test_perplexity']/endpoints['s0']['test_perplexity']-1),
        'best_validation_epochs':{arm:[r['best_validation_epoch'] for r in rows[1:]] for arm,rows in by.items()},
        'effective_synthetic_fraction':7589/37946,'blocks':{'real':30357,'synthetic':7589,'total':37946},
        'tokens':{'real':30357*64,'synthetic':7589*64,'total':2428544},
        'real_sampling':{'unique_blocks':len(real_union),'block_events':real_events,'available_blocks':37946},
        'new_work_seconds':{'training':sum(r['train_seconds'] for r in by['mixed_20']),'generation':sum(r['generation_seconds'] for r in by['mixed_20'])},
        'scope':'Saved-record/probe/lineage arithmetic audit only. No Runpod API, SSH, model forward pass or full weights/training-array recovery. One seed; fixed first 256 prompts from six adjacent articles; no significance claim.'}
    anchors=[0]
    for arm in by:
        empty=next((i for i,e in enumerate(samples[arm][10]) if not e['continuation'].strip()),None)
        if empty is not None and empty not in anchors:anchors.append(empty)
    examples=[{'prompt_index':i,'selection_rule':'Fixed historical anchor' if i==0 else 'First whitespace-only endpoint in at least one arm',
               'prompt':prompts[i],'outputs':{arm:{'text':samples[arm][10][i]['continuation'],'within_repeated4':float(within[arm][10][i])} for arm in by}} for i in anchors]
    csv_write(out/'metrics.csv',[r for rows in by.values() for r in rows]);csv_write(out/'training_history.csv',histories)
    csv_write(out/'mixture_counts.csv',mixture_rows)
    csv_write(out/'paired_test_blocks.csv',[{'block':i,'s0_nll':float(seqs['mixed_20'][0][i]),**{a+'_nll':float(seqs[a][10][i]) for a in by}} for i in range(4494)])
    atomic_json(out/'summary.json',summary);atomic_json(out/'examples.json',examples)
    make_figures(out/'figures',by,histories,seqs,within)
    write_provenance(PROJECT,out,__file__,inputs|{RECORDS/'recovery_manifest.json',out/'historical_audit/publication_reproduction.json',
        PROJECT/'comparisons/continued_mixed50_vs_synthetic_final5/analyse.py',PROJECT/'scripts/analysis/square_panel_figures.py'},scope=summary['scope'])
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recover',action='store_true');p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.recover:recover()
    s=analyse(args.out);print(json.dumps({'passed':s['passed'],'endpoints':{k:v['test_perplexity'] for k,v in s['endpoint_metrics'].items()},'paired':s['paired_test_blocks']},indent=2))
