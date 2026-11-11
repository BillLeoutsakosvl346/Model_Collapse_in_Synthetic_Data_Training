"""Offline analysis of greedy, sampling and historical beam self-training.

Reads byte-preserved small records. Optional --bulk points to separately
recovered private probe/target arrays for recomputation. Never runs inference,
trains, reads credentials or contacts a cloud service.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from model_collapse.analysis.completion import _collect, comparison_contract
from model_collapse.common import read_json, sha256, atomic_json
from model_collapse.science.metrics import diversity, probe_diagnostics
from scripts.analysis.publication import new_destination

NEW='20261110T234038Z-full-completion64-decoding'
OLD='20261106T193918Z-full-completion64'
LABELS={'real':'Real targets (historical)','beam':'Five-beam targets (historical)',
        'greedy':'Greedy targets','sampling':'Pure-sampling targets'}
COLORS={'real':'#197C80','beam':'#A84150','greedy':'#D28218','sampling':'#6655B1'}


def content_hash(a):
    return hashlib.sha256(np.asarray(a,dtype=np.int32).tobytes()).hexdigest()


def close(a,b):
    if not math.isclose(a,b,abs_tol=1e-10,rel_tol=1e-10):
        raise AssertionError((a,b))


def audit_records(records, bulk=None, bank=None):
    """Check saved metadata and losses; optional arrays independently recompute probes."""
    records=Path(records); checks=[]; probe_checks=[]; input_hashes={}; historical_unindexed_logs=[]
    for parent in (NEW,OLD):
        root=records/parent;children=read_json(root/'children.json')
        assert read_json(root/'completion.json')['complete']
        assert len(children)==2
        for arm,rid in children.items():
            child=records/rid;index=read_json(child/'artifact_index.json')['files']
            snap=read_json(child/'recovery_snapshot.json')
            assert sha256(child/'artifact_index.json')==snap['artifact_index_sha256']
            assert all(len(v)==40 for v in snap['revisions'].values())
            for p in child.rglob('*.json'):
                rel=p.relative_to(child).as_posix()
                if rel in index: assert sha256(p)==index[rel]['sha256'],str(p)
            prev=read_json(child/'seed_42/gen_00/metrics.json')['weights_sha256']; optimizer=None
            for cycle in range(11):
                stage=child/f'seed_42/gen_{cycle:02d}';m=read_json(stage/'metrics.json')
                seq=read_json((stage if cycle else root/'seed_42/gen_00')/'test_sequences.json')
                x=np.asarray(seq['sequence_mean_nll']);n=np.asarray(seq['scored_tokens_per_sequence'])
                assert len(x)==7151 and np.all(n==64)
                close(float(np.average(x,weights=n)),m['test']['loss'])
                close(float(np.exp(np.average(x,weights=n))),m['test']['perplexity'])
                close(float(np.median(x)),m['test']['median_example_nll'])
                close(float(np.percentile(x,95)),m['test']['p95_example_nll'])
                assert m['test']['scored_tokens']==457664 and m['validation']['scored_tokens']==468608
                for split in ('test','validation'):
                    close(m[split]['loss']*m[split]['scored_tokens'],m[split]['nll_sum'])
                    close(math.exp(m[split]['loss']),m[split]['perplexity'])
                if cycle:
                    t=m['training'];inp=read_json(stage/'training_input/manifest.json')
                    seal=read_json(stage/'stage_completion.json')['files']
                    for rel,expected in seal.items():
                        if parent==OLD and rel=='probe/events.jsonl' and stage.relative_to(child).as_posix()+'/'+rel not in index:
                            historical_unindexed_logs.append(str(stage.relative_to(records)/rel))
                            continue
                        assert stage.relative_to(child).as_posix()+'/'+rel in index or (stage/rel).exists()
                        if (stage/rel).exists(): assert sha256(stage/rel)==expected
                        if stage.relative_to(child).as_posix()+'/'+rel in index:
                            assert index[stage.relative_to(child).as_posix()+'/'+rel]['sha256']==expected
                    assert t['initial_weights_sha256']==prev==inp['lineage']['initializer_weights_sha256']
                    assert t['optimizer_parent_sha256']==optimizer==inp['lineage']['optimizer_parent_sha256']
                    assert t['global_step']==79*cycle and t['total_optimizer_steps']==790
                    assert t['selected_epoch']==t['executed_epochs']==1
                    assert t['checkpoint_selection']=='final_epoch'
                    assert t['selected_weights_sha256']==m['weights_sha256']==t['terminal_weights_sha256']
                    assert t['initial_weights_sha256']!=m['weights_sha256']
                    assert inp['pairs']==10000 and inp['scored_targets']==640000
                    history=read_json(stage/'training_history.json');assert len(history)==1
                    assert history[0]['scored_training_tokens']==640000
                    close(history[0]['learning_rate_end'],1e-6+2e-6*(1+math.cos(math.pi*cycle/10)))
                    if bank:
                        bp=Path(bank)/f'cycle_{cycle:02d}'
                        prompts=np.load(bp/'prompts.npy')
                        assert content_hash(prompts)==inp['prompt_content_sha256']
                        assert sha256(bp/'manifest.json')==inp['lineage']['chunk_manifest_sha256']
                        if arm=='real': assert content_hash(np.load(bp/'targets.npy'))==inp['target_content_sha256']
                    if arm!='real':
                        gm=read_json(child/f'synthetic_for_gen_{cycle:02d}/manifest.json')
                        assert gm['provenance']['parent_weights_sha256']==prev==inp['lineage']['generator_weights_sha256']
                        assert sha256(child/f'synthetic_for_gen_{cycle:02d}/manifest.json')==inp['lineage']['generation_manifest_sha256']
                        assert gm['provenance']['prompt_count']==10000 and gm['provenance']['new_tokens']==64
                        if bank:
                            generated_prompts=np.column_stack([np.full(len(prompts),2,dtype=np.int32),prompts])
                            assert content_hash(generated_prompts)==gm['provenance']['prompt_tokens_sha256']
                        decoding=inp['lineage'].get('decoding',gm['provenance'].get('decoding'))
                        if arm=='greedy': assert decoding['num_beams']==1 and not decoding['do_sample']
                        if arm=='sampling': assert decoding['num_beams']==1 and decoding['do_sample'] and decoding['temperature']==1 and decoding['top_k']==0 and decoding['top_p']==1
                    prev=m['weights_sha256'];optimizer=t['state_files']['optimizer_state/cycle.pt']
                    checks.append({'run_id':rid,'cycle':cycle,'model_chain':True,'optimizer_chain':True,'source_chunk':cycle,
                                   'selected_terminal_weights':m['weights_sha256'],'test_nll_from_saved_examples':float(np.mean(x))})
                if bulk:
                    br=Path(bulk)/(rid if cycle else parent)
                    files=read_json(br/'artifact_index.json')['files']
                    candidates=[k for k in files if k.startswith(f'seed_42/gen_{cycle:02d}/probe/') and k.endswith('/tokens.npy')]
                    assert len(candidates)==1,(rid,cycle,candidates)
                    p=br/candidates[0];assert sha256(p)==files[candidates[0]]['sha256']
                    v=np.load(p);assert v.shape==(256,64)
                    actual={**diversity(v),**probe_diagnostics(v)}
                    for key,val in actual.items(): close(val,m['probe'][key])
                    probe_checks.append({'run_id':rid,'cycle':cycle,'array_sha256':sha256(p),'metrics_recomputed':len(actual)})
    for p in records.rglob('*.json'):
        input_hashes[p.relative_to(records).as_posix()]=sha256(p)
    return {'training_stages_verified':len(checks),'stage_checks':checks,'logical_probe_rows_recomputed':len(probe_checks),
            'probe_checks':probe_checks,'inputs':input_hashes,
            'historical_unindexed_probe_event_logs':historical_unindexed_logs,
            'scope':'Byte/pinned-index and saved-loss audit. No model tensors loaded or fresh inference in this analysis.'}


def build(records,out,bulk=None,bank=None):
    records=Path(records).resolve();out=new_destination(records,out)
    c=comparison_contract(records/NEW);old=comparison_contract(records/OLD)
    old['settings']['required_gpu']=c['settings']['required_gpu']
    assert c==old,'Unexpected scientific-contract mismatch beyond declared GPU type'
    audit=audit_records(records,bulk,bank)
    rows=[];examples={}
    for parent in (NEW,OLD):
        current,_,e=_collect(records/parent)
        assert len(current)==22
        for row in current:
            row['condition']={'synthetic':'beam'}.get(row['arm'],row['arm'])
            rows.append(row)
        examples[parent]=e
    out.mkdir(parents=True)
    atomic_json(out/'metrics.json',rows);atomic_json(out/'examples.json',examples);atomic_json(out/'audit.json',audit)
    with (out/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
    plt.rcParams.update({'font.size':11,'axes.titlesize':14,'axes.labelsize':12})
    figures=[]
    def finish(fig,name,title):
        fig.tight_layout()
        for ext in ('png','pdf'):
            fig.savefig(out/f'{name}.{ext}',dpi=180,metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else None)
        plt.close(fig);figures.append({'name':name,'title':title})
    def curve(name,field,title,ylabel,keys=tuple(LABELS),percent=False):
        fig,ax=plt.subplots(figsize=(8,8));ax.set_box_aspect(1)
        for key in keys:
            r=sorted([r for r in rows if r['condition']==key and r.get(field) is not None],key=lambda r:r['cycle'])
            ax.plot([x['cycle'] for x in r],[x[field]*(100 if percent else 1) for x in r],
                    marker='o',color=COLORS[key],label=LABELS[key],linestyle='--' if key in ('real','beam') else '-',linewidth=2)
        ax.set(xlabel='Continued training cycle',ylabel=ylabel,title=title,xticks=range(11));ax.grid(alpha=.23);ax.legend(fontsize=10)
        finish(fig,name,title)
    curve('01_test_perplexity','test_perplexity','Real-test perplexity: training-target decoder','Corpus continuation perplexity')
    curve('02_sampling_real_zoom','test_perplexity','Sampling and real targets: enlarged scale','Corpus continuation perplexity',('real','sampling'))
    curve('03_validation_perplexity','validation_perplexity','Fixed real-validation perplexity','Corpus continuation perplexity')
    curve('04_pooled_entropy','token_entropy_nats','Token diversity across the fixed probe','Pooled token entropy (nats)')
    curve('05_within_entropy','mean_within_entropy_nats','Token diversity inside individual outputs','Mean within-output token entropy (nats)')
    curve('06_distinct2','distinct_2','Unique two-token phrases across the probe','Pooled distinct-2 (%)',percent=True)
    curve('07_distinct3','distinct_3','Unique three-token phrases across the probe','Pooled distinct-3 (%)',percent=True)
    curve('08_repeated4','mean_within_repeated_4gram_rate','Repetition inside individual outputs','Mean within-output repeated 4-grams (%)',percent=True)
    curve('09_nll_p95','test_nll_p95','Harder real passages: 95th-percentile NLL','P95 example NLL (nats / target)')
    curve('10_training_nll','training_loss','Likelihood of each arm’s own training targets','Post-epoch training NLL (nats / target)')
    curve('11_learning_rate','learning_rate_end','Preserved global cosine learning-rate schedule','Learning rate')
    curve('12_newlines','newline_token_50118_share','Formatting in the fixed generated probe','Newline tokens (%)',percent=True)
    final={r['condition']:r for r in rows if r['cycle']==10};base={r['condition']:r for r in rows if r['cycle']==0}
    corpora=['fineweb_edu','wikitext103','c4_en','pg19','ag_news']
    matrix=np.array([[final[k][f'secondary_{n}_perplexity']/base[k][f'secondary_{n}_perplexity'] for n in corpora] for k in LABELS])
    fig,ax=plt.subplots(figsize=(8,8));im=ax.imshow(np.log2(matrix),cmap='RdBu_r',vmin=-1,vmax=4,aspect='auto')
    for i in range(4):
        for j in range(5): ax.text(j,i,f'{matrix[i,j]:.2f}×',ha='center',va='center',fontsize=11,color='white' if matrix[i,j]>3 else 'black')
    ax.set(xticks=range(5),xticklabels=['FineWeb','WikiText\n103','C4','PG-19','AG News'],yticks=range(4),yticklabels=list(LABELS.values()),title='Full-text perplexity / pretrained baseline')
    fig.colorbar(im,ax=ax,label='log₂ perplexity ratio',shrink=.65);finish(fig,'13_secondary_corpora','Transfer across frozen real-text corpora')
    target_stats=[]
    if bulk and bank:
        for condition in LABELS:
            if condition=='real': p=Path(bank)/'cycle_10/targets.npy';generator='real source text'
            else:
                rid=(OLD+'-synthetic') if condition=='beam' else NEW+'-'+condition
                p=Path(bulk)/rid/'synthetic_for_gen_10/tokens.npy'
                inp=read_json(records/rid/'seed_42/gen_10/training_input/manifest.json')
                assert content_hash(np.load(p))==inp['target_content_sha256']
                generator=inp['lineage']['generator_weights_sha256']
            a=np.load(p);assert a.shape==(10000,64)
            target_stats.append({'condition':condition,'source':'Cycle-10 training targets; synthetic generated by M9',
                                 'generator':generator,'array_sha256':sha256(p),**diversity(a),**probe_diagnostics(a)})
        atomic_json(out/'training_target_statistics.json',target_stats)
    atomic_json(out/'figure_manifest.json',figures)
    summary={'baseline_test_perplexity':base['greedy']['test_perplexity'],'final':final,
             'relative_change_test_ppl_percent':{k:100*(final[k]['test_perplexity']/base[k]['test_perplexity']-1) for k in LABELS},
             'sampling_vs_greedy_nll_difference':final['sampling']['test_loss']-final['greedy']['test_loss'],
             'hardware_difference':'New greedy/sampling: RTX PRO 4500 Blackwell; historical real/beam: RTX PRO 4000 Blackwell.',
             'scope':'One seed; finite ten-cycle schedule. All generated-output metrics use the same five-beam probe decoder.'}
    atomic_json(out/'summary.json',summary)
    try: revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except subprocess.CalledProcessError: revision=None
    atomic_json(out/'analysis_manifest.json',{'inputs':audit['inputs'],'script_sha256':sha256(__file__),'revision':revision,
        'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__,'contract':c,
        'outputs':{p.name:sha256(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({'out':str(out),'rows':len(rows),'audited_stages':len(audit['stage_checks']),
                      'probe_rows':len(audit['probe_checks']),'final_ppl':{k:final[k]['test_perplexity'] for k in LABELS}}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--records',required=True);p.add_argument('--out',required=True)
    p.add_argument('--bulk');p.add_argument('--bank');args=p.parse_args();build(**vars(args))
