"""Audit four saved continued-training conditions; default execution is offline."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from model_collapse.common import PROJECT, STATE, atomic_json, read_json, sha256
sys.path.insert(0, str(PROJECT))
from comparisons.continued_mixed50_vs_synthetic_final5 import analyse as previous
from scripts.analysis.publication import new_destination, verified_publication, write_provenance
from scripts.analysis.square_panel_figures import make_figures, square_grid, save, COLORS, LABELS

RUN = '20261102T154833Z-full-continued'
ROOT = PROJECT / 'reports' / RUN
RECORDS = ROOT / 'records'
SYNTH = PROJECT / 'reports/20261030T180755Z-full-paired-final5'
FIFTY = PROJECT / 'reports/20261031T163428Z-full-continued'
TWENTY = PROJECT / 'reports/20261101T030306Z-full-mixed-campaign-continued-mixed20'
ARMS = {'synthetic_100': SYNTH / 'records/continued', 'mixed_50': FIFTY / 'records',
        'mixed_20': TWENTY / 'records', 'real_0': RECORDS}

def recover():
    """Recover pinned small records/probes; never invoke Runpod or load a model."""
    from model_collapse.common import load_secrets
    from model_collapse.hub import fetch, download_snapshot_file
    load_secrets()
    receipt = read_json(PROJECT / 'docs/validation/2026-11-02_real-only-completion-status.json')
    repos = read_json(PROJECT / 'docs/validation/2026-11-02_real-only-full-launch.json')['recovery']['private_repositories']
    snapshot = {'run_id': RUN, 'repos': repos, 'revisions': receipt['pinned_backup']['revisions'],
                'artifact_index_sha256': receipt['pinned_backup']['artifact_index_sha256']}
    local = STATE / 'real_only_execution_audit_20261102/records'
    proof = read_json(local / 'recovery_snapshot.json')
    if not proof or proof['revisions'] != snapshot['revisions']:
        proof = fetch(RUN, repos, local, metadata_only=True, revisions=snapshot['revisions'])
    assert sha256(local / 'artifact_index.json') == snapshot['artifact_index_sha256']
    index = read_json(local / 'artifact_index.json')
    names = ['config.json', 'manifest.json', 'environment.json', 'data/manifest.json', 'benchmark.json',
             'initial_dataset_import.json', 'completion.json']
    for g in range(11):
        names += [f'seed_42/gen_{g:02d}/{n}' for n in ['metrics.json', 'training.json', 'samples.json',
                  'test_sequences.json', 'reference_sequences.json', 'stage_completion.json']]
        names += [f'synthetic_for_gen_{g+1:02d}/manifest.json']
        if g: names += [f'seed_42/gen_{g:02d}/training_history.json', f'mixed_for_gen_{g:02d}/manifest.json']
    selected = {}
    for rel in names:
        meta = index['files'][rel]
        previous.immutable_copy(local / rel, RECORDS / rel, meta['sha256']); selected[rel] = meta
    def probe(g):
        from huggingface_hub.errors import LocalEntryNotFoundError
        rel = f'synthetic_for_gen_{g+1:02d}/shard_0000000.npy'; meta = index['files'][rel]
        assert meta['bytes'] == 65664
        dst = f'probes/gen_{g:02d}.npy'
        if (RECORDS/dst).is_file() and sha256(RECORDS/dst)==meta['sha256']:
            return dst, {**meta, 'source_relative_path': rel, 'source_run': RUN}
        for attempt in range(4):
            try:
                source = download_snapshot_file(RUN, repos, rel, meta, snapshot['revisions']);break
            except LocalEntryNotFoundError:
                if attempt==3:raise
                time.sleep(2)
        previous.immutable_copy(source, RECORDS / dst, meta['sha256'])
        return dst, {**meta, 'source_relative_path': rel, 'source_run': RUN}
    with ThreadPoolExecutor(max_workers=4) as pool:
        selected.update(dict(pool.map(probe, range(1,11))))
    rel = 'probes/gen_00.npy'; meta = read_json(FIFTY / 'records/recovery_manifest.json')['selected_files'][rel]
    previous.immutable_copy(FIFTY / 'records' / rel, RECORDS / rel, meta['sha256'])
    selected[rel] = {**meta, 'shared_s0_probe': True}
    previous.immutable_copy(local / 'artifact_index.json', RECORDS / 'artifact_index.json', snapshot['artifact_index_sha256'])
    selected['artifact_index.json'] = {'sha256': snapshot['artifact_index_sha256'], 'bytes': (local / 'artifact_index.json').stat().st_size}
    atomic_json(RECORDS / 'recovery_manifest.json', {'run_id': RUN, 'snapshot': snapshot, 'selected_files': selected,
        'selected_files_count': len(selected), 'selected_files_bytes': sum(x['bytes'] for x in selected.values()),
        'new_probes': 10, 'new_probe_bytes': 656640, 'shared_s0_probe_reused': True,
        'metadata_recovery': {k:proof[k] for k in ['verified_files','verified_bytes','metadata_only']},
        'scope': 'Pinned small-record/probe recovery only; no Runpod, SSH, weights or new training.'})

def csv_write(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader(); writer.writerows(rows)

def analyse(out):
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerBase
    out=new_destination(ROOT,out);out.mkdir(parents=True)
    for published in (FIFTY,TWENTY): verified_publication(published)
    legacy=read_json(SYNTH/'records/recovery_manifest.json')
    for rel,meta in legacy['files'].items():assert sha256(SYNTH/'records'/rel)==(meta if isinstance(meta,str) else meta['sha256'])
    recovery=read_json(RECORDS/'recovery_manifest.json');inputs=set()
    def read(path):inputs.add(Path(path));return read_json(path)
    for rel,meta in recovery['selected_files'].items():
        assert sha256(RECORDS/rel)==meta['sha256'];inputs.add(RECORDS/rel)
    index=read(RECORDS/'artifact_index.json')['files']
    c=read(RECORDS/'config.json'); other=read(TWENTY/'records/config.json')
    matching=('seed','model','model_revision','dataset','dataset_revision','dataset_config','block_size','new_tokens','num_beams',
        'repetition_penalty','epochs','checkpoint_selection','learning_rate','min_learning_rate','weight_decay','microbatch',
        'effective_batch','generation_batch','eval_batch','precision','probe_count','start_checkpoint')
    for arm,root in ARMS.items():
        cfg=read(root/'config.json')
        assert all(cfg[k]==c[k] for k in matching),arm
        assert cfg['initialization']=='continued'
        if arm!='synthetic_100':assert cfg['initial_dataset']['tokens_sha256']==c['initial_dataset']['tokens_sha256']
    assert c['synthetic_fraction']==0 and c['kind']=='full' and all(c[k] is None for k in ['train_blocks','validation_blocks','test_blocks'])
    new_env=read(RECORDS/'environment.json');old_env=read(TWENTY/'records/environment.json')
    assert {k:v for k,v in new_env.items() if k!='gpu'}=={k:v for k,v in old_env.items() if k!='gpu'}
    assert read(RECORDS/'data/manifest.json')['files']==read(TWENTY/'records/data/manifest.json')['files']
    timing=read(ROOT/'analysis/timing_audit.json');assert all(timing['checks'].values())
    core=read(RECORDS/'runtime_core_identity.json');assert core['all_core_ast_equal']
    tokenizer_path=SYNTH/'records/continued/seed_42/gen_00/tokenizer/tokenizer.json';inputs.add(tokenizer_path)
    tokenizer=Tokenizer.from_file(str(tokenizer_path));prompts=None
    by={};histories=[];seqs={};within={};samples={};refs={};probe_hashes=set()
    for arm,root in ARMS.items():
        by[arm]=[];seqs[arm]={};within[arm]={};samples[arm]={};refs[arm]={}
        for g in range(11):
            stage=root/f'seed_42/gen_{g:02d}'
            probe_path=SYNTH/f'records/probes/continued/gen_{g:02d}.npy' if arm=='synthetic_100' else root/f'probes/gen_{g:02d}.npy'
            inputs.add(probe_path);probe=np.load(probe_path,allow_pickle=False);assert probe.shape==(256,64)
            probe_hashes.add(hashlib.sha256(probe.tobytes()).hexdigest())
            saved=read(stage/'samples.json');stats,rep=previous.probe_statistics(probe,saved)
            if prompts is None:prompts=[s['prompt'] for s in saved]
            assert [s['prompt'] for s in saved]==prompts
            for tokens,sample in zip(probe,saved):
                assert PreTrainedTokenizerBase.clean_up_tokenization(tokenizer.decode(tokens.tolist(),skip_special_tokens=False))==sample['continuation']
            m=read(stage/'metrics.json');t=read(stage/'training.json');assert m['training']==t
            seqs[arm][g]=np.asarray(read(stage/'test_sequences.json')['sequence_mean_nll'])
            refs[arm][g]=np.asarray(read(stage/'reference_sequences.json')['sequence_mean_nll'])
            within[arm][g]=rep;samples[arm][g]=saved
            assert len(seqs[arm][g])==4494 and len(refs[arm][g])==256
            for key,count,vals in [('test',283122,seqs[arm][g]),('validation',247086,None),('frozen_g0_reference',16128,refs[arm][g])]:
                z=m[key];assert z['scored_tokens']==count and z['context_tokens']==64
                assert math.isclose(z['loss'],z['nll_sum']/count,abs_tol=1e-10) and math.isclose(math.exp(z['loss']),z['perplexity'],rel_tol=1e-10)
                if vals is not None:assert np.isfinite(vals).all() and math.isclose(float(vals.mean()),z['loss'],abs_tol=1e-10)
            for key,value in {**m['diversity'],**m.get('probe_diagnostics',{})}.items():assert math.isclose(stats[key],value,abs_tol=1e-12),(arm,g,key)
            row={'arm':arm,'cycle':g,'test_nll':m['test']['loss'],'test_perplexity':m['test']['perplexity'],
                'validation_nll':m['validation']['loss'],'validation_perplexity':m['validation']['perplexity'],
                'test_block_nll_p50':float(np.percentile(seqs[arm][g],50)),'test_block_nll_p95':float(np.percentile(seqs[arm][g],95)),
                'frozen_s0_nll_mean':float(refs[arm][g].mean()),'frozen_s0_nll_p95':float(np.percentile(refs[arm][g],95)),
                'selected_epoch':t['selected_epoch'],'executed_epochs':t.get('executed_epochs',0),'best_validation_epoch':t.get('best_validation_epoch'),
                'train_seconds':t['seconds'] if g else 0,'generation_seconds':m['generation_timing']['seconds'] if g else 0,
                'selected_weights_sha256':t['selected_weights_sha256'],**stats}
            by[arm].append(row)
            if g:
                prior=read(root/f'seed_42/gen_{g-1:02d}/training.json')
                assert t['initial_weights_sha256']==prior['selected_weights_sha256'] and t['initial_weights_sha256']!=t['selected_weights_sha256']
                assert t['selected_epoch']==t['executed_epochs']==5 and t['selected_weights_sha256']==t['terminal_weights_sha256']
                assert t['checkpoint_selection']=='final_epoch' and t['optimizer_initialization']=='fresh_each_cycle'
                h=read(stage/'training_history.json');assert [e['epoch'] for e in h]==list(range(1,6))
                assert math.isclose(h[-1]['validation_loss'],m['validation']['loss'],abs_tol=1e-10)
                histories.extend({'arm':arm,'cycle':g,**e} for e in h)
                if arm=='real_0':
                    mix=read(root/f'mixed_for_gen_{g:02d}/manifest.json')
                    assert (mix['real_blocks'],mix['synthetic_blocks'],mix['training_blocks'])==(37946,0,37946)
                    assert mix['training_tokens_sha256']==t['input_tokens_sha256']
                    streams=[np.random.default_rng(np.random.SeedSequence([42,g,k])) for k in range(3)]
                    selection={'real_indices':streams[0].choice(37946,37946,replace=False).tolist(),'synthetic_indices':streams[1].choice(256 if g>1 else 37946,0,replace=False).tolist(),'permutation':streams[2].permutation(37946).tolist()}
                    selection['source_labels']=[0]*37946
                    assert hashlib.sha256((json.dumps(selection,indent=2,allow_nan=False)+'\n').encode()).hexdigest()==mix['files']['selection.json']
                    assert t['lineage']['initializer_weights_sha256']==t['lineage']['generator_weights_sha256']==t['initial_weights_sha256']
                    assert m['generation_timing']['provenance']['prompt_count']==256
    assert len(histories)==200 and len(probe_hashes)==41
    assert all(np.array_equal(seqs[a][0],seqs['real_0'][0]) for a in ARMS)
    assert all(np.array_equal(refs[a][0],refs['real_0'][0]) for a in ARMS)
    endpoints={'s0':by['real_0'][0],**{a:rows[-1] for a,rows in by.items()}}
    paired={};paired_rep={}
    for arm in list(ARMS)[:-1]:
        gap=seqs[arm][10]-seqs['real_0'][10];rgap=within[arm][10]-within['real_0'][10]
        paired[arm]={'real_only_lower':int(np.sum(gap>0)),'other_lower':int(np.sum(gap<0)),'ties':int(np.sum(gap==0)),'mean_other_minus_real_nll':float(gap.mean())}
        paired_rep[arm]={'real_only_more_repetitive':int(np.sum(rgap<0)),'other_more_repetitive':int(np.sum(rgap>0)),'ties':int(np.sum(rgap==0))}
    exposure=[]
    for arm,nreal,nsynth in [('synthetic_100',0,37946),('mixed_50',18973,18973),('mixed_20',30357,7589),('real_0',37946,0)]:
        exposure.append({'arm':arm,'real_blocks_per_cycle':nreal,'synthetic_blocks_per_cycle':nsynth,'total_blocks':37946,
            'real_tokens_per_cycle':nreal*64,'synthetic_tokens_per_cycle':nsynth*64,'mean_additional_real_block_passes':nreal*50/37946,
            'total_real_input_token_exposures':nreal*64*50,'total_synthetic_input_token_exposures':nsynth*64*50})
    summary={'passed':True,'run_id':RUN,'arms':list(ARMS),'metric_rows':44,'unique_probes':41,'epoch_entries':200,
        'matched_configuration_keys':matching,'same_s0_d1':True,'same_full_evaluation_arrays':True,'same_s0_evaluation_arrays':True,
        'environment':{'real_only':new_env,'historical_comparators':old_env,'difference':'GPU only; software/driver settings match.'},
        'endpoint_metrics':endpoints,'first_update_metrics':{a:rows[1] for a,rows in by.items()},'paired_test_blocks':paired,'paired_repetition':paired_rep,
        'real_only_vs_final_ppl_percent':{a:100*(endpoints['real_0']['test_perplexity']/endpoints[a]['test_perplexity']-1) for a in list(ARMS)[:-1]},
        'best_validation_epochs':{a:[r['best_validation_epoch'] for r in rows[1:]] for a,rows in by.items()},
        'worse_than_s0_test_blocks':{a:int(np.sum(seqs[a][10]>seqs[a][0])) for a in ARMS},'exposures':exposure,
        'timing_audit_passed':True,'real_only_work_seconds':timing['timing_seconds'],
        'scope':'Offline saved-record/probe audit; no Runpod/SSH/model forward passes. One seed, fixed 256-prompt six-article probe; GPU differs for real-only control. No full checkpoint restoration.'}
    anchors=[0,52]
    examples=[{'prompt_index':i,'selection_rule':'Fixed historical anchor' if i==0 else 'Previously reported formatting diagnostic anchor',
        'prompt':prompts[i],'outputs':{a:{'text':samples[a][10][i]['continuation'],'within_repeated4':float(within[a][10][i])} for a in ARMS}} for i in anchors]
    csv_write(out/'metrics.csv',[r for rows in by.values() for r in rows]);csv_write(out/'training_history.csv',histories)
    csv_write(out/'training_exposures.csv',exposure)
    csv_write(out/'paired_test_blocks.csv',[{'block':i,'s0_nll':float(seqs['real_0'][0][i]),**{a+'_nll':float(seqs[a][10][i]) for a in ARMS}} for i in range(4494)])
    atomic_json(out/'summary.json',summary);atomic_json(out/'examples.json',examples)
    make_figures(out/'figures',by,histories,seqs,within,cycle_label='Continued-training cycle',target_label='real-only',two_panel_height=5.25)
    fig,axes=square_grid(2,height=5.25)
    axes[0].plot([x['mean_additional_real_block_passes'] for x in exposure],[by[x['arm']][-1]['test_perplexity'] for x in exposure],color='#aaa',lw=1)
    for x in exposure:
        arm=x['arm'];xx=x['mean_additional_real_block_passes'];yy=by[arm][-1]['test_perplexity']
        axes[0].scatter([xx],[yy],color=COLORS[arm],s=38,label=LABELS[arm],zorder=3)
    axes[0].set(title='A  Real-data exposure and final PPL',xlabel='Mean real-block passes after S0',ylabel='Real-test perplexity')
    axes[1].plot(range(1,11),[r['train_loss'] for r in histories if r['arm']=='real_0' and r['epoch']==5],'o-',color='#7755a0',label='Real-only: training')
    axes[1].plot(range(1,11),[r['validation_nll'] for r in by['real_0'][1:]],'s-',color='#ba4d55',label='Real-only: validation')
    axes[1].axhline(by['real_0'][0]['validation_nll'],ls='--',color='#626b73',label='S0 validation')
    axes[1].set(title='B  Real-only fitting / generalization',xlabel='Continued-training cycle',ylabel='NLL (nats/target)');axes[1].legend(fontsize=8)
    save(fig,axes,out/'figures/exposure_and_overfitting.png')
    fig,axes=square_grid(4)
    ordered=[('real_0',0),('mixed_20',20),('mixed_50',50),('synthetic_100',100)]
    for ax,key,title,scale,ylabel in [
        (axes[0],'test_perplexity','A  Final real-test perplexity',1,'Corpus perplexity'),
        (axes[1],'token_entropy_nats','B  Final pooled token entropy',1,'Entropy (nats)'),
        (axes[2],'distinct_2','C  Final pooled distinct-2',100,'Unique bigram windows (%)'),
        (axes[3],'mean_within_repeated_4gram_rate','D  Final within-output repetition',100,'Repeated 4-gram windows (%)')]:
        ax.plot([x for a,x in ordered],[by[a][-1][key]*scale for a,x in ordered],color='#aaa',lw=1)
        for arm,x in ordered:ax.scatter([x],[by[arm][-1][key]*scale],s=45,color=COLORS[arm],label=LABELS[arm],zorder=3)
        ax.axhline(by['real_0'][0][key]*scale,color=COLORS['s0'],ls='--',lw=1,label='S0 reference')
        ax.set(title=title,xlabel='Synthetic share of training blocks (%)',ylabel=ylabel,xticks=[0,20,50,100])
    save(fig,axes,out/'figures/endpoints_vs_synthetic_fraction.png')
    write_provenance(PROJECT,out,__file__,inputs|{RECORDS/'recovery_manifest.json',PROJECT/'scripts/analysis/square_panel_figures.py'},scope=summary['scope'])
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recover',action='store_true');p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.recover:recover()
    s=analyse(args.out);print(json.dumps({'passed':s['passed'],'endpoint_ppl':{k:v['test_perplexity'] for k,v in s['endpoint_metrics'].items()},'paired':s['paired_test_blocks']},indent=2))
