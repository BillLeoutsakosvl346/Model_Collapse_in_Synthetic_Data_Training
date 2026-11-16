"""Offline figures for independent whole-epoch source-removal trajectories."""
from __future__ import annotations

import csv
import json
import math
import platform
from pathlib import Path

from ..common import atomic_json, read_json, sha256, verify_files


def report(root, out=None, *, for_backup=False):
    root=Path(root); out=Path(out) if out else root/'generated/epoch-ablation'
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a new analysis destination')
    reference=read_json(root/'reference/manifest.json')
    if not reference:
        raise RuntimeError('Pinned historical plotting references are missing')
    verify_files(root/'reference',reference['files'])
    recovery=read_json(root/'source_recovery.json')
    if reference['source_snapshot']!=recovery['child']['snapshot']:
        raise RuntimeError('Historical plotting source differs from restored snapshot')
    for name,original in reference['source_files'].items():
        if reference['files'][name]!=recovery['child']['files'][original]['sha256']:
            raise RuntimeError('Historical plotting reference differs from pinned source hash')
    inputs={str(Path('reference')/p):h for p,h in reference['files'].items()}
    for name in ('config.json','source_recovery.json','reference/manifest.json'):
        inputs[name]=sha256(root/name)
    rows=[]; endpoints=[]
    for cycle in (3,6,10):
        for source in ('both','real','synthetic'):
            branch=root/f'cases/cycle_{cycle:02d}'/source
            done=read_json(branch/'branch_completion.json')
            if not done or not done.get('complete'):
                raise RuntimeError('A source-removal branch is incomplete')
            verify_files(branch,{p:h for p,h in done['files'].items() if not p.endswith(('.pt','.safetensors'))})
            inputs[str((branch/'branch_completion.json').relative_to(root))]=sha256(branch/'branch_completion.json')
            rr=[json.loads(line) for line in (root/done['records']).read_text().splitlines()]
            if len(rr)!=79 or [r['local_step'] for r in rr]!=list(range(1,80)):
                raise RuntimeError('Wrong update timeline')
            rows.extend({'cycle':cycle,**r} for r in rr)
            endpoints.append({'cycle':cycle,'source':source,'probe_nll_before':done['initial_probe']['nll'],
                'probe_nll_after':done['final_probe_nll'],'probe_ppl_after':math.exp(done['final_probe_nll']),
                'validation_nll':done['metrics']['validation']['loss'],
                'validation_ppl':done['metrics']['validation']['perplexity'],
                'test_nll':done['metrics']['test']['loss'],'test_ppl':done['metrics']['test']['perplexity'],
                'relative_parameter_movement':done['movement']['relative_l2']})
    out.mkdir(parents=True,exist_ok=True)
    for name,values in (('per_update.csv',rows),('endpoints.csv',endpoints)):
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles={'both':('#7651af','-','Both: replay of 99% mixture'),
            'real':('#767676','--','Real-only: original 1% weight'),
            'synthetic':('#444444',':','Synthetic-only: original 99% weight')}
    for cycle in (3,6,10):
        for key,ylabel in (('probe_nll','Fixed real-validation probe NLL'),('probe_perplexity','Fixed real-validation probe perplexity')):
            fig,ax=plt.subplots(figsize=(8,8));ax.set_box_aspect(1)
            for source,(color,ls,label) in styles.items():
                rr=[r for r in rows if r['cycle']==cycle and r['source']==source]
                initial=rr[0]['probe_nll_before']; initial=math.exp(initial) if key=='probe_perplexity' else initial
                ax.plot([cycle-1]+[cycle-1+r['local_step']/79 for r in rr],[initial]+[r[key] for r in rr],color=color,ls=ls,label=label,lw=2)
            record_path=root/'reference'/f'updates_{cycle:02d}.jsonl'
            historic=json.loads(record_path.read_text().splitlines()[-1])['states']['mixed']
            ax.scatter([cycle],[historic['loss' if key=='probe_nll' else 'perplexity']],marker='*',s=110,c='black',label='Historical mixed endpoint',zorder=5)
            ax.set(title=f'Cycle {cycle}: one epoch from identical M{cycle-1} weights',xlabel='Training cycle (fraction = update position)',ylabel=ylabel)
            ax.grid(alpha=.2);ax.legend(fontsize=9)
            fig.text(.5,.02,'128 fixed real pairs; branches evolve independently. Original 128/16-row denominators retained.',ha='center',fontsize=8)
            fig.tight_layout(rect=(0,.035,1,1));name=f'cycle_{cycle:02d}_{key}'
            fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.pdf'),metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,8));ax.set_box_aspect(1)
    historical=[read_json(root/'reference'/f'metrics_{i:02d}.json')['test']['perplexity'] for i in range(11)]
    ax.plot(range(11),historical,'o-',color='#7651af',label='Historical 99%: complete real-test PPL')
    for source in ('real','synthetic'):
        color,ls,label=styles[source]
        for j,cycle in enumerate((3,6,10)):
            value=next(r for r in endpoints if r['cycle']==cycle and r['source']==source)['test_ppl']
            ax.plot([cycle-1,cycle],[historical[cycle-1],value],ls=ls,color=color,label=label if j==0 else None)
    ax.set(xlabel='Training cycle',ylabel='Complete fixed real-test perplexity',title='One-epoch forks from the observed 99% trajectory',xticks=range(11));ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.text(.5,.02,'Grey segments join full-test endpoints; they are not measured within-epoch test trajectories.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,1));fig.savefig(out/'timeline.png',dpi=180);fig.savefig(out/'timeline.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    text=['# Whole-epoch removal of real or synthetic contributions','',
          'Independent branches start from M2, M5 and M9. Each consumes its original saved epoch targets and follows 79 batch positions at SGD 0.01. Prompt masks, original batch denominators and the 16-example tail remain fixed. Synthetic targets are not regenerated.','',
          'The probe figures measure all updates on 128 fixed real-validation pairs. Complete validation/test endpoints are separate. Removing a source preserves its original contribution weight; this is not a separately renormalized 100%-real training run.','',
          '| Cycle | Branch | Probe NLL change | Complete test PPL |','| --- | --- | ---: | ---: |']
    for r in endpoints:
        text.append(f"| {r['cycle']} | {r['source']} | {r['probe_nll_after']-r['probe_nll_before']:+.6f} | {r['test_ppl']:.3f} |")
    text+=['','Whole-epoch effects accumulate along different model trajectories and need not add. Cycle 6 was deliberately selected for its historical correction. One seed and these three diagnostic cases do not establish a universal mechanism or permanent stability.','',
           'Both-source replay checks must pass before interpreting the ablations. Initializer, targets, masks, RNG state and source selections are pinned; recovery scope is recorded separately.']
    (out/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    atomic_json(out/'analysis_manifest.json',{'inputs':inputs,'python':platform.python_version(),'script_sha256':sha256(__file__),
                                            'outputs':{p.name:sha256(p) for p in out.iterdir() if p.is_file()}})
    return {'out':str(out),'branches':9,'update_positions':len(rows),'endpoints':endpoints}
