"""Create a new human-readable analysis of the saved gradient diagnostic.

Usage: python scripts/analysis/interpret_gradient_diagnostic.py --run runs/ID --out reports/ID/generated/ANALYSIS
No cloud access or model evaluation is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model_collapse.common import atomic_json, read_json, sha256
from model_collapse.analysis.gradients import report


def build(run, out, layers=None):
    run, out = Path(run), Path(out)
    from model_collapse.artifacts.hub import verify_recovered_files
    from model_collapse.science.gradients import paired_change
    import numpy as np
    index = read_json(run/'artifact_index.json')
    if not index: raise RuntimeError('Fetch a pinned diagnostic index before analysis')
    verify_recovered_files(run,index['files'])
    rendered = report(run, out)
    gradients = read_json(out / "gradients.json")
    updates = read_json(out / "updates.json")
    counterfactuals, extra_inputs = [], {}
    for row in gradients:
        if row['arm']!='mixed99':continue
        folder=run/'cases'/f"mixed99-cycle{row['cycle']:02d}"
        with np.load(folder/'probe_before.npz') as source:
            counts=source['counts'];documents=source['documents']
        for rate in ('historical','common'):
            paths=[folder/f'{rate}-{name}-probe.npz' for name in ('synthetic_only','mixed')]
            before,after=[np.load(p)['nll_sum'] for p in paths]
            counterfactuals.append({'cycle':row['cycle'],'learning_rate_policy':rate,
                                   **paired_change(before,after,counts,documents)})
            for p in [folder/'probe_before.npz',*paths]:extra_inputs[p.relative_to(run).as_posix()]=sha256(p)
    atomic_json(out/'mixed_vs_synthetic.json',counterfactuals)
    import matplotlib.pyplot as plt
    more = []
    fig, ax = plt.subplots(figsize=(7.5, 7.5)); ax.set_box_aspect(1)
    for rate, color, label in [('historical','#147d81','Recorded learning rate'),('common','#8b5fbf','Common learning rate: 3e-6')]:
        rows=sorted((r for r in counterfactuals if r['learning_rate_policy']==rate),key=lambda r:r['cycle'])
        ax.plot([r['cycle'] for r in rows],[r['delta_nll'] for r in rows],'o-',color=color,label=label)
        ax.vlines([r['cycle'] for r in rows],[r['delta_nll_document_bootstrap_95'][0] for r in rows],
                  [r['delta_nll_document_bootstrap_95'][1] for r in rows],color=color,alpha=.4)
    ax.axhline(0,color='black',lw=1)
    ax.set(xlabel='Before training cycle',ylabel='Mixed-step NLL minus synthetic-only-step NLL',
           title='Does the 1% real component improve the update?\nSame 99% checkpoint and AdamW state')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();name='06_effect_of_real_injection'
    fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.pdf'),metadata={'CreationDate':None,'ModDate':None});plt.close(fig);more.append(name+'.png')
    mixed_rows=sorted((r for r in gradients if r['arm']=='mixed99'),key=lambda r:r['cycle'])
    if mixed_rows:
        fig,ax=plt.subplots(figsize=(7.5,7.5));ax.set_box_aspect(1)
        for key,label,color in [('weighted_real_norm','1% × real gradient','#147d81'),('weighted_synthetic_norm','99% × synthetic gradient','#ad355c'),('mixed_norm','Resulting mixed gradient','#475569')]:
            ax.plot([r['cycle'] for r in mixed_rows],[r[key] for r in mixed_rows],'o-',label=label,color=color)
        ax.set_yscale('log');ax.set(xlabel='Before training cycle',ylabel='Gradient L2 norm (log scale)',title='Actual 99% mixture: component strengths')
        ax.grid(alpha=.2);ax.legend();fig.tight_layout();name='05_weighted_gradient_strength_log'
        fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.pdf'),metadata={'CreationDate':None,'ModDate':None});plt.close(fig);more.append(name+'.png')
    for arm in ("mixed99", "synthetic"):
        fig, ax = plt.subplots(figsize=(7.5, 7.5)); ax.set_box_aspect(1)
        for name, color in (("real_only", "#147d81"), ("synthetic_only", "#ad355c"), ("mixed", "#8b5fbf")):
            rows = sorted((r for r in updates if r["arm"] == arm and r["action"] == name and r["learning_rate_policy"] == "historical"), key=lambda r:r["cycle"])
            if not rows: continue
            values = [r["relative_to_optimizer_history"] for r in rows]
            ax.plot([r["cycle"] for r in rows], [r["delta_nll"] for r in values], "o-", label=name.replace("_", " "), color=color)
            ax.vlines([r["cycle"] for r in rows], [r["delta_nll_document_bootstrap_95"][0] for r in values],
                      [r["delta_nll_document_bootstrap_95"][1] for r in values], color=color, alpha=.35)
        ax.axhline(0, color="black", lw=1)
        ax.set(xlabel="Before training cycle", ylabel="Validation NLL relative to optimizer-history step",
               title=f"{'99%' if arm=='mixed99' else '100%'} checkpoints: incremental data-gradient effect")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3,3)); ax.grid(alpha=.2); ax.legend()
        fig.tight_layout(); name=f"04_{arm}_incremental_data_effect"
        fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.pdf'),metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
        more.append(name+'.png')
    summary=[]
    for g in gradients:
        item={"arm":g["arm"],"before_cycle":g["cycle"],"checkpoint":'M'+str(g['checkpoint_cycle']),
              "real_mean_gradient_norm":g['real_vs_synthetic']['norm'],"synthetic_mean_gradient_norm":g['real_vs_synthetic']['other_norm'],
              "weighted_real_norm":g['weighted_real_norm'],"weighted_synthetic_norm":g['weighted_synthetic_norm'],
              "actual_source_cosine":g['real_vs_synthetic']['cosine'],"matched_prompt_cosine":g['matched_real_vs_synthetic']['cosine'],
              "matched_prompt_real_mean_gradient_norm":g['matched_real_vs_synthetic']['norm'],
              "real_source_nll":g['real_source']['loss'],"synthetic_source_nll":g['synthetic_source']['loss'],
              "source_real_pairs":g['real_source']['pairs'],"source_synthetic_pairs":g['synthetic_source']['pairs'],
              "historical_learning_rate":g['original_learning_rate']}
        for action in ['optimizer_history','real_only','synthetic_only','mixed']:
            r=next((r for r in updates if r['arm']==g['arm'] and r['cycle']==g['cycle'] and r['action']==action and r['learning_rate_policy']=='historical'),None)
            if r:
                item[action+'_probe_delta_nll']=r['delta_nll']
                item[action+'_incremental_delta_nll']=r['relative_to_optimizer_history']['delta_nll']
                if 'full_validation' in r:item[action+'_full_validation_delta_nll']=r['full_validation']['delta_nll']
        if g['arm']=='mixed99':
            item['mixed_vs_synthetic_delta_nll']=next(r['delta_nll'] for r in counterfactuals if r['cycle']==g['cycle'] and r['learning_rate_policy']=='historical')
        summary.append(item)
    atomic_json(out/'summary.json',summary)
    layer_rows=[]
    directional_rows=[]
    if layers:
        layer_root=Path(layers)
        verify_recovered_files(layer_root,read_json(layer_root/'artifact_index.json')['files'])
        for path in sorted((layer_root/'layers').glob('*.json')):
            item=read_json(path)
            original=next(r for r in gradients if r['arm']==item['arm'] and r['cycle']==item['cycle'])
            if item['weights_sha256']!=original['weights_sha256'] or item['global_norm_relative_error']>1e-5:
                raise RuntimeError('Layer supplement differs from primary measurements')
            for name,value in item['synthetic_geometry']['parameters'].items():
                real=item['real_parameter_norms'][name]
                layer_rows.append({'arm':item['arm'],'cycle':item['cycle'],'parameter':name,'real_norm':real,
                    'matched_real_norm':item['matched_real_parameter_norms'][name],'synthetic_norm':value,
                    'actual_weighted_real_norm':.01*real if item['arm']=='mixed99' else 0.,
                    'actual_weighted_synthetic_norm':(.99 if item['arm']=='mixed99' else 1.)*value})
            for direction in item.get('directional_interventions',[]):
                array_path=layer_root/'directional_arrays'/f"{item['arm']}-cycle{item['cycle']:02d}-{direction['action']}.npz"
                with np.load(array_path) as arrays, np.load(run/'cases'/f"{item['arm']}-cycle{item['cycle']:02d}"/'probe_before.npz') as primary_probe:
                    uncertainty=paired_change(arrays['before_nll_sum'],arrays['after_nll_sum'],arrays['counts'],primary_probe['documents'])
                    direction={**direction,'observed_validation_delta_nll_document_bootstrap_95':uncertainty['delta_nll_document_bootstrap_95']}
                extra_inputs['layer_supplement/directional_arrays/'+array_path.name]=sha256(array_path)
                directional_rows.append({'arm':item['arm'],'cycle':item['cycle'],
                    'synthetic_validation_gradient_cosine':item['synthetic_vs_validation_gradient']['cosine'],**direction})
            extra_inputs['layer_supplement/'+path.name]=sha256(path)
        with (out/'per_parameter_norms.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(layer_rows[0]));writer.writeheader();writer.writerows(layer_rows)
        if directional_rows:
            with (out/'directional_effects.csv').open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(directional_rows[0]));writer.writeheader();writer.writerows(directional_rows)
            atomic_json(out/'directional_effects.json',directional_rows)
        import re
        fig,ax=plt.subplots(figsize=(7.5,7.5));ax.set_box_aspect(1)
        for cycle,color in [(1,'#147d81'),(5,'#8b5fbf'),(10,'#ad355c')]:
            groups={}
            for row in layer_rows:
                match=re.search(r'(?:^|\.)layers\.(\d+)\.',row['parameter'])
                if row['arm']!='mixed99' or row['cycle']!=cycle or not match:continue
                totals=groups.setdefault(int(match[1]),[0.,0.])
                totals[0]+=row['actual_weighted_real_norm']**2
                totals[1]+=row['actual_weighted_synthetic_norm']**2
            indices=sorted(groups)
            ax.plot(indices,[(groups[n][0]/groups[n][1])**.5 for n in indices],'o-',color=color,label=f'Before cycle {cycle}')
        ax.axhline(1,color='black',lw=1,ls='--');ax.set_yscale('log')
        ax.set(xlabel='Transformer layer (zero-indexed)',ylabel='Weighted real / synthetic gradient norm',
               title='Actual 99% mixture: relative gradient strength by layer')
        ax.grid(alpha=.2);ax.legend();fig.tight_layout();name='07_layer_gradient_strength_ratio'
        fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.pdf'),metadata={'CreationDate':None,'ModDate':None});plt.close(fig);more.append(name+'.png')
        if directional_rows:
            fig,ax=plt.subplots(figsize=(7.5,7.5));ax.set_box_aspect(1)
            for arm,label,color in [('mixed99','99% checkpoints','#147d81'),('synthetic','100% checkpoints','#ad355c')]:
                for action,style,suffix in [('synthetic_only','o-','saved momentum'),('synthetic_without_momentum','s--','first moment zeroed')]:
                    rows=sorted((r for r in directional_rows if r['arm']==arm and r['action']==action),key=lambda r:r['cycle'])
                    ax.plot([r['cycle'] for r in rows],[r['observed_validation_delta_nll'] for r in rows],style,color=color,label=f'{label}, {suffix}')
            ax.axhline(0,color='black',lw=1);ax.grid(alpha=.2)
            ax.set(xlabel='Before training cycle',ylabel='Real-validation NLL change',
                   title='Exploratory: inherited momentum changes the update\nSame synthetic gradient, second moments and step count')
            ax.legend(fontsize=8);fig.tight_layout();name='08_optimizer_momentum_control'
            fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.pdf'),metadata={'CreationDate':None,'ModDate':None});plt.close(fig);more.append(name+'.png')
    fields=list(dict.fromkeys(k for r in summary for k in r))
    with (out/'summary.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(summary)
    lines=['# Why do a few real continuations change recursive training?', '',
           'This is a local intervention study of saved SmolLM2 checkpoints, not a new ten-cycle training experiment.', '',
           '## Method', '',
           '- Restore the original M0 and the 99%/100% conditions’ M4/M9, together with exact cycle-1/5/10 targets and saved AdamW states.',
           '- Calculate complete-source, token-averaged gradients with FP32 arithmetic and dropout disabled. Prompt labels remain masked; every continuation has 64 scored targets.',
           '- Decompose the actual 99% objective into its 100 real and 9,900 synthetic rows. Independently compare original and generated continuations on identical prompt rows.',
           '- Apply one isolated aggregate-gradient update per intervention from the same starting model/optimizer state. These do not replay the historical 128-example minibatches.',
           '- Evaluate a frozen 1,024-pair validation probe; confirm historical-rate mixed/history effects on all 7,447 validation pairs. No validation gradients update the model.',
           '- Compare recorded learning rates with a fixed 3e-6 diagnostic rate. Explicit zero gradients retain AdamW momentum and weight decay as a separate control.', '',
           '## Source-gradient measurements', '',
           '| Condition | Checkpoint | Real source NLL | Synthetic source NLL | 1% × real norm | 99% × synthetic norm | Same-prompt cosine |',
           '|---|---|---:|---:|---:|---:|---:|']
    def fmt(v):return '—' if v is None else f'{v:.6g}'
    for r in summary:
        lines.append('| '+' | '.join([r['arm'],r['checkpoint'],fmt(r['real_source_nll']),fmt(r['synthetic_source_nll']),fmt(r['weighted_real_norm']),
                                     fmt(r['weighted_synthetic_norm']) if r['arm']=='mixed99' else '—',fmt(r['matched_prompt_cosine'])])+' |')
    lines += ['', 'The 100% condition has no actual real training component. Its real-only diagnostic uses the original real targets on the same 10,000 prompts. Its unweighted synthetic norm remains available in the raw records.', '',
              '## Immediate effects at the historical learning rate', '',
              'All entries below are changes in real-validation NLL. Negative values improve predictions; positive values worsen them.', '',
              '| Condition | Checkpoint | History-only | Real-only | Synthetic-only | Mixed | Mixed minus synthetic |',
              '|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        keys=['optimizer_history_probe_delta_nll','real_only_probe_delta_nll','synthetic_only_probe_delta_nll','mixed_probe_delta_nll','mixed_vs_synthetic_delta_nll']
        lines.append('| '+' | '.join([r['arm'],r['checkpoint']]+[fmt(r.get(k)) for k in keys])+' |')
    lines += ['', 'The difference from the history-only step matters: an update can improve on the optimizer’s inherited direction while still worsening loss relative to taking no step.', '',
              '## Reading the evidence', '',
              '- Compare the real and synthetic update directions and held-out effects together. A large gradient is not automatically useful.',
              '- The actual source groups differ in size, so their mean-gradient norms also reflect cancellation and sampling variability. The matched-prompt comparison is a separate, equal-row check.',
              '- Gradient components add linearly before AdamW. Their separate AdamW updates do not add linearly, because normalization and saved moments intervene.',
              '- Small or sign-changing effects are retained. Document-bootstrap intervals describe this validation sample, not training-seed uncertainty.',
              '- Six checkpoint cases from one historical seed do not establish a universal mixture threshold, permanent equilibrium or a complete explanation of model collapse.', '',
              ]
    if layers:
        lines += ['', '## Supplementary mechanism check — exploratory', '',
                  'Per-parameter real, matched-real and synthetic norms are in [per_parameter_norms.csv](per_parameter_norms.csv). The synthetic breakdown was recovered in a separately accepted pass, with global norms/losses checked against the primary measurements.', '',
                  'After positive real/synthetic alignment was observed at the middle checkpoint, we added a diagnostic validation gradient. It is never supplied to an optimizer. We project the actual synthetic/mixed AdamW displacement onto that validation gradient: `first-order loss change = validation_gradient · weight_displacement`.', '',
                  '| Condition | Before cycle | Update | Synthetic/validation gradient cosine | First-order NLL change | Observed NLL change | Nonlinear residual |',
                  '|---|---:|---|---:|---:|---:|---:|']
        for r in directional_rows:
            lines.append('| '+' | '.join([r['arm'],str(r['cycle']),r['action'],fmt(r['synthetic_validation_gradient_cosine']),
                fmt(r['first_order_validation_delta_nll']),fmt(r['observed_validation_delta_nll']),fmt(r['nonlinear_residual_nll'])])+' |')
        lines += ['', 'Positive first-order values mean that the actual AdamW movement points toward worse held-out predictions locally. The residual includes higher-order loss changes and finite-precision effects. Raw gradient alignment and AdamW displacement alignment are different quantities. These additions are exploratory rather than preregistered confirmation.', '']
    lines += ['', '## Figures', '']
    for name in rendered['figures']+more:lines += [f'![{name}]({name})','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    manifest=read_json(out/'analysis_manifest.json');manifest['interpretation_script_sha256']=sha256(__file__)
    manifest['inputs'].update(extra_inputs)
    manifest['outputs']={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='analysis_manifest.json'}
    atomic_json(out/'analysis_manifest.json',manifest)
    print(json.dumps({'report':str(out/'report.md'),'summary':summary},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--out',required=True);p.add_argument('--layers')
    build(**vars(p.parse_args()))
