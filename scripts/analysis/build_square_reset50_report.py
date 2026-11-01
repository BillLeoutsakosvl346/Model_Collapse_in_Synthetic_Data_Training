"""Create a new square-panel presentation revision without altering evidence."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from model_collapse.common import PROJECT, read_json
sys.path.insert(0,str(PROJECT))
from comparisons.mixed_fraction_final5.analyse_continued_fractions import read_csv
from comparisons.continued_mixed50_vs_synthetic_final5.analyse import probe_statistics
from scripts.analysis.build_reset50_comparison_report import ROOT, draft, markdown
from scripts.analysis.square_panel_figures import make_figures
from scripts.analysis.publication import new_destination, verified_publication, write_provenance


def build(out):
    out=new_destination(ROOT,out);verified_publication(ROOT);out.mkdir(parents=True)
    rows=read_csv(ROOT/'analysis/metrics.csv');hist=read_csv(ROOT/'analysis/training_history.csv')
    by={a:[r for r in rows if r['arm']==a] for a in ('continued','reset')}
    roots={'continued':PROJECT/'reports/20261031T163428Z-full-continued/records','reset':ROOT/'records'}
    seqs={};within={};inputs={ROOT/'analysis/metrics.csv',ROOT/'analysis/training_history.csv',ROOT/'analysis/summary.json',ROOT/'analysis/examples.json'}
    for arm,root in roots.items():
        seqs[arm]={};within[arm]={}
        for g in range(11):
            pp=root/f'probes/gen_{g:02d}.npy';sp=root/f'seed_42/gen_{g:02d}/samples.json';tp=root/f'seed_42/gen_{g:02d}/test_sequences.json'
            inputs.update((pp,sp,tp))
            seqs[arm][g]=np.asarray(read_json(tp)['sequence_mean_nll'])
            within[arm][g]=probe_statistics(np.load(pp,allow_pickle=False),read_json(sp))[1]
    figures=make_figures(out/'figures',by,hist,seqs,within,shared=True)
    d=draft(ROOT/'analysis');e={p['id']:p for p in d['entries']}
    def add(i,text,kind='body',page=False):e[i]={'id':i,'text':text,'kind':kind,'page':page}
    e['curves_image']['text']='prediction_and_phrases'
    e['curves_caption']['text']='Figure 12a. Larger square panels: matched real-test/validation perplexity (A/B), pooled distinct-2/3 (C/D). Teal: continued; purple: fixed-S0 reset; grey: exact shared S0/S1. Measurements and source data are unchanged.'
    add('curves_b','2b. Generated-text diversity','heading2',True)
    add('entropy_image','entropy_and_repetition','image')
    add('entropy_caption','Figure 12b. Pooled token entropy (E) and mean within-output four-gram repetition (F), on the same fixed 256-output probe. These describe emitted text, separately from real-test prediction.','caption')
    e['training_image']['text']='real_text_and_training'
    e['diagnostics_image']['text']='repetition_and_formatting'
    e['diagnostics_caption']['text']='Figure 14a. Larger square panels: pooled four-gram repetition (A), within-output entropy (B), formatting-token shares (C) and the final within-output repetition distribution (D). Same inputs, arithmetic and colour mapping as the earlier report.'
    add('reference_heading','5b. Frozen-S0 predictability','heading2',True)
    add('reference_image','frozen_s0_scores','image')
    add('reference_caption','Figure 14b. Frozen-S0 continuation NLL: mean (E) and P95 (F). Each continuation is scored alone, without its original prompt. Larger square panels change the presentation, not the metric or the underlying scores.','caption')
    e['conclusion_heading'].update(text='7. Conclusions, accounting and limitations',kind='heading2',page=True)
    order=['title','meta','methods','objective','shared','difference','sampling','methods_caption','methods_table',
        'curves','curves_image','curves_caption','ppl','cause',
        'curves_b','entropy_image','entropy_caption','rank','definitions','distinct','entropy',
        'statistics','stats_intro','endpoint_caption','endpoint_table',
        'training','training_image','training_caption','coverage','epochs','overfit',
        'diagnostics','diagnostics_image','diagnostics_caption','repetition','format',
        'reference_heading','reference_image','reference_caption','frozen',
        'examples','prompt_label','prompt','examples_caption','examples_table',
        'conclusion_heading','conclusion','mechanism','limits','exposures','recovery','operations','source']
    d['entries']=[e[i] for i in order];d['images']=figures
    d['source_url']='https://github.com/BillLeoutsakosvl346/Model_Collapse_in_Synthetic_Data_Training/blob/main/reports/presentation_revisions/2026-11-01_square-reset50/report.md'
    (out/'draft.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'report.md').write_text(markdown(d,out),encoding='utf-8')
    write_provenance(PROJECT,out,__file__,inputs|{PROJECT/'scripts/analysis/square_panel_figures.py',PROJECT/'scripts/analysis/build_reset50_comparison_report.py',
        PROJECT/'comparisons/mixed_fraction_final5/analyse_continued_fractions.py',PROJECT/'comparisons/continued_mixed50_vs_synthetic_final5/analyse.py'},scope='Presentation-only square-panel revision; original reports, statistics, figures and source records unchanged. No cloud requests.')
    print(json.dumps({'figures':len(figures),'entries':len(order),'all_panels_box_aspect':1}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);build(p.parse_args().out)
