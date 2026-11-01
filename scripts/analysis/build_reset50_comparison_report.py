"""Author the matched 50/50 reset report from frozen, audited observations.

Writes a separate Markdown report and native-Docs draft. No cloud or model calls.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from model_collapse.common import PROJECT, read_json
try:
    from .publication import new_destination, write_provenance
except ImportError:
    from publication import new_destination, write_provenance

RUN = '20261101T030306Z-full-mixed-campaign-reset-mixed50'
ROOT = PROJECT / 'reports' / RUN


def draft(analysis):
    s = read_json(analysis/'summary.json')
    assert s['passed'] and s['shared_s0_s1_d2_m2_verified']
    p = s['endpoint_metrics']; base, shared, c, a = [p[k] for k in ('s0','shared_s1','continued','reset')]
    examples = read_json(analysis/'examples.json')
    entries = []
    def add(identifier, text, kind='body', page=False):
        entries.append({'id':identifier,'text':text,'kind':kind,'page':page})
    add('title','Week 3, Part 3: Matched 50/50 reset and continued training','heading1',True)
    add('meta','OPT-125M / WikiText-2; seed 42. Reset cycles 2–10 completed 1 November 2026; analysis uses its sealed private backup.','small')
    add('methods','1. Objective and shared-start methodology','heading2')
    add('objective','Does resetting to a fixed real-trained model limit recursive deterioration when half of every training dataset is real text? We compare the completed 50/50 continued trajectory with a new reset branch sharing its exact first update.')
    add('shared','Shared beginning: S0 generates D1; train S0 on M1 = mix(real, D1) for five epochs to obtain S1; S1 generates D2. Both arms reuse exactly the same S0, S1, D2 and cycle-2 mixture M2. Their first jump is therefore identical, rather than a separately repeated training run.','bullet')
    add('difference','From cycle 2: continued initializes from its previous epoch-5 checkpoint; reset initializes from fixed real-trained S0. Each model then generates its own next synthetic pool. “Reset” refers to training initialization, not replacing the recursively generated data with S0 outputs.','bullet')
    add('sampling','Mixtures: sample 18,973 real and 18,973 synthetic blocks without replacement from each source, then shuffle. Cycle-specific seeded streams match across arms; mixtures stay fixed through all five epochs. Real blocks may recur across cycles. Synthetic targets contain continuations only, with no prompts or held-out real text.','bullet')
    add('methods_caption','Table 18. Matched scientific settings. Only the checkpoint used to initialize cycles 2–10 differs by design; later synthetic targets consequently diverge.','caption')
    add('methods_table','methods','table')
    add('curves','2. Matched trajectories: prediction and diversity','heading2',True)
    add('curves_image','trajectories','image')
    add('curves_caption','Figure 12. Teal: 50/50 continued; purple: reset to S0; grey: exact shared S0/S1. A/B evaluate frozen real text; C–F describe the same 256-output probe. Distinct-n and repeated phrases use token IDs and windows within outputs. No independent-run error bars.','caption')
    add('ppl',f"Real-text prediction: common test PPL rises from {base['test_perplexity']:.2f} to {shared['test_perplexity']:.2f}. Reset then stays near 37.1–37.3, ending at {a['test_perplexity']:.2f}; continued rises every cycle to {c['test_perplexity']:.2f}. Reset ends {s['reset_final_ppl_reduction_percent']:.2f}% lower than continued, but {s['reset_ppl_above_s0_percent']:.2f}% above S0.",'bullet')
    add('cause','Likely explanation: resetting removes accumulated parameter changes, while recurring real targets anchor each new update. Data feedback still persists through each generated pool. A stable held-out PPL can therefore coexist with changing or worsening generated output; it does not establish that collapse has stopped.','bullet')
    add('rank','Diversity does not follow the PPL ranking. Reset ends with lower distinct-2/3 and entropy, and more within-output repetition. Continued entropy partly rebounds as standalone-space concentration falls from 21.17% at cycle 4 to 9.02% at cycle 10; its real-test PPL nevertheless keeps rising.','bullet')
    add('statistics','3. Endpoint statistics','heading2',True)
    add('stats_intro','Lower real-text NLL/PPL means less surprise at actual held-out tokens. Output diversity describes generated strings; it measures neither factual correctness nor the same prediction task.')
    add('endpoint_caption','Table 19. Common S0 and final cycle-10 checkpoints. NLL and entropy use natural logarithms; percentages are token/window fractions. Duplicate outputs count copies beyond the first among 256 continuations. Shared S1 test PPL is 37.27.','caption')
    add('endpoint_table','endpoints','table')
    add('definitions','Reading the diversity statistics','heading3')
    add('distinct','Distinct-n = unique n-token phrases / all n-token windows, pooled across outputs. Both arms have the same denominators: 16,128 bigrams, 15,872 trigrams and 15,616 four-grams. Lower reset values mean fewer different phrase types in this fixed sample, even though it predicts held-out real text better.','bullet')
    add('entropy','Token entropy H = −Σ p(t) ln p(t) describes how evenly emitted token IDs are used. Within-output entropy computes this separately for each continuation, then averages. These are empirical output-frequency statistics, not the model’s predictive uncertainty.','bullet')
    add('training','4. Real-text coverage and five-epoch training','heading2',True)
    add('training_image','training','image')
    add('training_caption','Figure 13. A: cumulative final block-NLL distributions, with S0 dashed; curves farther left assign higher probability to real text. B: continued minus reset NLL for each identical test block; positive favours reset. C/D: cycle-10 losses over five epochs. The terminal checkpoint is always retained.','caption')
    blocks=s['paired_test_blocks']
    add('coverage',f"Broad descriptive advantage: reset has lower final NLL on {blocks['reset_lower']:,}/4,494 test blocks ({100*blocks['reset_lower']/4494:.2f}%), continued on {blocks['continued_lower']}, with no ties. Mean gap is {blocks['mean_continued_minus_reset_nll']:.3f} nats/target. Reset remains worse than S0 on 3,973 blocks; continued on 4,389.",'bullet')
    add('epochs','Epoch selection: every unique stage executed all five epochs and retained epoch 5. Epoch 1 had the lowest real-validation loss in all 19 unique stages. In cycle 10, validation NLL rises 3.569 → 3.619 for reset and 4.018 → 4.130 for continued, while training NLL falls 2.250 → 1.976 and 1.539 → 1.337, respectively.','bullet')
    add('overfit','Interpretation: fitting the current mixture more strongly does not improve generalization here. Continued also starts already adapted to earlier synthetic targets; reset starts at S0. Their training losses involve different arm-specific mixtures, so lower continued training loss is not evidence of better general language modelling.','bullet')
    add('exposures','Accounting: continued has ten new updates / fifty epochs; this reset branch adds nine updates / forty-five epochs and imports S1. The matched pair contains 19 unique stages / 95 epochs, not twenty independent stages. Each endpoint’s recursive lineage has ten cycles, but reset A10 inherits only its own five-epoch update from S0.','bullet')
    add('diagnostics','5. Repetition, formatting and the frozen-S0 score','heading2',True)
    add('diagnostics_image','diagnostics','image')
    add('diagnostics_caption','Figure 14. A/B separate pooled phrase repetition from mean within-output entropy. C: final formatting tokens. D: cumulative within-output repetition. E/F: teacher-forced continuation NLL under frozen S0, reporting mean and 95th percentile over the same 256 outputs.','caption')
    add('repetition','Repetition: pooled repeated-4 = 1 − distinct-4, so it is not independent evidence. Within-output repeated-4 instead counts excess occurrences inside each continuation before averaging. Reset is more repetitive on 131 paired prompts, continued on 88, with 37 ties; lower reset PPL does not imply less looping.','bullet')
    add('format','Formatting: newlines are only 0.27% (reset) and 0.31% (continued), with no newline-heavy outputs. Excluding them leaves the entropy and pooled repetition rankings unchanged. Standalone spaces remain about 9% in both, with 26 space-heavy outputs each; reset has one whitespace-only decoded output, continued none. Blank-line concentration does not explain this endpoint contrast.','bullet')
    add('frozen','Frozen-S0 score: feed each generated continuation alone to S0 and average −ln p(next token | preceding continuation tokens), over 63 targets. The original prompt is omitted. Reset has lower mean NLL (1.470 vs 1.562) but higher P95 (3.333 vs 3.139): typical predictability and the upper tail can rank differently. Easy repetitive strings can score well; this is not model distance, factuality or real-test performance.','bullet')
    add('examples','6. Saved examples, conclusion and limitations','heading2',True)
    add('prompt_label','Fixed anchor: prompt 0, retained from earlier comparisons before inspecting these outcomes. Generated factual claims below are unverified; the example is not a representative quality assessment.')
    add('prompt',examples[0]['prompt'],'quote')
    add('examples_caption','Table 20. Complete saved 64-token continuations for the same prompt. Here continued loops more strongly (86.89% within-output repeated-4 versus reset 49.18%), despite reset being more repetitive on average. Original token IDs and whitespace remain in the saved records.','caption')
    add('examples_table','examples','table')
    add('conclusion_heading','Conclusions and scope','heading3')
    add('conclusion','Main finding: with 50/50 mixing, fixed-S0 reset largely stabilizes real-test prediction after the first update, while continued training deteriorates. It does not preserve S0 performance or improve all diversity measures. Both still generate loops, and the reset probe has lower phrase diversity.','bullet')
    add('mechanism','A plausible reason for the diversity trade-off is that repeated resetting and deterministic beam decoding favour familiar high-probability templates, while data feedback progressively narrows them. This is a hypothesis: the observations establish the trade-off, not that specific mechanism.','bullet')
    add('limits','Limits: one seed, OPT-125M, WikiText-2, ten cycles and five-beam decoding. The fixed probe covers the first 256 consecutive training prompts from six adjacent articles, not a random corpus sample. Paired test blocks are descriptive observations, not independent model runs. Reused real text is not fresh external data; no universal safe synthetic fraction is established.','bullet')
    add('recovery','Evidence: checksum-verified 1,421 metadata files / 53,699,537 bytes from pinned backups, plus nine new 256×64 probes / 590,976 bytes; shared S0/S1 and the continued evidence were reused. All 22 metric rows, twenty unique probes, 100 saved epoch entries (95 unique), mixture-selection hashes and recorded handoffs were audited. No full weights/training arrays were downloaded or models re-evaluated.','small')
    add('operations',f"Execution scope: recorded new reset training time {s['runtime']['new_reset_training_seconds']/3600:.2f} h and generation time {s['runtime']['new_reset_generation_seconds']/3600:.2f} h. This is one completed job in a larger campaign, not a separately shut-down Pod or itemized invoice. Analysis used Hugging Face backups only: no Runpod API calls, SSH sessions or changes to the running campaign.",'small')
    add('source','Per-cycle metrics, epoch histories, matched outputs, sealed-source records and reproducible analysis.','link')
    specs=[('Test corpus perplexity','test_perplexity','.2f',1),('Validation perplexity','validation_perplexity','.2f',1),
        ('Test NLL (nats/target)','test_nll','.3f',1),('P95 test-block NLL','test_block_nll_p95','.3f',1),
        ('Distinct-1 (%)','distinct_1','.2f',100),('Distinct-2 (%)','distinct_2','.2f',100),('Distinct-3 (%)','distinct_3','.2f',100),
        ('Distinct-4 (%)','distinct_4','.2f',100),('Pooled repeated 4-grams (%)','repeated_4gram_rate','.2f',100),
        ('Mean within-output repeated-4 (%)','mean_within_repeated_4gram_rate','.2f',100),
        ('Duplicate outputs / 256','duplicate_outputs','.0f',1),('Pooled entropy (nats)','token_entropy_nats','.3f',1),
        ('Mean within-output entropy','mean_within_entropy_nats','.3f',1),('Frozen-S0 mean NLL','frozen_s0_nll_mean','.3f',1),
        ('Frozen-S0 P95 NLL','frozen_s0_nll_p95','.3f',1),('Newline tokens (%)','newline_token_50118_share','.2f',100),
        ('Standalone-space tokens (%)','standalone_space_share','.2f',100),('Whitespace-only outputs / 256','whitespace_only_decoded_outputs','.0f',1)]
    tables={'methods':[['Setting','Both 50/50 arms'],
        ['Model / corpus','Pinned facebook/opt-125m / WikiText-2 raw; unchanged row/split packing'],
        ['Shared prefix','Exact real-trained S0, mixed-trained epoch-5 S1, D2 and M2; imported, not retrained'],
        ['Mixture per cycle','18,973 real + 18,973 synthetic blocks × 64 tokens; 2,428,544 total input tokens'],
        ['Generation','Same real 64-token prompts → exactly 64 new tokens; five beams, penalty 1.0; full pool before mixing'],
        ['Training / selection','Five epochs; always retain epoch 5. Fresh AdamW/cosine each cycle; LR 2e-5, seed 42, effective batch 128, BF16'],
        ['Evaluation / probe','3,922 validation and 4,494 test real blocks; fixed 256-output probe; 63 shifted scored targets per block'],
        ['Recorded environment','RTX PRO 4000 Blackwell; PyTorch 2.8.0+cu128, Transformers 4.37.2, Datasets 3.4.1; matching driver/software']],
        'endpoints':[['Measure','S0','Continued C10','Reset A10']]+[[label,*[format(r[key]*scale,fmt) for r in [base,c,a]]] for label,key,fmt,scale in specs],
        'examples':[['Checkpoint','Complete saved continuation'],['Shared S1',examples[0]['shared_s1']],
                    ['Continued C10',examples[0]['outputs']['continued']['text']],['Reset A10',examples[0]['outputs']['reset']['text']]]}
    images={k:str((analysis/'figures'/name).resolve()) for k,name in
            [('trajectories','matched_trajectories.png'),('training','training_and_real_text.png'),('diagnostics','repetition_formatting_and_s0.png')]}
    return {'entries':entries,'tables':tables,'images':images,
        'source_url':f'https://github.com/BillLeoutsakosvl346/Model_Collapse_in_Synthetic_Data_Training/blob/main/reports/{RUN}/report.md'}


def markdown(d,directory):
    def link(path):return Path(os.path.relpath(Path(path).resolve(),Path(directory).resolve())).as_posix()
    lines=[]
    for e in d['entries']:
        kind,text=e['kind'],e['text']
        if kind=='table':
            rows=d['tables'][text]
            lines+=['| '+' | '.join(rows[0])+' |','| '+' | '.join(['---']*len(rows[0]))+' |']
            lines+=['| '+' | '.join(v.replace('\n','\\n').replace('|','\\|') for v in row)+' |' for row in rows[1:]]
        elif kind=='image':lines+=[f'![Matched 50/50 comparison: {text}]({link(d["images"][text])})']
        elif kind=='bullet':lines+=['- '+text]
        elif kind=='quote':lines+=['> '+text.replace('\n','\\n')]
        elif kind=='link':lines+=[f'[Per-cycle metrics]({link(ROOT/"analysis/metrics.csv")}) | [Epoch histories]({link(ROOT/"analysis/training_history.csv")}) | [Examples]({link(ROOT/"analysis/examples.json")}) | [Audit and provenance]({link(ROOT/"analysis/publication_reproduction.json")}) | [Analysis script]({link(PROJECT/"comparisons/mixed_fraction_final5/analyse_reset50.py")})']
        else:lines+=[{'heading1':'# ','heading2':'## ','heading3':'### '}.get(kind,'')+text]
        lines+=['']
    return '\n'.join(lines)


def build(analysis,out):
    analysis=Path(analysis).resolve();out=new_destination(ROOT,out);d=draft(analysis)
    out.mkdir(parents=True,exist_ok=True)
    (out/'draft.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'report.md').write_text(markdown(d,out),encoding='utf-8')
    write_provenance(PROJECT,out,__file__,[analysis/'summary.json',analysis/'examples.json',analysis/'publication_reproduction.json',*map(Path,d['images'].values())],scope='Authored matched 50/50 reset/continued report and native-Docs draft; no model or cloud operations.')
    print(json.dumps({'entries':len(d['entries']),'tables':len(d['tables']),'figures':len(d['images']),
                      'prose_words':sum(len(e['text'].split()) for e in d['entries'] if e['kind'] not in ('table','image'))}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,default=ROOT/'analysis');p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();build(args.analysis,args.out)
