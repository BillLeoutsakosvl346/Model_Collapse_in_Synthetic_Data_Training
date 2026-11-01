"""Build the authored mixed-data comparison and native-Docs draft offline."""
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

RUN = '20261031T163428Z-full-continued'
ROOT = PROJECT / 'reports' / RUN


def draft(analysis):
    s = read_json(analysis/'summary.json')
    assert s['checks_passed']
    rows = s['endpoint_metrics']; b, m, base = rows['synthetic_100'], rows['mixed_50'], rows['s0']
    examples = read_json(analysis/'examples.json')
    receipt = read_json(PROJECT/'docs/validation/2026-11-01_mixed-50-full-completion.json')
    blocks = s['paired_test_blocks']
    entries = []
    def add(identifier, text, kind='body', page=False):
        entries.append({'id':identifier,'text':text,'kind':kind,'page':page})
    add('title','Week 3, Part 2: Real/synthetic data mixing','heading1',True)
    add('meta','OPT-125M / WikiText-2; seed 42. Mixed run completed 31 October 2026; comparison analysed 1 November.','small')
    add('methods','1. Objective and matched methodology','heading2')
    add('objective','We tested whether retaining real text in every recursive training dataset reduces deterioration relative to five-epoch synthetic-only continued training. Both conditions carry the preceding weights and always retain epoch 5; the reset arm is outside this comparison.')
    add('matched','Shared start: exact historical real-trained S0 and its D1 pool. Model/data revisions, optimizer restarts, learning-rate schedule, token budget, evaluation, decoding and recorded GPU/software match. The common training/data engine is unchanged; the new selection contract adds the mixture fraction to its integrity hash.','bullet')
    add('change','Deliberate change: replace half of each synthetic training dataset with randomly selected real training blocks. S1 therefore differs from the first update; only S0 and the initial source pool are shared.','bullet')
    add('method_caption','Table 15. The comparator is the final-epoch-5 continued arm. Equal total training budgets isolate the mixture condition within these seeded trajectories.','caption')
    add('method_table','methods','table')
    add('sampling','Sampling: without replacement within each source/cycle; independently seeded selection/shuffle. Reuse the same finite real split across cycles and freeze each mixture for five epochs/resume. Across ten cycles, 37,909/37,946 real blocks appear at least once.','bullet')
    add('budget','Token accounting: 121,427,200 input-token exposures per arm over fifty epochs. D1 is imported; nine new full pools and a final probe are generated. Prompts are excluded from synthetic targets; validation/test remain held out.','bullet')
    add('curves','2. Prediction and generation trajectories','heading2',True)
    add('metric_image','metrics','image')
    add('metric_caption','Figure 9. Orange: 100% synthetic; teal: 50/50 mixture; grey: common S0. A uses fixed real-test text; B-E describe the fixed 256-output probe, including formatting. C distinguishes bigrams/trigrams; F scores each continuation alone under frozen S0 (solid mean, dashed P95). No independent-run error bars.','caption')
    add('ppl','Real-text prediction: first-update PPL is 54.46 for synthetic-only versus 37.27 for mixing. Final values are 107.37 versus 62.71: mixing is 41.60% lower, but still 85.06% above S0 (33.89). Mixing rises every cycle; synthetic-only first overshoots, partially recovers, then rises.','bullet')
    add('trajectory','Likely interpretation: repeated real targets counteract some synthetic-distribution drift, limiting the initial jump and later deterioration. This is consistent with the matched comparison, not a demonstrated causal mechanism. The full synthetic source pool remains generated each cycle, even though only half is sampled for mixed training.','bullet')
    add('statistics','3. Endpoint statistics and interpretation','heading2',True)
    add('stat_intro','Prediction loss and output diversity answer different questions. Lower held-out NLL/PPL means the model assigns higher probability to real WikiText targets; generated-text statistics describe the text produced for our fixed prompts.')
    add('endpoint_caption','Table 16. Same S0, then the final model in each continued-weight condition. Fractions are reported as percentages; duplicate/blank counts are among 256 outputs.','caption')
    add('endpoint_table','endpoints','table')
    add('distinct','Distinct-n: mixing has fewer unique two-/three-token combinations across the whole probe, despite lower average within-output repetition. Both use identical denominators (16,128 bigram and 15,872 trigram positions). These are different properties, not an arithmetic inconsistency: fewer types can coexist with more balanced token frequencies and less severe loops.','bullet')
    add('reference','Frozen-S0 scores: mixed outputs are less surprising to S0 (mean 1.562 versus 2.238 nats/target), consistent with retaining more familiar patterns. This scores different generated texts, omitting their original prompts; it is not model distance, factuality or real-test accuracy. Predictable repetition can also score well.','bullet')
    add('generalization','4. Where real-text performance deteriorates','heading2',True)
    add('training_image','training','image')
    add('training_caption','Figure 10. A: common-bin distributions of mean real-test block NLL; B: synthetic-only minus mixed final NLL on the same blocks. C/D: cycle 1 (dashed) and 10 (solid) across five epochs; stars retain epoch 5. Training losses use different target datasets and are not interchangeable quality scores.','caption')
    add('coverage',f"Coverage: mixing has lower final loss on {blocks['mixed_lower']:,}/4,494 test blocks ({100*blocks['mixed_lower']/4494:.2f}%), with mean gap {blocks['mean_nll_synthetic_minus_mixed']:.3f} nats/target. This is a descriptive paired-block result, not 4,494 independent training replicates. Mixing remains worse than S0 on 4,389 blocks; synthetic-only on 4,465.",'bullet')
    add('selection','Five-epoch effect: epoch 1 has the lowest real validation loss in all twenty stages, but both conditions intentionally retain epoch 5. In cycle 10, validation NLL rises from 4.018 to 4.130 for mixing and 4.564 to 4.675 for synthetic-only while their training losses fall.','bullet')
    add('mechanism','Likely cause: the models increasingly fit their current, partly or wholly self-generated targets without improving generalization to held-out real text. Real-data mixing reduces that gap here but does not remove it. Different training losses also reflect the mixtures\' different difficulty, not simply better or worse language modelling.','bullet')
    add('formatting','5. Diversity, repetition and formatting failures','heading2',True)
    add('format_image','formatting','image')
    add('format_caption','Figure 11. A/E distinguish newline token 50118 from standalone-space token 1437. B/D dashed curves exclude newline tokens/original windows containing them, respectively; unequal retained counts make these sensitivity checks. C averages repetition within outputs. F pairs the same final prompts; red marks anchor prompt 0.','caption')
    add('blank','Formatting: synthetic-only ends with 38.62% newline tokens, 111 newline-heavy outputs and 14 decoded whitespace-only outputs (ten consist solely of token 50118). Mixing has 0.31% newline tokens and no whitespace-only outputs, but 26 outputs contain at least 32 standalone-space tokens; these tokens account for 9.02% of its probe.','bullet')
    add('sensitivity','Unexpected rankings: removing newlines changes entropy from 4.431 to 6.132 for synthetic-only, versus 5.590 to 5.586 for mixing. Non-newline four-gram repetition is 35.99% versus 59.57%, on 9,286 versus 15,435 retained original windows. Raw rankings therefore partly reflect blank-line concentration; filtered rankings describe unequal survivors, not overall quality.','bullet')
    add('examples','6. Saved examples, conclusion and limitations','heading2',True)
    add('prompt_label','Fixed anchor: prompt 0, selected before this comparison; factual claims in generated text are unverified.','body')
    add('prompt',examples[0]['prompt'],'quote')
    add('examples_caption','Table 17. Complete 64-token continuations for that same prompt. Both final models loop; the example illustrates a counterexample to an aggregate ranking rather than typical quality. Original whitespace is retained in the raw samples.','caption')
    add('examples_table','examples','table')
    add('nonuniform','Repetition is not uniformly better: synthetic-only has greater within-output repetition on 145 prompts, mixing on 93, with 18 ties. Anchor prompt 0 is more repetitive under mixing. The aggregate improvements coexist with worse individual examples and lower pooled distinct-2/3.','bullet')
    add('conclusion','Conclusion: 50/50 mixing attenuates real-test deterioration and the blank-line failure in this continued-weight experiment. It does not preserve S0 performance, remove repetitive output or improve every diversity measure. This is evidence of a relative benefit under this protocol, not prevention of model collapse.','bullet')
    add('limits','Limits: one seed, ten cycles, OPT-125M, WikiText-2, deterministic five-beam decoding and a fixed total training budget. The probe covers the same first 256 consecutive prompts from six adjacent articles, not a random full-corpus sample. Resampling an existing finite real split is not fresh external data, and these results establish no universal safe synthetic fraction.','bullet')
    add('execution',f"Execution: mixed run {receipt['pipeline_hours']:.2f} hours; automatic GPU shutdown verified. Observed account debit ${receipt['cost']['observed_account_balance_delta_usd']:.2f}, including concurrent storage and billing timing, not an itemized compute invoice. Recovery volumes remain preserved.",'small')
    add('evidence','Evidence: all 1,563 small metadata files / 52,418,722 bytes were recovered and checksum-verified, plus ten new 256x64 probe arrays / 656,640 bytes. The shared S0 probe and existing comparator evidence were reused. Twenty-one unique probes, NLL/PPL arithmetic, all 100 epoch records, handoffs and mixture-selection hashes were checked; no weights/full training arrays were downloaded or models re-evaluated.','small')
    add('source','Per-cycle statistics, fixed-prompt outputs, source records, analysis scripts and provenance.','link')
    specs=[('Test corpus perplexity','test_perplexity','.2f',1),('Validation perplexity','validation_perplexity','.2f',1),
        ('Mean test NLL (nats)','test_nll','.3f',1),('P95 test-block NLL','test_block_nll_p95','.3f',1),
        ('Distinct-1 (%)','distinct_1','.2f',100),('Distinct-2 (%)','distinct_2','.2f',100),('Distinct-3 (%)','distinct_3','.2f',100),
        ('Pooled repeated 4-grams (%)','repeated_4gram_rate','.2f',100),('Mean within-output repeated 4-grams (%)','mean_within_repeated_4gram_rate','.2f',100),
        ('Duplicate outputs / 256','duplicate_outputs','.0f',1),('Pooled token entropy (nats)','token_entropy_nats','.3f',1),
        ('Mean within-output entropy','mean_within_entropy_nats','.3f',1),('Frozen-S0 mean NLL','frozen_s0_nll_mean','.3f',1),
        ('Frozen-S0 P95 NLL','frozen_s0_nll_p95','.3f',1),('Newline tokens (%)','newline_token_50118_share','.2f',100),
        ('Standalone-space tokens (%)','standalone_space_share','.2f',100),('Whitespace-only outputs / 256','whitespace_only_decoded_outputs','.0f',1),
        ('Space-heavy outputs (>=32/64)','standalone_space_heavy_outputs_ge32','.0f',1)]
    tables={'methods':[['Setting','100% synthetic','50/50 mixture'],
        ['Model / corpus','Pinned OPT-125M / WikiText-2 raw','Identical'],['Start / weights','Exact real S0 and D1; previous selected weights thereafter','Identical S0/D1; mixed S1 onward'],
        ['Blocks / tokens per dataset','37,946 synthetic / 2,428,544 tokens','18,973 real + 18,973 synthetic / same total tokens'],
        ['Training / selection','Five epochs per cycle; retain epoch 5','Identical; ten cycles per arm'],
        ['Generation','Same real 64-token prompts -> 64 new tokens; five beams, penalty 1.0, EOS suppressed','Identical full source pool, then sample half'],
        ['Optimization','Fresh AdamW/cosine per cycle; LR 2e-5; seed 42; batch 128; BF16','Identical'],
        ['Evaluation','3,922 real validation / 4,494 real test blocks; no test selection','Identical frozen arrays'],
        ['Environment','RTX PRO 4000 Blackwell; torch 2.8.0+cu128; Transformers 4.37.2; Datasets 3.4.1','Identical recorded driver/software']],
        'endpoints':[['Measure','S0','100% S10','50/50 S10']]+[[label,*[format(r[key]*scale,fmt) for r in [base,b,m]]] for label,key,fmt,scale in specs],
        'examples':[['Model','Complete saved continuation'],['S0',examples[0]['s0']],['100% synthetic S10',examples[0]['outputs']['synthetic_100']['text']],['50/50 S10',examples[0]['outputs']['mixed_50']['text']]]}
    images={'metrics':str((analysis/'figures/recursive_metrics.png').resolve()),'training':str((analysis/'figures/real_text_and_training.png').resolve()),'formatting':str((analysis/'figures/formatting_diagnostics.png').resolve())}
    return {'entries':entries,'tables':tables,'images':images,'source_url':f'https://github.com/BillLeoutsakosvl346/Model_Collapse_in_Synthetic_Data_Training/blob/main/reports/{RUN}/report.md','summary':{'blocks_mixed_lower':blocks['mixed_lower'],'ppl_reduction_percent':s['final_ppl_reduction_percent']}}


def markdown(d, directory):
    lines=[]
    def link(path):return Path(os.path.relpath(Path(path).resolve(),Path(directory).resolve())).as_posix()
    for e in d['entries']:
        kind,text=e['kind'],e['text']
        if kind=='table':
            rows=d['tables'][text]
            lines += ['| '+' | '.join(rows[0])+' |','| '+' | '.join(['---']*len(rows[0]))+' |']
            lines += ['| '+' | '.join(v.replace('\n','\\n').replace('|','\\|') for v in r)+' |' for r in rows[1:]]
        elif kind=='image':lines += [f'![Mixed-data comparison: {text}]({link(d["images"][text])})']
        elif kind=='link':lines += [f'[Per-cycle metrics]({link(ROOT/"analysis/metrics.csv")}) | [Epoch histories]({link(ROOT/"analysis/training_history.csv")}) | [Examples]({link(ROOT/"analysis/examples.json")}) | [Audit and provenance]({link(ROOT/"analysis/publication_reproduction.json")}) | [Analysis script]({link(PROJECT/"comparisons/continued_mixed50_vs_synthetic_final5/analyse.py")})']
        elif kind=='bullet':lines += ['- '+text]
        elif kind=='quote':lines += ['> '+text.replace('\n','\\n')]
        else:lines += [{'heading1':'# ','heading2':'## ','heading3':'### '}.get(kind,'')+text]
        lines += ['']
    return '\n'.join(lines)


def build(analysis,out):
    analysis=Path(analysis).resolve();out=new_destination(ROOT,out)
    d=draft(analysis)
    out.mkdir(parents=True,exist_ok=True)
    (out/'draft.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'report.md').write_text(markdown(d,out),encoding='utf-8')
    write_provenance(PROJECT,out,__file__,[analysis/'summary.json',analysis/'examples.json',analysis/'publication_reproduction.json',*map(Path,d['images'].values()),PROJECT/'docs/validation/2026-11-01_mixed-50-full-completion.json'],scope='Authored mixed-data comparison and native-Docs draft from verified analysis; no cloud writes or model computation.')
    print(json.dumps({'entries':len(d['entries']),'native_tables':len(d['tables']),'figure_families':len(d['images']),'prose_words':sum(len(e['text'].split()) for e in d['entries'] if e['kind'] not in ['table','image'])}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,default=ROOT/'analysis');p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();build(args.analysis,args.out)
