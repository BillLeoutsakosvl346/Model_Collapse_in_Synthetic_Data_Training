"""Author the matched 20/80 reset study from verified saved evidence."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
from model_collapse.common import PROJECT,read_json
sys.path.insert(0,str(PROJECT))
from scripts.analysis.publication import new_destination,write_provenance

RUN='20261101T030306Z-full-mixed-campaign-reset-mixed20'
ROOT=PROJECT/'reports'/RUN


def draft(analysis):
    s=read_json(analysis/'summary.json');assert s['passed'] and s['shared_s0_s1_d2_m2_verified']
    p=s['endpoint_metrics'];z,shared,c,a=[p[k] for k in ('s0','shared_s1','continued','reset')]
    ctx=read_json(analysis/'reset_fraction_context.json');ex=read_json(analysis/'examples.json')[0]
    e=[]
    def add(i,text,kind='body',page=False):e.append({'id':i,'text':text,'kind':kind,'page':page})
    add('title','Week 3, Part 5: Matched 20/80 reset and continued training','heading1',True)
    add('meta','OPT-125M / WikiText-2; seed 42. 20% synthetic / 80% real. Reset branch completed 1 November 2026; sealed results analysed 2 November.','small')
    add('methods','1. Objective and exact shared beginning','heading2')
    add('objective','We test whether resetting to the fixed real-trained S0 limits deterioration when synthetic text is only one fifth of the training mixture. The main comparison holds the entire first update fixed against the completed 20/80 continued arm.')
    add('prefix','Shared start: S0 generates historical D1; train S0 on M1 = mix(real, D1) for five epochs → shared S1; S1 generates D2. Both arms import exactly that S0/S1/D2 and the identical cycle-2 mixture M2, including token IDs, selected rows and shuffle. Shared points are reused observations, not independent repetitions.','bullet')
    add('arms','From cycle 2, continued trains its preceding epoch-5 checkpoint; reset trains fixed S0. The preceding model in each arm still generates that arm’s next synthetic pool. Resetting weights therefore removes accumulated parameter updates while retaining recursive data feedback.','bullet')
    add('sampling','Each mixture samples 30,357 real and 7,589 synthetic blocks without replacement within each source/cycle, then shuffles. Selection streams are matched across arms; later synthetic contents differ. Freeze each mixture for all five epochs. Real text is reused from the training split only; prompts and held-out real text never enter synthetic targets.','bullet')
    add('methods_cap','Table 24. Matched protocol. The deliberate difference is the initializer for cycles 2–10. Synthetic fraction is 7,589/37,946 = 19.99947% after block rounding.','caption')
    add('methods_table','methods','table')
    add('prediction','2. Real-text prediction and phrase diversity','heading2',True)
    add('prediction_image','prediction_and_phrases','image')
    add('prediction_cap','Figure 18a. Large square panels: fixed real-test/validation perplexity (A/B), pooled distinct-2/3 (C/D). Blue: 20/80 continued; purple: reset to S0; grey: exact shared S0/S1. One seeded trajectory per arm; no independent-run error bars.','caption')
    add('ppl',f"The common test-PPL jump is {z['test_perplexity']:.2f} → {shared['test_perplexity']:.2f}. Reset then stays approximately 35.9–36.0, ending at {a['test_perplexity']:.2f}; continued rises to {c['test_perplexity']:.2f}. Reset is {s['reset_final_ppl_reduction_percent']:.2f}% lower at cycle 10, but still {s['reset_ppl_above_s0_percent']:.2f}% above S0.",'bullet')
    add('ppl_cause','Interpretation: restarting from S0 prevents accumulated weight changes from carrying forward; the high real-data fraction also anchors each new update. These results are consistent with reduced cumulative drift, but do not identify its exact cause. Stable held-out PPL does not establish stable generated-text quality.','bullet')
    add('diversity','3. Output entropy and repetition','heading2',True)
    add('diversity_image','entropy_and_repetition','image')
    add('diversity_cap','Figure 18b. Pooled token entropy (E) and mean repeated four-gram rate inside individual outputs (F), measured on the same 256 fixed prompts and 64-token continuations. These describe emitted token frequencies/loops, not predictive uncertainty.','caption')
    add('tradeoff',f"At cycle 10, reset has lower distinct-2/3 ({100*a['distinct_2']:.2f}% / {100*a['distinct_3']:.2f}%) than continued ({100*c['distinct_2']:.2f}% / {100*c['distinct_3']:.2f}%). Reset entropy is {a['token_entropy_nats']:.3f} versus {c['token_entropy_nats']:.3f} nats, while within-output repetition is {100*a['mean_within_repeated_4gram_rate']:.2f}% versus {100*c['mean_within_repeated_4gram_rate']:.2f}%.",'bullet')
    add('separate','Why this can coexist with much lower PPL: real-test loss measures probabilities assigned to actual held-out text. Distinct-n and repetition count what deterministic beam decoding emits for a small, fixed prompt set. A model can predict real targets better yet choose narrower, more repetitive high-probability completions.','bullet')
    add('diversity_cause','A plausible mechanism is repeated selection of familiar templates under deterministic beams, carried forward through the synthetic pools even when weights reset. Formatting and token-frequency concentration also affect entropy. The data show these effects on the outputs; they do not prove a particular internal mechanism.','bullet')
    add('statistics','4. Endpoint statistics','heading2',True)
    add('stats_intro','Prediction and generated-text statistics are separate outcomes. Lower real-test NLL/PPL means less surprise at unseen real targets; a diversity ranking is not a general ranking of factuality or model quality.')
    add('stats_cap','Table 25. Common S0 and final checkpoints C10 (continued) / A10 (reset). Natural-log units. Percentages use identical token/window denominators; duplicate counts are copies beyond the first among 256 outputs. Shared S1 test PPL is 35.95.','caption')
    add('stats_table','endpoints','table')
    add('stats_note','Pooled repeated-4 = 1 − distinct-4, so those entries are complementary. Within-output repetition instead averages excess four-gram occurrences separately inside each continuation. Token entropy uses emitted-token frequencies, not the model’s softmax distribution.')
    add('training','5. Real-text coverage and five-epoch training','heading2',True)
    add('training_image','real_text_and_training','image')
    add('training_cap','Figure 19. A: cumulative real-test block NLL, with S0 dashed. B: continued minus reset loss on identical blocks; positive favours reset. C/D: cycle-10 validation/training losses over five epochs. Epoch 5 is retained regardless of validation.','caption')
    blocks=s['paired_test_blocks']
    add('coverage',f"Reset has lower final NLL on {blocks['reset_lower']:,}/4,494 blocks ({100*blocks['reset_lower']/4494:.2f}%), continued on {blocks['continued_lower']}, with no ties. Mean continued-minus-reset gap is {blocks['mean_continued_minus_reset_nll']:.3f} nats/target. Reset still exceeds S0 loss on {s['final_vs_s0_blocks']['reset']['worse']:,} blocks, versus {s['final_vs_s0_blocks']['continued']['worse']:,} for continued.",'bullet')
    add('epochs','Epoch 1 gives the lowest fixed real-validation loss in all 19 unique stages, but all execute five epochs and retain the fifth. In cycle 10, validation NLL rises 3.547 → 3.587 for reset and 4.158 → 4.265 for continued; training NLL falls 2.841 → 2.560 and 1.753 → 1.563.','bullet')
    add('fit','These histories show that fitting the current mixture more strongly does not improve held-out generalization. Training losses are online averages over different arm-specific targets, not final-checkpoint real-training loss, so they do not independently prove real-data memorization.','bullet')
    add('formatting','6. Repetition and formatting diagnostics','heading2',True)
    add('formatting_image','repetition_and_formatting','image')
    add('formatting_cap','Figure 20a. A/B: pooled four-gram repetition and within-output entropy. C: endpoint newline/space token shares. D: final within-output repetition distributions. Pooled statistics and within-output loops are kept separate.','caption')
    rep=s['paired_within_repetition']
    add('repetition',f"Reset is more repetitive on {rep['reset_more']} paired prompts, continued on {rep['continued_more']}, with {rep['ties']} ties. Both end with nine duplicate outputs. The reset repetition curve generally worsens even while its real-test PPL remains nearly flat.",'bullet')
    add('format',f"Newlines are only {100*a['newline_token_50118_share']:.2f}% (reset) and {100*c['newline_token_50118_share']:.2f}% (continued), with no newline-heavy or whitespace-only outputs. Standalone-space tokens are {100*a['standalone_space_share']:.2f}% versus {100*c['standalone_space_share']:.2f}%, with {a['standalone_space_heavy_outputs_ge32']} versus {c['standalone_space_heavy_outputs_ge32']} space-heavy outputs. Newline exclusion leaves the entropy/repetition ranking unchanged; blank-line collapse is not the explanation here.",'bullet')
    add('reference','7. Frozen-S0 continuation scores','heading2',True)
    add('reference_image','frozen_s0_scores','image')
    add('reference_cap','Figure 20b. Frozen-S0 NLL mean (E) and P95 (F). Feed each generated continuation alone to S0 and score its 63 shifted next-token targets; omit the original prompt. Lower means more predictable to S0, not necessarily better text.','caption')
    add('score',f"Reset has lower mean frozen-S0 NLL ({a['frozen_s0_nll_mean']:.3f} versus {c['frozen_s0_nll_mean']:.3f}) but higher P95 ({a['frozen_s0_nll_p95']:.3f} versus {c['frozen_s0_nll_p95']:.3f}). A lower average can coexist with a more surprising upper tail. These are scores of different generated strings, not a comparison on one fixed real target set.",'bullet')
    add('score_limit','Repeated familiar phrases may be easy for S0 to predict. That can keep the mean score low despite visibly poor completions and lower diversity. Frozen-S0 scoring therefore cannot be interpreted as model distance, factuality or an independent accuracy score.','bullet')
    add('context','8. Context: all three fixed-S0 reset fractions','heading2',True)
    add('context_image','reset_fraction_context','image')
    add('context_cap','Figure 21. Reset conditions at 100%, 50% and 20% synthetic: real-test PPL, pooled distinct-3, token entropy and within-output repetition. Only S0 is shared across ratios; S1 and later synthetic pools differ. Settings/evaluation and recorded environments match.','caption')
    ends=ctx['endpoints'];r100,r50,r20=[ends[k] for k in ('synthetic_100','mixed_50','mixed_20')]
    add('context_ppl',f"Reset endpoint PPL decreases as real fraction increases: {r100['test_perplexity']:.2f} → {r50['test_perplexity']:.2f} → {r20['test_perplexity']:.2f}. This differs from continued training, where 20% synthetic ends worse than 50%. The initializer/cumulative-training condition matters; no universal optimum follows from these single-seed trajectories.",'bullet')
    add('context_cause','The continued-only reversal is consistent with accumulated parameter changes contributing to the outcome. It does not establish whether the cause is real-data overfitting, generated-data feedback or another optimization effect. Reset also changes the generators and later datasets, so it is not a perfectly isolated test of one mechanism.','bullet')
    add('context_diversity','Diversity is not monotonically better with more real data: 20% reset has lower entropy and slightly higher within-output repetition than 50% reset, despite slightly higher distinct-3. The synthetic-only reset has higher phrase diversity but worse real-test PPL.','bullet')
    add('examples','9. Same-prompt output examples','heading2',True)
    add('prompt_label','Fixed historical anchor, prompt 0. This example is retained across comparisons rather than selected for typicality. Generated factual claims are unverified.')
    add('prompt',ex['prompt'],'quote')
    add('examples_cap',f"Table 26. Complete saved 64-token continuations. Here within-output repeated-4 is {100*ex['outputs']['continued']['within_repeated4']:.2f}% continued versus {100*ex['outputs']['reset']['within_repeated4']:.2f}% reset. Both loop; this example illustrates, but does not establish, the aggregate result.",'caption')
    add('examples_table','examples','table')
    add('example_note','A second saved comparison, prompt 116, also loops on repeated ordnance/ammunition phrases: within-output repeated-4 is 80.33% continued and 86.89% reset. It was selected as the first prompt nearest the median paired repetition gap; all 256 raw outputs remain available.')
    add('conclusion','10. Findings, accounting and limitations','heading2',True)
    add('finding','Main finding: at 20/80, fixed-S0 reset largely stabilizes real-text likelihood after the common first update, whereas continued training deteriorates. Yet reset loses phrase diversity and becomes more repetitive on the fixed probe. Low and stable PPL is therefore insufficient evidence that collapse has been prevented.','bullet')
    add('account','Accounting: this branch imports S0/S1/D2, then adds nine stages / forty-five epochs for cycles 2–10. Together with its continued arm there are 19 unique stages / 95 epochs; the 100 saved epoch entries include the shared first stage twice. A10 inherits its own five-epoch update from S0, while C10 carries all fifty; data history is recursive in both.','bullet')
    add('limits','Limits: one seed, OPT-125M, WikiText-2, ten cycles and deterministic five-beam decoding. The probe covers the first 256 consecutive training prompts from six adjacent articles, not a random corpus sample. Corpus-block comparisons are descriptive, not independent training replicates. Real text is resampled from a finite existing split, not newly collected.','bullet')
    add('evidence','Evidence: recovered and checksum-verified 1,421 metadata files / 53,710,269 bytes plus nine new probe arrays / 590,976 bytes. The main pair audit checks 22 metric rows, twenty unique probes, 100 saved epoch entries (95 unique), exact S0/S1/D2/M2, initializer/generator/data handoffs and terminal checkpoint selection. Saved probe metrics and real-test NLL/PPL were recalculated; source/shared observations are reused. Exact file hashes and the separate reset-fraction context are preserved alongside the report.','small')
    add('operations','The campaign’s recorded completion is 1 November 23:20:44 UTC/London, with provider-confirmed GPU shutdown at 23:21:09 and watchdog reason complete. This analysis made no Runpod calls, SSH connections or resource changes. Bulk weights/full training arrays were not restored or models re-evaluated; recovery storage remains preserved.','small')
    add('source','Per-cycle statistics, epoch histories, mixture identities, output examples, context comparisons and reproducible square-panel figures.','link')
    specs=[('Real-test PPL','test_perplexity','.2f',1),('Validation PPL','validation_perplexity','.2f',1),('Test NLL (nats)','test_nll','.3f',1),('P95 block NLL','test_block_nll_p95','.3f',1),
        ('Distinct-1 (%)','distinct_1','.2f',100),('Distinct-2 (%)','distinct_2','.2f',100),('Distinct-3 (%)','distinct_3','.2f',100),('Distinct-4 (%)','distinct_4','.2f',100),
        ('Pooled repeated-4 (%)','repeated_4gram_rate','.2f',100),('Mean within repeated-4 (%)','mean_within_repeated_4gram_rate','.2f',100),('Duplicate outputs / 256','duplicate_outputs','.0f',1),
        ('Pooled entropy (nats)','token_entropy_nats','.3f',1),('Within entropy (nats)','mean_within_entropy_nats','.3f',1),('Frozen-S0 mean NLL','frozen_s0_nll_mean','.3f',1),('Frozen-S0 P95 NLL','frozen_s0_nll_p95','.3f',1),
        ('Newline tokens (%)','newline_token_50118_share','.2f',100),('Space tokens (%)','standalone_space_share','.2f',100),('Space-heavy outputs / 256','standalone_space_heavy_outputs_ge32','.0f',1)]
    tables={'methods':[['Setting','Matched 20/80 arms'],['Model / corpus','Pinned OPT-125M / WikiText-2 raw; identical real row/split packing'],['Common prefix','Exact real-trained S0, new 20/80 epoch-5 S1, D2 and M2; imported, not retrained'],['Dataset per cycle','30,357 real + 7,589 synthetic blocks × 64 tokens; 2,428,544 total tokens'],['Source tokens','1,942,848 real + 485,696 synthetic. Full synthetic source pool: 37,946 blocks before sampling'],['Training / decoding','Five epochs, always retain fifth; fresh AdamW/cosine; LR 2e-5; batch 128; BF16; seed 42. Fixed 64-token prompts → 64 new tokens, five beams, penalty 1.0'],['Evaluation','3,922 real validation / 4,494 real test blocks; 63 scored targets/block; fixed 256-output probe'],['Environment','Same RTX PRO 4000 Blackwell; PyTorch 2.8.0+cu128, Transformers 4.37.2, Datasets 3.4.1']],
        'endpoints':[['Measure','S0','Continued C10','Reset A10']]+[[name,*[format(r[key]*scale,fmt) for r in (z,c,a)]] for name,key,fmt,scale in specs],
        'examples':[['Checkpoint','Complete saved continuation'],['Shared S1',ex['shared_s1']],['Continued C10',ex['outputs']['continued']['text']],['Reset A10',ex['outputs']['reset']['text']]]}
    return {'entries':e,'tables':tables,'images':{p.stem:str(p.resolve()) for p in (analysis/'figures').glob('*.png')},
        'source_url':f'https://github.com/BillLeoutsakosvl346/Model_Collapse_in_Synthetic_Data_Training/blob/main/reports/{RUN}/report.md'}


def markdown(d,directory):
    lines=[]
    def link(path):return Path(os.path.relpath(path,directory)).as_posix()
    for e in d['entries']:
        k,t=e['kind'],e['text']
        if k=='table':
            rs=d['tables'][t];lines+=['| '+' | '.join(rs[0])+' |','| '+' | '.join(['---']*len(rs[0]))+' |']
            lines+=['| '+' | '.join(x.replace('\n','\\n').replace('|','\\|') for x in row)+' |' for row in rs[1:]]
        elif k=='image':lines+=[f'![20/80 reset study: {t}]({link(d["images"][t])})']
        elif k=='bullet':lines+=['- '+t]
        elif k=='quote':lines+=['> '+t.replace('\n','\\n')]
        elif k=='link':lines+=[f'[Metrics]({link(ROOT/"analysis/metrics.csv")}) | [Epoch histories]({link(ROOT/"analysis/training_history.csv")}) | [Examples]({link(ROOT/"analysis/examples.json")}) | [Context]({link(ROOT/"analysis/reset_fraction_context.json")}) | [Audit/provenance]({link(ROOT/"analysis/publication_reproduction.json")})']
        else:lines+=[{'heading1':'# ','heading2':'## ','heading3':'### '}.get(k,'')+t]
        lines+=['']
    return '\n'.join(lines)


def build(analysis,out):
    analysis=Path(analysis).resolve();out=new_destination(ROOT,out);d=draft(analysis);out.mkdir(parents=True)
    (out/'draft.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'report.md').write_text(markdown(d,out),encoding='utf-8')
    write_provenance(PROJECT,out,__file__,[analysis/'summary.json',analysis/'examples.json',analysis/'reset_fraction_context.json',analysis/'publication_reproduction.json',*map(Path,d['images'].values())],scope='Authored matched reset20/continued20 study and descriptive cross-fraction reset context; no cloud or model operations.')
    print(json.dumps({'entries':len(d['entries']),'tables':len(d['tables']),'figures':len(d['images'])}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,default=ROOT/'analysis');p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.analysis,a.out)
