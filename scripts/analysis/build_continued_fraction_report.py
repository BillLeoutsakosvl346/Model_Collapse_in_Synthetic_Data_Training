"""Author the three-fraction continued-weight study; no external operations."""
import argparse,json,os,sys
from pathlib import Path
from model_collapse.common import PROJECT,read_json
sys.path.insert(0,str(PROJECT))
from scripts.analysis.publication import new_destination,write_provenance

RUN='20261101T030306Z-full-mixed-campaign-continued-mixed20'
ROOT=PROJECT/'reports'/RUN


def draft(analysis):
    s=read_json(analysis/'summary.json');assert s['passed']
    p=s['endpoint_metrics'];z,b,f,m=[p[k] for k in ('s0','synthetic_100','mixed_50','mixed_20')]
    ex=read_json(analysis/'examples.json')[0];entries=[]
    def add(i,text,kind='body',page=False):entries.append({'id':i,'text':text,'kind':kind,'page':page})
    add('title','Week 3, Part 4: Continued training across synthetic fractions','heading1',True)
    add('meta','OPT-125M / WikiText-2; seed 42. Completed 20% synthetic / 80% real continued run, compared with the preserved 50/50 and synthetic-only continued runs.','small')
    add('methods','1. Objective and controlled methodology','heading2')
    add('objective','We tested whether replacing more synthetic targets with real training text consistently limits recursive deterioration. All three arms carry their preceding weights, execute five epochs per cycle and retain epoch 5.')
    add('shared','Shared start: exact pinned real-trained S0 and its historical D1 synthetic pool. Model/data revisions, real split packing, fixed prompts/evaluation, optimizer/scheduler restarts, seed, batches, precision, decoding and recorded GPU/software match. S1 differs across ratios because their training targets already differ.','bullet')
    add('loop','Loop: form M1 = mix(real, D1), train S0 → S1; S1 generates D2, form M2 = mix(real, D2), train S1 → S2; continue through S10. No resetting to S0 in any arm. A full synthetic pool is generated before sampling the required fraction; prompts are excluded from targets.','bullet')
    add('sampling','Sampling: independent deterministic cycle-specific source selections and shuffle; no replacement within a source/cycle, real blocks may recur across cycles. Each mixture is frozen for its five epochs. Only the original real training split is eligible, never validation/test.','bullet')
    add('methods_cap','Table 21. Equal total blocks, training exposures and evaluation context. “20%” rounds to 7,589/37,946 = 19.99947% synthetic blocks; S1 onward is newly trained for this ratio.','caption')
    add('methods_table','methods','table')
    add('prediction','2. Real-text prediction and phrase diversity','heading2',True)
    add('prediction_image','prediction_and_phrases','image')
    add('prediction_cap','Figure 15a. Larger square panels: fixed real-test/validation PPL (A/B) and pooled distinct-2/3 (C/D). Orange: 100% synthetic; teal: 50%; blue: 20%; grey marker: shared S0. Lines join one-seed observations, with no independent-run uncertainty bands.','caption')
    add('ppl','Observed PPL: 20% starts closest to S0 after one update (35.95 versus 37.27 at 50%, 54.46 at 100%). It then rises every cycle, crosses 50% between cycles 3 and 4, and ends at 72.06 versus 62.71 and 107.37. Thus the endpoint ranking is not monotonic in synthetic fraction.','bullet')
    add('contrast','At cycle 10, 20% PPL is 32.88% lower than synthetic-only, 14.92% higher than 50/50 and 112.67% above S0 (33.89). The validation curve agrees with the test comparison. The three trajectories support relative benefits of mixing, not the claim that reducing synthetic fraction always improves prediction.','bullet')
    add('entropy','3. Generated-text diversity and repetition','heading2',True)
    add('entropy_image','entropy_and_repetition','image')
    add('entropy_cap','Figure 15b. Pooled token entropy (E) and mean within-output repeated four-grams (F). All fractions use the same 256×64 output probe and token/window definitions. Output-frequency entropy is not predictive entropy or accuracy.','caption')
    add('diversity','The 20% endpoint has the largest pooled distinct-2/3 of the three final models (30.03% / 40.25%) and less mean within-output repetition (46.73%) than both comparators. Its pooled entropy is 5.706 nats, close to S0’s 5.736; that does not mean its real-text prediction is preserved.','bullet')
    add('distinction','Why this can disagree with PPL: held-out loss checks probabilities assigned to fixed real targets; these diversity statistics count patterns emitted for a small set of prompts under beam decoding. Broader phrase variety or fewer loops can coexist with worse probability calibration on actual held-out text.','bullet')
    add('denominators','Distinct-n counts unique token phrases divided by all windows, never spanning outputs. All arms use 16,128 bigram, 15,872 trigram and 15,616 four-gram positions. Pooled repeated-4 is 1 − distinct-4, so these two are complementary, not independent findings.','bullet')
    add('statistics','4. Endpoint statistics','heading2',True)
    add('stats_intro','The table separates real-text prediction, pooled diversity, within-output repetition and formatting. Lower NLL/PPL is better on held-out real text; the remaining columns describe generated output rather than a single overall quality score.')
    add('stats_cap','Table 22. Shared S0 and final S10 for all three continued-weight conditions. Natural-log units; percentages are fractions of tokens/windows. Duplicate counts are copies beyond the first among 256 outputs.','caption')
    add('stats_table','endpoints','table')
    add('stats_note','The 20% model has nine duplicate outputs, versus ten for 50/50 and the larger synthetic-only count shown above. Neither mixed condition reproduces the major synthetic-only newline concentration, but both retain space-heavy outputs. All per-cycle measurements are saved alongside this report.')
    add('training','5. Real-text coverage and training dynamics','heading2',True)
    add('training_image','real_text_and_training','image')
    add('training_cap','Figure 16. A: cumulative final test-block NLL, S0 dashed. B: other arm minus 20% loss on identical blocks; positive favours 20%. C/D: cycle-10 validation/training loss over five epochs. D compares different arm-specific target datasets.','caption')
    add('coverage','Paired blocks: 20% has lower final NLL than synthetic-only on 3,750/4,494 blocks (83.44%), but lower NLL than 50/50 on only 737 (16.40%). It remains worse than S0 on 4,410 blocks. These are descriptive corpus-block counts, not independent model replicates.','bullet')
    add('epochs','Five-epoch effect: epoch 1 minimizes real validation loss in all thirty unique stages, but epoch 5 is intentionally retained. For 20% cycle 10, training NLL falls 1.753 → 1.563 while validation NLL rises 4.158 → 4.265. Better fitting of the current mixture does not improve real-text generalization here.','bullet')
    add('cause','Possible cause of the 20%/50% reversal: stronger repeated fitting of the same finite real corpus, combined with different self-generated trajectories. Each real block is selected on average about eight times across the 20% mixtures, versus five at 50%, before five epoch passes. This is a plausible overfitting explanation, not a demonstrated mechanism; no real-only or multi-seed control isolates it.','bullet')
    add('formatting','6. Repetition and formatting diagnostics','heading2',True)
    add('formatting_image','repetition_and_formatting','image')
    add('formatting_cap','Figure 17a. A/B: pooled phrase repetition and within-output entropy. C: endpoint newline/space token shares. D: cumulative within-output four-gram repetition; curves farther left have less repetition. Original unfiltered output statistics are retained.','caption')
    add('repetition','On paired prompts, synthetic-only is more repetitive than 20% on 165 outputs, 20% on 78, with 13 ties. Against 50/50, 50/50 is more repetitive on 130, 20% on 90, with 36 ties. Aggregate improvements are not universal prompt-level improvements.','bullet')
    add('format','The 20% probe has 0.31% newline tokens, no newline-heavy or whitespace-only decoded outputs, but 8.97% standalone-space tokens and 26 space-heavy outputs. Synthetic-only has 38.62% newlines and 111 newline-heavy outputs. Entropy and pooled phrase rankings can therefore reflect formatting as well as substantive vocabulary.','bullet')
    add('reference','7. Frozen-S0 continuation scores','heading2',True)
    add('reference_image','frozen_s0_scores','image')
    add('reference_cap','Figure 17b. E/F: mean and 95th-percentile continuation NLL under the same frozen real-trained S0. Each 64-token output is teacher-forced on its own, scoring 63 targets; the original prompt is omitted. Lower means easier for S0 to predict.','caption')
    add('score','The 20% mean score is 1.653 nats/target versus 1.562 at 50% and 2.238 for synthetic-only; its P95 is 2.845 versus 3.139 and 3.834. Mean and tail can rank differently. This metric evaluates different generated strings, not prediction of one fixed real dataset.','bullet')
    add('interpret','Predictable repetition can receive a low S0 loss. Conversely, more varied output may be less predictable to S0 without having worse factuality or real-test behaviour. These scores cannot be read as model distance or a second accuracy measure; compare them with the held-out losses and formatting diagnostics.','bullet')
    add('examples','8. Same-prompt examples','heading2',True)
    add('prompt_label','Prompt 0 is the fixed historical anchor, not chosen to represent typical quality. Generated factual claims are unverified. The full 64-token outputs are reproduced below.')
    add('prompt',ex['prompt'],'quote')
    add('examples_cap','Table 23. Final checkpoints on the exact same 64-token prompt. Within-output repeated-4 is 73.77% (100%), 86.89% (50%) and 37.70% (20%) here. All still repeat; this example does not establish factual correctness or aggregate typicality.','caption')
    add('examples_table','examples','table')
    add('example_note','Additional formatting example: prompt 52 gives an all-newline synthetic-only output, while both mixed arms emit “References :” followed by spaces. Those mixed outputs contain alphabetic characters and are not counted as whitespace-only; they remain degenerate completions. Exact token arrays and original whitespace are preserved in the saved examples.')
    add('conclusion','9. Findings, limits and evidence','heading2',True)
    add('finding','Main finding: both real-data mixtures reduce final real-test deterioration relative to synthetic-only continued training. Among these seeded trajectories, 50/50 has lower final PPL than 20/80, while 20/80 has stronger endpoint phrase-diversity and repetition statistics. There is no single ranking across all metrics.','bullet')
    add('limits','Limits: one seed, OPT-125M, WikiText-2, ten updates and deterministic five-beam decoding. The fixed probe is the first 256 consecutive training prompts from six adjacent articles, not a random full-corpus sample. More real means repeatedly sampling the same finite split, not acquiring new external data. No optimum or universal safe synthetic percentage is established.','bullet')
    add('account','Accounting: all arms train 37,946×64 input tokens per cycle, fifty epochs total: 121,427,200 input-token exposures per arm. At 20%, 97,142,400 are real and 24,284,800 synthetic. S0/D1 are imported; each continued arm adds ten stages, nine full next pools and a final probe.','bullet')
    add('audit','Evidence: recovered and checksum-verified 1,560 metadata files / 52,452,006 bytes, plus ten new probes / 656,640 bytes. Shared S0 and preserved comparator evidence are reused. The audit checks 33 metric rows, 31 unique probes, 150 epoch entries, all thirty model/data handoffs and exact deterministic source-selection hashes. The selected-record publication lists 117 scientific files / 5,291,854 bytes, including its probes.','small')
    add('scope','This analysis used pinned Hugging Face backups and offline records only: no Runpod API calls, SSH sessions, worker/resource changes, model forward passes or full checkpoint/training-array downloads. The completed 20/80 continued child is one job in a larger campaign; this report does not claim that the final reset job or GPU shutdown has completed.','small')
    add('source','Per-cycle metrics, epoch histories, mixture provenance, fixed-prompt examples and reproducible square-panel figures.','link')
    specs=[('Real-test PPL','test_perplexity','.2f',1),('Validation PPL','validation_perplexity','.2f',1),('Test NLL (nats)','test_nll','.3f',1),('P95 block NLL','test_block_nll_p95','.3f',1),
        ('Distinct-1 (%)','distinct_1','.2f',100),('Distinct-2 (%)','distinct_2','.2f',100),('Distinct-3 (%)','distinct_3','.2f',100),('Distinct-4 (%)','distinct_4','.2f',100),
        ('Pooled repeated-4 (%)','repeated_4gram_rate','.2f',100),('Mean within repeated-4 (%)','mean_within_repeated_4gram_rate','.2f',100),('Duplicate outputs / 256','duplicate_outputs','.0f',1),
        ('Pooled entropy (nats)','token_entropy_nats','.3f',1),('Within entropy (nats)','mean_within_entropy_nats','.3f',1),('Frozen-S0 mean NLL','frozen_s0_nll_mean','.3f',1),('Frozen-S0 P95 NLL','frozen_s0_nll_p95','.3f',1),
        ('Newline tokens (%)','newline_token_50118_share','.2f',100),('Space tokens (%)','standalone_space_share','.2f',100),('Whitespace-only / 256','whitespace_only_decoded_outputs','.0f',1)]
    tables={'methods':[['Setting','Common protocol / change'],['Model / source','Pinned OPT-125M / WikiText-2 raw; exact shared real-trained S0 and D1'],['Weights / selection','Continue previous epoch-5 weights; fresh AdamW/cosine each cycle; five epochs always, no early stopping'],['Training size','37,946 blocks × 64 tokens = 2,428,544 tokens in every arm'],['Source counts: real / synthetic','100%: 0 / 37,946; 50%: 18,973 / 18,973; 20%: 30,357 / 7,589'],['20% source tokens','1,942,848 real + 485,696 synthetic per cycle; source pool is still 37,946 synthetic blocks'],['Decoding / optimization','Fixed real 64-token prompts → 64 new tokens; five beams, penalty 1.0. LR 2e-5; seed 42; batch 128; BF16'],['Evaluation / environment','3,922 validation / 4,494 test real blocks; fixed 256-output probe. Same RTX PRO 4000 Blackwell and recorded software']],
        'endpoints':[['Measure','S0','100%','50%','20%']]+[[name,*[format(r[key]*scale,fmt) for r in (z,b,f,m)]] for name,key,fmt,scale in specs],
        'examples':[['Final model','Complete saved continuation']]+[[label,ex['outputs'][arm]['text']] for arm,label in [('synthetic_100','100% synthetic'),('mixed_50','50% synthetic'),('mixed_20','20% synthetic')]]}
    images={p.stem:str(p.resolve()) for p in (analysis/'figures').glob('*.png')}
    return {'entries':entries,'tables':tables,'images':images,'source_url':f'https://github.com/BillLeoutsakosvl346/Model_Collapse_in_Synthetic_Data_Training/blob/main/reports/{RUN}/report.md'}


def markdown(d,directory):
    lines=[]
    def link(path):return Path(os.path.relpath(path,directory)).as_posix()
    for e in d['entries']:
        k,t=e['kind'],e['text']
        if k=='table':
            rs=d['tables'][t];lines+=['| '+' | '.join(rs[0])+' |','| '+' | '.join(['---']*len(rs[0]))+' |']
            lines+=['| '+' | '.join(x.replace('\n','\\n').replace('|','\\|') for x in row)+' |' for row in rs[1:]]
        elif k=='image':lines+=[f'![Continued fractions: {t}]({link(d["images"][t])})']
        elif k=='bullet':lines+=['- '+t]
        elif k=='quote':lines+=['> '+t.replace('\n','\\n')]
        elif k=='link':lines+=[f'[Metrics]({link(ROOT/"analysis/metrics.csv")}) | [Epoch histories]({link(ROOT/"analysis/training_history.csv")}) | [Examples]({link(ROOT/"analysis/examples.json")}) | [Audit/provenance]({link(ROOT/"analysis/publication_reproduction.json")})']
        else:lines+=[{'heading1':'# ','heading2':'## ','heading3':'### '}.get(k,'')+t]
        lines+=['']
    return '\n'.join(lines)


def build(analysis,out):
    analysis=Path(analysis).resolve();out=new_destination(ROOT,out);out.mkdir(parents=True);d=draft(analysis)
    (out/'draft.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'report.md').write_text(markdown(d,out),encoding='utf-8')
    write_provenance(PROJECT,out,__file__,[analysis/'summary.json',analysis/'examples.json',analysis/'publication_reproduction.json',*map(Path,d['images'].values())],scope='Authored continued-fraction comparison from verified saved metrics, without cloud operations.')
    print(json.dumps({'entries':len(d['entries']),'tables':len(d['tables']),'figures':len(d['images'])}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,default=ROOT/'analysis');p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.analysis,a.out)
