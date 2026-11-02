"""Author the four-condition real-only comparison; no external operations."""
import argparse, json, os, sys
from pathlib import Path
from model_collapse.common import PROJECT, read_json
sys.path.insert(0,str(PROJECT))
from scripts.analysis.publication import new_destination, write_provenance

RUN='20261102T154833Z-full-continued'
ROOT=PROJECT/'reports'/RUN

def draft(analysis):
    analysis=Path(analysis);s=read_json(analysis/'summary.json');assert s['passed']
    p=s['endpoint_metrics'];z,b,f,m,r=[p[k] for k in ['s0','synthetic_100','mixed_50','mixed_20','real_0']]
    entries=[]
    def add(i,text,kind='body',page=False):entries.append({'id':i,'text':text,'kind':kind,'page':page})
    add('title','Week 3, Part 6: Real-only continued-training control','heading1',True)
    add('meta','OPT-125M / WikiText-2; seed 42. Ten continued updates, five epochs each, retaining epoch 5. Comparison with the completed 100%, 50% and 20% synthetic continued runs.','small')
    add('methods','1. Question and controlled methodology','heading2')
    add('objective','Can repeated training on the same finite real corpus worsen held-out prediction without any synthetic training targets? This control tests a possible contributor to the earlier 20%/50% reversal.')
    add('start','Shared start and training: use the exact pinned real-trained S0, then carry the preceding selected weights through ten cycles. Restart AdamW/cosine each cycle; train five epochs and always retain epoch 5. Model/data revisions, seed 42, learning rate 2e-5, batch 128, microbatch 32 and BF16 match the earlier arms.','bullet')
    add('selection','Real-only selection: use all 37,946 blocks from the real training split each cycle, with a new deterministic selection order/shuffle. Sampling the entire pool changes order, not contents. The dataset remains frozen across its five epochs. No validation or test blocks are eligible.','bullet')
    add('measurement','Fixed measurements: 3,922 real validation and 4,494 real test blocks; corpus NLL/PPL scores 63 next-token targets per block. Generated-text metrics use the same 256 fixed training prompts and 64-token continuations, five deterministic beams and repetition penalty 1.0. No prompt tokens enter synthetic targets.','bullet')
    add('hardware','Execution difference: the real-only run uses RTX PRO 4500 Blackwell Server Edition rather than RTX PRO 4000. Recorded software/driver settings match; archived training, data, metric and checkpoint-selection code agree. Only measurement probes are generated at 0% synthetic, since full synthetic pools would supply no training targets.','bullet')
    add('composition_cap','Table 27. Training composition per cycle. Every arm uses 37,946×64 = 2,428,544 input tokens, repeated for fifty epochs. Mean real passes are additional average exposures per original real block after S0, not fresh independently collected data.','caption')
    add('composition_table','composition','table')
    add('split_note','The frozen-array audit found zero exact 64-token overlaps between the training split and either held-out split. This does not establish absence of shorter shared passages or pretrained-model contamination.','small')
    add('prediction','2. Held-out prediction and phrase diversity','heading2',True)
    add('prediction_image','prediction_and_phrases','image')
    add('prediction_cap','Figure 22a. Fixed real-test/validation PPL (A/B) and pooled distinct-2/3 (C/D). Orange: 100% synthetic; teal: 50%; blue: 20%; purple: real-only. Grey marks the exact common S0. Lines join one-seed measurements.','caption')
    add('ppl','The real-only test curve rises every cycle: 33.89 at S0, 35.26 after one update, 49.99 at cycle 5 and 84.56 at cycle 10. Validation PPL also rises, ending at 83.23. Repeated continued training can therefore deteriorate on real held-out text even without synthetic targets.','bullet')
    add('ranking','Final test PPL ranks 50% synthetic (62.71), 20% (72.06), real-only (84.56), then synthetic-only (107.37). Real-only is 34.85% worse than 50/50 and 17.35% worse than 20/80, but 21.24% lower than synthetic-only. None preserves S0 prediction.','bullet')
    add('diversity','3. Output diversity is a separate outcome','heading2',True)
    add('diversity_image','entropy_and_repetition','image')
    add('diversity_cap','Figure 22b. Pooled token-frequency entropy (E) and mean repeated four-gram rate within individual outputs (F), using equal 256×64-token probes. Higher frequency entropy is not lower held-out error or higher factual accuracy.','caption')
    add('diverse','Real-only ends with pooled entropy 6.082 nats and distinct-2 of 45.14%, both higher than S0 (5.736; 28.81%) and all three final comparators. Mean within-output repetition falls to 17.28%, versus S0 40.45%, 20/80 46.73%, 50/50 50.24% and synthetic-only 57.03%.','bullet')
    add('meaning','Meaning: the real-only outputs use a wider collection of token patterns and fewer local loops on this probe. A model can do that while assigning worse probabilities to unseen real text. Real-test PPL and generated-text diversity measure different behaviours; their rankings need not agree.','bullet')
    add('collapse','Interpretation: the real-only control shows predictive deterioration with expanding probe diversity, whereas synthetic-only also shows narrowing entropy, loops and formatting failures. PPL increase alone is insufficient to attribute every deterioration curve to recursive synthetic-data collapse.','bullet')
    add('stats','4. Endpoint statistics','heading2',True)
    add('stats_intro','S0 is a single common starting observation. The four S10 columns are separately trained endpoints; percentages describe identical output-token/window denominators. Distinct-n and repetition describe output patterns, not a universal quality score.')
    add('stats_cap','Table 28. S0 and cycle-10 endpoints. 100/50/20/0 denote the percentage of synthetic training blocks; 0 is entirely real. NLL/entropy use natural-log units. Duplicate counts are copies beyond the first among 256 outputs.','caption')
    add('stats_table','endpoints','table')
    add('stats_note','Pooled repeated-4 is exactly 1−distinct-4. Within-output repetition counts repeated windows separately inside each continuation; it must not be interpreted as pooled vocabulary entropy. All original outputs, including formatting tokens, remain in these statistics.','small')
    add('training','5. Generalization and fitting dynamics','heading2',True)
    add('training_image','real_text_and_training','image')
    add('training_cap','Figure 23. A: final real-test block-loss distributions, with S0 dashed. B: each other arm minus real-only loss on identical blocks; positive favours real-only. C/D: cycle-10 validation/training losses across five epochs. D compares different target datasets across arms.','caption')
    add('coverage','The 50/50 model has lower final NLL than real-only on 4,257/4,494 test blocks (94.73%); 20/80 does so on 3,941 (87.69%). Real-only outperforms synthetic-only on 2,935 blocks (65.31%), while remaining worse than S0 on 4,436 (98.71%). These are descriptive corpus-block counts, not independent model replications.','bullet')
    add('selection_loss','In all ten real-only cycles, epoch 1 has the lowest validation loss, but epoch 5 is deliberately retained. At cycle 10, continued fitting lowers training loss while validation loss rises. This is the intended final-epoch condition, not an accidental early-stop or checkpoint-selection error.','bullet')
    add('exposure','6. What the real-only control adds','heading2',True)
    add('exposure_image','exposure_and_overfitting','image')
    add('exposure_cap','Figure 24. A: final PPL versus mean additional real-block exposure; fraction changes together with exposure, so this is not an independently identified dose-response curve. B: real-only epoch-5 training and fixed validation NLL across cycles, with S0 validation dashed.','caption')
    add('overfit','Observed overfitting pattern: real-only training loss falls from 3.239 at the first new epoch to 1.575 at the final epoch, while validation loss rises from S0 3.533 to 4.422. The model fits the finite training pool better and generalizes to held-out text worse.','bullet')
    add('supports','This supports repeated finite-corpus fitting as a contributor to the 20%/50% reversal. The 20% mixtures expose each real block about forty additional times on average, versus twenty-five at 50% and fifty in the real-only control. These are repeated passages, not ever-growing supplies of fresh real text.','bullet')
    add('causality','It does not prove repetition is the sole cause or that 50/50 is optimal. Mixture fraction also changes self-generated targets; each cycle restarts optimization at the original learning rate, and epoch 5 is retained despite worsening validation. The real-only GPU differs, and only one seed was run.','bullet')
    add('format','7. Repetition and formatting diagnostics','heading2',True)
    add('format_image','repetition_and_formatting','image')
    add('format_cap','Figure 25a. Pooled repeated four-grams (A), mean within-output entropy (B), endpoint formatting-token shares (C) and within-output repetition distributions (D). More lexical variety can coexist with worse real-test probabilities.','caption')
    add('formatting','Real-only has no duplicate or whitespace-only outputs in the probe; newline and standalone-space shares are 0.52% and 0.62%. Synthetic-only has 21 duplicate outputs and 38.62% newline tokens. The mixed endpoints avoid the large newline concentration but retain roughly 9% standalone-space tokens.','bullet')
    add('pairedrep','Real-only is less repetitive on the same prompt than synthetic-only in 197/256 cases, 50/50 in 220 and 20/80 in 217. It is still not free of repetition, and this prompt-level advantage establishes neither factual correctness nor broad task competence.','bullet')
    add('reference','8. Frozen-S0 continuation scores','heading2',True)
    add('reference_image','frozen_s0_scores','image')
    add('reference_cap','Figure 25b. Mean (E) and 95th-percentile (F) NLL assigned by frozen S0 to each generated continuation alone. Teacher-force its 64 tokens, score 63 targets and omit the original prompt. Lower means more predictable to S0.','caption')
    add('reference_value','Real-only mean NLL is 2.063 versus 2.238 for synthetic-only, 1.562 for 50/50 and 1.653 for 20/80. Its P95 is 2.859. The mean and tail assess different aspects of the distribution and can rank conditions differently.','bullet')
    add('reference_meaning','S0 judges different strings produced by each endpoint. Varied text may be harder to predict than a repetitive loop; a low score can therefore reward predictable degeneration. This is not a distance between model weights, a factuality score or a second measurement on the fixed real test set.','bullet')
    add('examples','9. Exact same-prompt examples','heading2',True)
    ex=read_json(analysis/'examples.json')[0]
    add('example_intro','Prompt 0 is the historical fixed anchor, chosen before seeing these results. Factual claims in all continuations are unverified; reduced looping is not evidence that dates, platforms or developers are correct.')
    add('prompt',ex['prompt'],'quote')
    add('examples_cap','Table 29. Complete saved 64-token continuations at cycle 10. Within-output repeated-4 is 73.77% (100%), 86.89% (50%), 37.70% (20%) and 4.92% (real-only) on this prompt. It illustrates a local result, not aggregate representativeness.','caption')
    add('examples_table','examples','table')
    add('example_note','The real-only text is less trapped in a repeated phrase here, but it still contains suspicious factual claims and ends mid-word/token sequence. Fixed-length generation does not guarantee a complete sentence. Full prompt-level examples and original token IDs are saved.','small')
    add('endpoint','10. Cycle-10 outcomes versus synthetic fraction','heading2',True)
    add('endpoint_image','endpoints_vs_synthetic_fraction','image')
    add('endpoint_cap','Figure 26. Requested endpoint summary: final real-test PPL (A), pooled token entropy (B), pooled distinct-2 (C) and mean within-output repeated four-grams (D) versus synthetic block fraction. Dashed lines mark S0. Connecting segments join four measured points; no optimum is fitted.','caption')
    add('endpoint_ppl','Among these four seeded endpoints, 50% has the lowest test PPL. Moving from 100% toward 50% helps prediction, but continuing to 20% and 0% reverses that direction. The result is non-monotonic; the unsampled fractions and other seeds remain unknown.','bullet')
    add('endpoint_output','Output statistics tell a different story: pooled entropy decreases as synthetic share increases (6.082 → 5.706 → 5.590 → 4.431 nats), while within-output repetition rises (17.28 → 46.73 → 50.24 → 57.03%). Distinct-2 has a non-monotonic intermediate ordering. These output trends coexist with the non-monotonic PPL result.','bullet')
    add('conclusions','11. Conclusions, limits and reproducibility','heading2',True)
    add('finding1','Finding 1: synthetic data are not required for deterioration under this repeated five-epoch continued-training schedule. Real-only test PPL increases by 149.56% relative to S0.','bullet')
    add('finding2','Finding 2: the 50/50 and 20/80 mixtures outperform both extremes on final real-test PPL in this comparison. This is consistent with a trade-off between fitting the finite real pool and learning from evolving synthetic targets, but does not identify the mechanism or establish that synthetic data are universally beneficial.','bullet')
    add('finding3','Finding 3: predictive generalization and output diversity must be reported separately. The real-only control is the least repetitive endpoint on the fixed probe despite its worse PPL than either mixture.','bullet')
    add('limit','Limits: one seed per condition, deterministic beam decoding, short contexts and a probe made from the first 256 consecutive training prompts spanning six adjacent articles. Hardware differs for real-only. No statistical significance or universally safe mixture fraction is claimed.','bullet')
    add('next','To identify the mechanism more sharply, useful controls are additional seeds and matched-hardware runs, and independently varying the amount of reused versus fresh real text while holding the total training budget fixed. None was run for this report.','bullet')
    add('timing','Execution audit: all fifty full-pool epochs ran in 48.54 minutes wall-clock, including 37.47 minutes of training loops. The earlier 8.51-hour estimate counted unused full synthetic generation; removing it yields 2.22 hours on the old benchmark and 1.71 hours on the actual GPU, both with conservative overhead. Compute stopped automatically at 16:37:06 UTC on 2 November.','small')
    add('evidence','Evidence: all 44 per-cycle metric rows, 200 epoch histories, 41 unique fixed probes, matching S0/evaluation arrays, deterministic mixture selections and model/data handoffs were audited. Ten new probes total 656,640 bytes. Earlier publications and report pages remain preserved.','small')
    add('scope','This section was prepared from pinned backups and offline records. No Runpod/SSH, training, paid compute or cloud-resource changes were used. A separate timing audit reconstructed all real training datasets. Full checkpoint restoration and independent model re-evaluation remain outside this analysis.','small')
    add('source','Per-cycle metrics, epoch histories, source counts, paired test losses, exact examples, timing audit and reproducible square-panel figures.','link')
    specs=[('Real-test PPL','test_perplexity','.2f',1),('Validation PPL','validation_perplexity','.2f',1),('Test NLL','test_nll','.3f',1),
        ('Distinct-2 (%)','distinct_2','.2f',100),('Distinct-3 (%)','distinct_3','.2f',100),('Pooled entropy','token_entropy_nats','.3f',1),
        ('Within-output entropy','mean_within_entropy_nats','.3f',1),('Within repeated-4 (%)','mean_within_repeated_4gram_rate','.2f',100),
        ('Pooled repeated-4 (%)','repeated_4gram_rate','.2f',100),('Duplicates / 256','duplicate_outputs','.0f',1),
        ('Newline tokens (%)','newline_token_50118_share','.2f',100),('Space tokens (%)','standalone_space_share','.2f',100),
        ('Frozen-S0 mean NLL','frozen_s0_nll_mean','.3f',1),('Frozen-S0 P95 NLL','frozen_s0_nll_p95','.3f',1)]
    tables={'composition':[['Synthetic / real','Real blocks','Synthetic blocks','Input tokens','Mean real passes']]+[
        [label,f'{x["real_blocks_per_cycle"]:,}',f'{x["synthetic_blocks_per_cycle"]:,}','2,428,544',f'{x["mean_additional_real_block_passes"]:.1f}']
        for label,x in zip(['100% / 0%','50% / 50%','20% / 80%','0% / 100%'],s['exposures'])],
        'endpoints':[['Measure','S0','100%','50%','20%','0% (real)']]+[[name,*[format(row[key]*scale,fmt) for row in (z,b,f,m,r)]] for name,key,fmt,scale in specs],
        'examples':[['Final model','Complete saved continuation']]+[[label,ex['outputs'][arm]['text']] for arm,label in [('synthetic_100','100% synthetic'),('mixed_50','50% synthetic'),('mixed_20','20% synthetic'),('real_0','0% synthetic / real-only')]]}
    images={p.stem:str(p.resolve()) for p in (analysis/'figures').glob('*.png')}
    return {'entries':entries,'tables':tables,'images':images,'source_url':f'https://github.com/BillLeoutsakosvl346/Model_Collapse_in_Synthetic_Data_Training/blob/main/reports/{RUN}/report.md'}

def markdown(d,directory):
    lines=[]
    for e in d['entries']:
        k,t=e['kind'],e['text']
        if k=='table':
            rows=d['tables'][t];lines+=['| '+' | '.join(rows[0])+' |','| '+' | '.join(['---']*len(rows[0]))+' |']
            lines+=['| '+' | '.join(x.replace('\n','\\n').replace('|','\\|') for x in row)+' |' for row in rows[1:]]
        elif k=='image':lines += [f'![Real-only comparison: {t}]({Path(os.path.relpath(d["images"][t],directory)).as_posix()})']
        elif k=='bullet':lines += ['- '+t]
        elif k=='quote':lines += ['> '+t.replace('\n','\\n')]
        elif k=='link':
            relative=lambda p:Path(os.path.relpath(p,directory)).as_posix()
            lines += [f'[Metrics]({relative(ROOT/"analysis/metrics.csv")}) | [Epoch histories]({relative(ROOT/"analysis/training_history.csv")}) | [Examples]({relative(ROOT/"analysis/examples.json")}) | [Timing audit]({relative(ROOT/"timing_audit.md")})']
        else:lines += [{'heading1':'# ','heading2':'## '}.get(k,'')+t]
        lines += ['']
    return '\n'.join(lines)

def build(analysis,out):
    analysis=Path(analysis).resolve();out=new_destination(ROOT,out);out.mkdir(parents=True);d=draft(analysis)
    (out/'draft.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'report.md').write_text(markdown(d,out),encoding='utf-8')
    write_provenance(PROJECT,out,__file__,[analysis/'summary.json',analysis/'examples.json',analysis/'publication_reproduction.json',*map(Path,d['images'].values())],scope='Authored real-only/four-fraction comparison from frozen analysis; no cloud operations.')
    print(json.dumps({'entries':len(d['entries']),'tables':len(d['tables']),'figures':len(d['images'])}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,default=ROOT/'analysis');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();build(a.analysis,a.out)
