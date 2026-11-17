"""Build native Slides/Sheets requests from preserved research CSVs (no inference).

Run from the repository root. A full native Slides snapshot is required only for
exemplar geometry; source records are read without modifying historical bytes.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAVY, GRAY, TEAL, RED = '#172638', '#526173', '#007F7F', '#CF3B3B'
PURPLE, BLUE, AMBER, LIGHT, WHITE = '#7651AF', '#2876BD', '#B77B0B', '#CAD5DF', '#FFFFFF'
MINT = '#67DDD1'
ARMS = ['real', 'synthetic', 'mixed99', 'mixed80', 'mixed50']
COLORS = [TEAL, RED, PURPLE, AMBER, BLUE]
LABELS = ['0%', '100%', '99%', '80%', '50%']
SOURCE = ROOT / 'reports/20261114T150912Z-full-completion64-mixtures/generated/source-effects-20261114-v2'
FORK = ROOT / 'reports/20261116T195118Z-epoch-ablation/generated/verified-20261116'
DOC = 'https://docs.google.com/document/d/1WyQia9SBCOtT9_X4X56BpTihz-X9q9LWZFVapFEB4uk/edit'
REPO = 'https://github.com/BillLeoutsakosvl346/Model_Collapse_in_Synthetic_Data_Training'


def rgb(h):
    return {k: int(h[i:i+2], 16)/255 for k, i in [('red', 1), ('green', 3), ('blue', 5)]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--template-json', required=True, type=Path)
    p.add_argument('--sheet-id', required=True)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sources = {}

    def read(path):
        sources[str(path.relative_to(ROOT)).replace('\\', '/')] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))

    trajectory = read(SOURCE/'test_trajectory.csv')
    loss = read(ROOT/'reports/20261114T001352Z-full-completion64-mixtures/generated/loss-shares-20261114/cycle_loss_shares.csv')
    effects = read(SOURCE/'cycle_effects.csv')
    gradients = read(SOURCE/'cycle_source_summary.csv')
    outputs = read(SOURCE/'output_metrics.csv')
    forks = read(FORK/'per_update.csv')
    endpoints = read(FORK/'endpoints.csv')
    four = read(ROOT/'reports/20261108T173505Z-full-completion64/analysis/four-condition/metrics.csv')
    adam = read(ROOT/'reports/20261111T210854Z-full-completion64-mixtures/generated/smollm2-20261112/metrics.csv')
    opt = read(ROOT/'reports/20261107T214406Z-full-completion64-mixed99p9/analysis/seven-fractions/metrics.csv')
    decoders = read(ROOT/'reports/20261110T234038Z-full-completion64-decoding/generated/decoding-20261111/metrics.csv')
    raw = json.loads(a.template_json.read_text(encoding='utf-8'))
    deck = raw.get('structuredContent', raw)
    oldslides = {s['objectId']: s for s in deck['slides']}
    oldelements = {e['objectId']: e for s in oldslides.values() for e in s.get('pageElements', [])}
    S, C, slides, notes, outline, geometry = [], [], [], {}, [], []
    sid, blocks, cursor, chartids = 20261117, {}, 0, []
    C.append({'addSheet': {'properties': {'sheetId': sid, 'title': 'SGD and ablations', 'index': 0, 'gridProperties': {'rowCount': 1600, 'columnCount': 28, 'frozenRowCount': 1}}}})

    def get(rows, arm, cycle, key, arm_key='arm'):
        r = [r for r in rows if r[arm_key] == arm and int(r['cycle']) == cycle]
        assert len(r) == 1, (arm, cycle, key, len(r))
        return float(r[0][key])

    def block(name, header, rows):
        nonlocal cursor
        data = [header] + rows
        b = {'start': cursor, 'end': cursor+len(data), 'headers': header, 'rows': rows}
        blocks[name] = b
        vals = [{'values': [{'userEnteredValue': {'numberValue': v} if isinstance(v, (int, float)) else {'stringValue': str(v)}} for v in row]} for row in data]
        C.append({'updateCells': {'start': {'sheetId': sid, 'rowIndex': cursor, 'columnIndex': 0}, 'rows': vals, 'fields': 'userEnteredValue'}})
        C.append({'repeatCell': {'range': {'sheetId': sid, 'startRowIndex': cursor, 'endRowIndex': cursor+1, 'startColumnIndex': 0, 'endColumnIndex': len(header)}, 'cell': {'userEnteredFormat': {'backgroundColor': rgb(NAVY), 'textFormat': {'bold': True, 'foregroundColor': rgb(WHITE)}}}, 'fields': 'userEnteredFormat'}})
        cursor += len(data)+3

    def rng(name, col):
        b = blocks[name]
        return {'sourceRange': {'sources': [{'sheetId': sid, 'startRowIndex': b['start'], 'endRowIndex': b['end'], 'startColumnIndex': col, 'endColumnIndex': col+1}]}}

    def makechart(name, cols, colors, yt, lo, hi, xt='Training cycle', xlo=0, xhi=10, dashed=None, points=3):
        cid = 1100+len(chartids)
        series = []
        for j, (col, color) in enumerate(zip(cols, colors)):
            series.append({'series': rng(name, col), 'targetAxis': 'LEFT_AXIS', 'colorStyle': {'rgbColor': rgb(color)}, 'lineStyle': {'width': 3, 'type': (dashed or {}).get(j, 'SOLID')}, 'pointStyle': {'shape': 'CIRCLE', 'size': points}})
        spec = {'fontName': 'Arial', 'backgroundColorStyle': {'rgbColor': rgb(WHITE)}, 'basicChart': {'chartType': 'LINE', 'legendPosition': 'NO_LEGEND', 'headerCount': 1, 'lineSmoothing': False, 'domains': [{'domain': rng(name, 0)}], 'axis': [{'position': 'BOTTOM_AXIS', 'title': xt, 'format': {'fontSize': 18}, 'viewWindowOptions': {'viewWindowMin': xlo, 'viewWindowMax': xhi, 'viewWindowMode': 'EXPLICIT'}}, {'position': 'LEFT_AXIS', 'title': yt, 'format': {'fontSize': 18}, 'viewWindowOptions': {'viewWindowMin': lo, 'viewWindowMax': hi, 'viewWindowMode': 'EXPLICIT'}}], 'series': series}}
        C.append({'addChart': {'chart': {'chartId': cid, 'spec': spec, 'position': {'overlayPosition': {'anchorCell': {'sheetId': sid, 'rowIndex': len(chartids)*25, 'columnIndex': 16}, 'widthPixels': 600, 'heightPixels': 600}}}}})
        chartids.append(cid)
        return cid

    block('sgd_log', ['Cycle']+[f'{x} synthetic: log10 test PPL' for x in LABELS], [[c]+[math.log10(get(trajectory, arm, c, 'test_perplexity')) for arm in ARMS] for c in range(11)])
    block('sgd_linear', ['Cycle']+[f'{x} synthetic: test PPL' for x in LABELS], [[c]+[get(trajectory, arm, c, 'test_perplexity') for arm in ARMS] for c in range(11)])
    block('loss', ['Cycle', 'Real share of mean batch loss (%)'], [[int(r['cycle']), float(r['real_loss_share_percent'])] for r in loss])
    block('effects99', ['Cycle', 'Real conditional effect', 'Synthetic conditional effect'], [[c, get(effects, 'mixed99', c, 'real_conditional_mean_delta_nll'), get(effects, 'mixed99', c, 'synthetic_conditional_mean_delta_nll')] for c in range(1, 11)])
    block('norms99', ['Cycle', 'Weighted real norm', 'Weighted synthetic norm', 'Combined norm'], [[c]+[float(next(r[k] for r in gradients if r['arm']=='mixed99' and int(r['cycle'])==c and r['population']=='all_updates')) for k in ['mean_real_gradient_norm','mean_synthetic_gradient_norm','mean_combined_gradient_norm']] for c in range(1,11)])
    block('cosines', ['Cycle', '50% synthetic', '80% synthetic', '99% synthetic'], [[c]+[float(next(r['median_source_cosine'] for r in gradients if r['arm']==arm and int(r['cycle'])==c and r['population']=='all_updates')) for arm in ['mixed50','mixed80','mixed99']] for c in range(1,11)])
    for c in [3,6,10]:
        f = [r for r in forks if int(r['cycle'])==c]
        b = math.exp(float(f[0]['probe_nll_before']))
        rows = [[0,b,b,b,b]]
        for step in range(1,80):
            v = {src:float(next(r['probe_perplexity'] for r in f if int(r['local_step'])==step and r['source']==src)) for src in ['both','real','synthetic']}
            rows.append([step, v['both'], v['real'], v['synthetic'], .5*(v['real']+v['synthetic'])])
        block(f'fork{c}', ['Update within epoch','Both sources','Real only','Synthetic only','Arithmetic mean of two branch PPLs'], rows)
    for key in ['mean_within_entropy_nats','mean_within_repeated_4gram_rate','distinct_2','distinct_3','token_entropy_nats']:
        scale = 100 if key in ['mean_within_repeated_4gram_rate','distinct_2','distinct_3'] else 1
        block(key,['Cycle']+[f'{x} synthetic' for x in LABELS],[[c]+[scale*get(outputs,arm,c,key) for arm in ARMS] for c in range(11)])
    # Baseline-relative PPL avoids comparing numerical PPL across tokenizers.
    fractions=[0,.5,.8,.95,.99,.999,1]
    def fracrow(rows,frac,c):
        matches=[r for r in rows if int(r['cycle'])==c and abs(float(r['synthetic_fraction'])-frac)<1e-9]
        assert len(matches)>=1
        return matches[0]
    block('adam_endpoints',['Synthetic targets (%)','OPT: final / baseline PPL','SmolLM2: final / baseline PPL'], [[100*f,float(fracrow(opt,f,10)['test_perplexity'])/float(fracrow(opt,f,0)['test_perplexity']),float(fracrow(adam,f,10)['test_perplexity'])/float(fracrow(adam,f,0)['test_perplexity'])] for f in fractions])
    C.append({'updateDimensionProperties': {'range': {'sheetId':sid,'dimension':'COLUMNS','startIndex':0,'endIndex':14},'properties':{'pixelSize':180},'fields':'pixelSize'}})

    def ep(s,x,y,w,h):
        return {'pageObjectId':s,'size':{'width':{'magnitude':w,'unit':'PT'},'height':{'magnitude':h,'unit':'PT'}},'transform':{'scaleX':1,'scaleY':1,'translateX':x,'translateY':y,'unit':'PT'}}
    def style(oid,size=None,bold=None,color=None,font=None):
        v={}
        if size is not None:v['fontSize']={'magnitude':size,'unit':'PT'}
        if bold is not None:v['bold']=bold
        if color:v['foregroundColor']={'opaqueColor':{'rgbColor':rgb(color)}}
        if font:v['fontFamily']=font
        S.append({'updateTextStyle':{'objectId':oid,'textRange':{'type':'ALL'},'style':v,'fields':','.join(v)}})
    def bounds(oid,source,x,y,w,h):
        sz=source['size']
        def pt(v):return v['magnitude']/(12700 if v['unit']=='EMU' else 1)
        S.append({'updatePageElementTransform':{'objectId':oid,'applyMode':'ABSOLUTE','transform':{'scaleX':w/pt(sz['width']),'scaleY':h/pt(sz['height']),'translateX':x,'translateY':y,'unit':'PT'}}})
    def replace(oid,txt):
        S.extend([{'deleteText':{'objectId':oid,'textRange':{'type':'ALL'}}},{'insertText':{'objectId':oid,'insertionIndex':0,'text':txt}}])
    def textbox(s,name,txt,x,y,w,h,size=20,bold=False,color=NAVY,font='Arial',bullets=False):
        oid=s+'_'+name
        S.extend([{'createShape':{'objectId':oid,'shapeType':'TEXT_BOX','elementProperties':ep(s,x,y,w,h)}},{'insertText':{'objectId':oid,'text':txt,'insertionIndex':0}},{'updateShapeProperties':{'objectId':oid,'shapeProperties':{'outline':{'propertyState':'NOT_RENDERED'},'autofit':{'autofitType':'NONE'}},'fields':'outline.propertyState,autofit.autofitType'}},{'updateParagraphStyle':{'objectId':oid,'textRange':{'type':'ALL'},'style':{'lineSpacing':110,'spaceAbove':{'magnitude':0,'unit':'PT'},'spaceBelow':{'magnitude':8 if bullets else 0,'unit':'PT'}},'fields':'lineSpacing,spaceAbove,spaceBelow'}}])
        style(oid,size,bold,color,font)
        if bullets:S.append({'createParagraphBullets':{'objectId':oid,'textRange':{'type':'ALL'},'bulletPreset':'BULLET_DISC_CIRCLE_SQUARE'}})
        geometry.append({'slide':s,'id':oid,'text':txt,'bounds':[x,y,w,h],'size':size})
        return oid
    def bg(s,color):S.append({'updatePageProperties':{'objectId':s,'pageProperties':{'pageBackgroundFill':{'solidFill':{'color':{'rgbColor':rgb(color)},'alpha':1}}},'fields':'pageBackgroundFill'}})
    def new(n,title,seconds,note,kind='text',backup=False,exemplar=None):
        s=f'iib17_s{n:02d}';slides.append(s)
        ex=exemplar or ('iib26_s07' if kind=='plot' else 'iib26_s03')
        keep={'title','num'}
        if kind=='plot':keep|={'headline','endpoint','desc','key0','key1','foot','legendhead'}
        if kind=='cover':keep={'title','sub','by'}
        if kind=='conclusion':keep={'title','num','points','limits','fine'}
        mapping={ex:s}
        for e in oldslides[ex]['pageElements']:mapping[e['objectId']]=s+'_'+e['objectId'].removeprefix(ex+'_')
        S.append({'duplicateObject':{'objectId':ex,'objectIds':mapping}})
        media=[]
        for e in oldslides[ex]['pageElements']:
            suffix=e['objectId'].removeprefix(ex+'_')
            if suffix not in keep:S.append({'deleteObject':{'objectId':mapping[e['objectId']]}})
            if 'sheetsChart' in e:media.append({'source':e['objectId'],'action':'replace with a verified current native chart' if kind=='plot' else 'delete: not needed for destination evidence'})
        if kind!='cover':
            replace(s+'_title',title);style(s+'_title',30,True,WHITE if kind=='conclusion' else NAVY)
            replace(s+'_num',('B'+str(n-21) if backup else str(n)));style(s+'_num',10,False,GRAY)
        S.append({'updateSlideProperties':{'objectId':s,'slideProperties':{'isSkipped':backup},'fields':'isSkipped'}})
        if kind not in ['cover','conclusion']:bg(s,WHITE)
        outline.append({'slide_id':s,'title':title,'seconds':seconds,'backup':backup,'exemplar':ex,'construction':'duplicate exemplar; edit retained native objects','media_map':media})
        notes[s]=f'{seconds} seconds.\n\n{note}\n\nMain report: {DOC}\nReproducible inputs and interpretation: {REPO}/tree/main/presentations/2026-11-17_research-update\nSources: '+', '.join(sources)
        return s
    def footer(s,txt):
        oid=s+'_foot'
        if any(r.get('insertText',{}).get('objectId')==oid for r in S):replace(oid,txt);style(oid,9.5,False,GRAY)
        elif any(e['objectId'].endswith('_foot') for e in oldslides['iib26_s07']['pageElements']) and next(o for o in outline if o['slide_id']==s)['exemplar']=='iib26_s07':replace(oid,txt)
        else:textbox(s,'foot',txt,29,382,610,17,9.5,color=GRAY)
    def chart(s,cid,x=22,y=72,size=307,name='chart'):
        S.append({'createSheetsChart':{'objectId':s+'_'+name,'spreadsheetId':a.sheet_id,'chartId':cid,'linkingMode':'LINKED','elementProperties':ep(s,x,y,size,size)}})
    def callouts(s,head,val,desc,color=NAVY,keys=None,foot=''):
        for name,txt in [('headline',head),('endpoint',val),('desc',desc),('legendhead','Curve legend · cycle-10 PPL')]:replace(s+'_'+name,txt)
        style(s+'_headline',22,True,NAVY);style(s+'_endpoint',40,True,color);style(s+'_desc',19,False,NAVY)
        bounds(s+'_headline',oldelements['iib26_s07_headline'],351,80,335,54)
        bounds(s+'_endpoint',oldelements['iib26_s07_endpoint'],351,136,335,54)
        bounds(s+'_desc',oldelements['iib26_s07_desc'],352,198,334,73)
        bounds(s+'_legendhead',oldelements['iib26_s07_legendhead'],352,276,334,22)
        keys=keys or []
        for j in range(2):
            if j<len(keys):
                replace(s+f'_key{j}',keys[j][0]);style(s+f'_key{j}',14,True,keys[j][1])
                bounds(s+f'_key{j}',oldelements[f'iib26_s07_key{j}'],352+(j%2)*171 if len(keys)>3 else 352,300+(j//2)*23 if len(keys)>3 else 300+j*23,163 if len(keys)>3 else 334,23)
            else:S.append({'deleteObject':{'objectId':s+f'_key{j}'}})
        for j,(t,c) in enumerate(keys[2:],2):textbox(s,f'key{j}',t,352+(j%2)*171 if len(keys)>3 else 352,300+(j//2)*23 if len(keys)>3 else 300+j*23,163 if len(keys)>3 else 334,23,14,True,c)
        if foot:footer(s,foot)
    def table(s,name,rows,x,y,w,h,widths=None):
        oid=s+'_'+name
        S.append({'createTable':{'objectId':oid,'rows':len(rows),'columns':len(rows[0]),'elementProperties':ep(s,x,y,w,h)}})
        for ri,row in enumerate(rows):
            for ci,t in enumerate(row):
                S.append({'insertText':{'objectId':oid,'cellLocation':{'rowIndex':ri,'columnIndex':ci},'text':str(t),'insertionIndex':0}})
                S.append({'updateTextStyle':{'objectId':oid,'cellLocation':{'rowIndex':ri,'columnIndex':ci},'textRange':{'type':'ALL'},'style':{'fontFamily':'Arial','fontSize':{'magnitude':16 if ri else 15,'unit':'PT'},'bold':ri==0,'foregroundColor':{'opaqueColor':{'rgbColor':rgb(WHITE if ri==0 else NAVY)}}},'fields':'fontFamily,fontSize,bold,foregroundColor'}})
                S.append({'updateParagraphStyle':{'objectId':oid,'cellLocation':{'rowIndex':ri,'columnIndex':ci},'textRange':{'type':'ALL'},'style':{'spaceAbove':{'magnitude':0,'unit':'PT'},'spaceBelow':{'magnitude':0,'unit':'PT'}},'fields':'spaceAbove,spaceBelow'}})
        S.append({'updateTableCellProperties':{'objectId':oid,'tableRange':{'location':{'rowIndex':0,'columnIndex':0},'rowSpan':1,'columnSpan':len(rows[0])},'tableCellProperties':{'tableCellBackgroundFill':{'solidFill':{'color':{'rgbColor':rgb(NAVY)},'alpha':1}}},'fields':'tableCellBackgroundFill'}})
        if widths:
            for c,v in enumerate(widths):S.append({'updateTableColumnProperties':{'objectId':oid,'columnIndices':[c],'tableColumnProperties':{'columnWidth':{'magnitude':v,'unit':'PT'}},'fields':'columnWidth'}})

    # Main narrative. Consecutive native slides serve as presenter-controlled builds.
    s=new(1,'Why can 1% real data matter so much?',15,'Open with the empirical puzzle, not a universal threshold. These are small-language-model experiments, one seed. The title anticipates the source-loss result; it does not claim 1% is generally sufficient.',kind='cover',exemplar='iib26_s01')
    replace(s+'_title','When a model becomes\nits own teacher');style(s+'_title',42,True,WHITE)
    replace(s+'_sub','Why can 1% real data matter so much?');style(s+'_sub',25,False,MINT)
    replace(s+'_by','Bill Leoutsakos · Cambridge Engineering IIB\nRecursive synthetic-data training\nResearch update · 17 November 2026');style(s+'_by',16,False,LIGHT)

    s=new(2,'More text. But whose text?',25,'Motivation: the demand for useful human text is growing; do not claim that the internet has already been exhausted. Synthetic targets can expand training data, but model biases can feed back. The model does not answer factual Q&A here: it continues a real prompt. References: Villalobos et al. https://arxiv.org/abs/2211.04325; Shumailov et al. https://arxiv.org/abs/2305.17493; Seddik et al. https://arxiv.org/abs/2404.05090.')
    textbox(s,'lead','Synthetic text scales.\nIts mistakes can scale too.',33,91,650,84,30,True,TEAL)
    textbox(s,'loop','Model → generated targets → training → next model',33,208,650,45,22,True)
    textbox(s,'question','Does fresh real text interrupt this feedback?',33,291,650,61,25,True)
    footer(s,'Model collapse: degradation of the learned distribution through recursive synthetic training')

    s=new(3,'Rising perplexity is not a diagnosis',35,'Week 1–3 exposed confounders: resetting to original pretrained OPT differs from resetting to real-trained S0; five epochs executed differed from selecting epoch 1. Repeating the WikiText real set alone raised test PPL from 33.89 to 84.56 over ten five-epoch cycles, with falling training loss. We then matched fresh/repeated real/synthetic continuations on FineWeb. The table reports completed OPT/AdamW/cosine outcomes, not the newer SGD experiment. Repeated synthetic means regenerated targets each epoch. Higher PPL means poorer prediction of real held-out text; it alone cannot uniquely identify collapse or overfitting.')
    textbox(s,'lead','PPL = exp(mean −log p(real next token))',33,81,650,40,25,False,TEAL,'Georgia')
    textbox(s,'context','Higher means more surprise at held-out real text.',33,126,650,34,20)
    table(s,'controls',[['OPT / FineWeb · cycle 10','Real targets','Synthetic targets'],['Fresh prompts each cycle','28.05','342.01'],['Same prompts repeated','28.61','215.39']],33,188,654,124,[326,164,164])
    textbox(s,'takeaway','WikiText real-only reuse: PPL 33.9 → 84.6.',33,331,650,39,21,True)
    footer(s,'Table: FineWeb / OPT / AdamW / cosine, baseline 29.18. WikiText: ten cycles × five real-only epochs.')

    s=new(4,'One matched prediction task',40,'The current experiments use pretrained SmolLM2-135M and a separate native-tokenizer FineWeb-Edu bank. Ten fresh, document-disjoint chunks of 10,000 pairs. A pair contains 64 real prompt tokens and 64 continuation tokens. BOS plus both gives 129 input positions; prompt attention stays active, but only the 64 continuation tokens contribute to loss. Every condition uses identical prompts and held-out real continuations. Synthetic targets come from that condition’s preceding checkpoint and are frozen before training. Independent arms start from identical original pretrained weights. Full test = 7,303 pairs / 467,392 scored targets. SmolLM2 pretraining includes FineWeb-Edu, so experimental separation is not proof of pretraining novelty.')
    textbox(s,'model','SmolLM2-135M  /  FineWeb-Edu',33,82,650,35,24,True,TEAL)
    textbox(s,'prompt','64 prompt tokens',33,145,300,32,24,True)
    textbox(s,'target','64 continuation tokens',366,145,321,32,24,True,PURPLE)
    textbox(s,'context','Real context, visible in both arms',33,188,290,56,19,color=GRAY)
    textbox(s,'loss','Real text OR the preceding\ncheckpoint’s generated answer',366,188,321,63,19,color=GRAY)
    textbox(s,'recipe','10 fresh chunks × 10,000 examples\nOne epoch per chunk; continue weights\nFixed real test after every epoch',33,281,650,88,20,bullets=True)

    s=new(5,'Strip the optimizer back to SGD',25,'The previous AdamW/cosine sweeps motivated a simpler recipe. This is not an optimizer-only causal comparison: optimizer, LR schedule, regularization, precision and decoding changed together. Within the new sweep only target fraction changes. SGD LR 0.01 was selected using the recorded bounded real-only calibration, not test or synthetic outcomes. No momentum, weight decay, clipping, warm-up or scheduler. FP32 with TF32 disabled. Seed 42, microbatch 32 and effective batch 128; 79 updates per cycle including a final 16-example batch, 790 per arm. Greedy deterministic generation, exactly 64 new tokens, repetition penalty 1.0.')
    bg(s,NAVY);style(s+'_title',30,True,WHITE)
    eq=textbox(s,'eq','θnext = θ − 0.01 × ∇L',39,110,650,66,33,False,WHITE,'Georgia')
    S.append({'updateTextStyle':{'objectId':eq,'textRange':{'type':'FIXED_RANGE','startIndex':1,'endIndex':5},'style':{'baselineOffset':'SUBSCRIPT','fontSize':{'magnitude':22,'unit':'PT'}},'fields':'baselineOffset,fontSize'}})
    textbox(s,'simple','Constant learning rate · no optimizer memory',39,207,640,43,25,True,MINT)
    textbox(s,'facts','FP32 · effective batch 128 · greedy generation\n790 updates per condition · same held-out evaluation',39,278,640,70,20,color=WHITE)
    footer(s,'Within this comparison only the real/synthetic target fraction changes. One seed.');style(s+'_foot',9.5,False,LIGHT)

    for i,(arm,label,color) in enumerate(zip(ARMS,LABELS,COLORS)):
        val=get(trajectory,arm,10,'test_perplexity')
        desc={'real':'Fresh real targets keep\nperformance near baseline.','synthetic':'Self-training deteriorates\nat every recorded cycle.','mixed99':'Just 100 real examples\nsubstantially change the path.','mixed80':'More real targets: an early\nrise, then partial recovery.','mixed50':'Half real targets stays\nclose to real-only training.'}[arm]
        s=new(6+i,'The real-data fraction changes the path',15 if i else 20,f'Build {i+1}/5. Added {label} synthetic targets. Complete fixed real-test continuation PPL ends at {val:.8f}. All curves use the same log10 vertical scale and position, with raw PPL values in the legend. The stored chart series are explicitly log10(PPL); a 1-unit rise is a tenfold PPL increase. Shared pretrained baseline 17.95797667. No learning-rate decrease. A finite ten-cycle trajectory does not establish permanent stabilization. 99% remains much worse than real-only; it does not prevent all deterioration.',kind='plot')
        cid=makechart('sgd_log',list(range(1,i+2)),COLORS[:i+1],'log₁₀(test perplexity)',1,3.3)
        chart(s,cid)
        keys=[(f'{LABELS[j]}     {get(trajectory,ARMS[j],10,"test_perplexity"):,.2f}',COLORS[j]) for j in range(i+1)]
        callouts(s,label+' synthetic targets',f'{val:,.2f}',desc,color,keys,'Same log₁₀ scale throughout. Axis 1 = PPL 10; 2 = 100; 3 = 1,000. Lower is better.')
        replace(s+'_legendhead','Synthetic share · cycle-10 PPL')

    s=new(11,'1% of examples ≠ 1% of training loss',45,'The real curve reports the ratio of mean source-weighted loss terms across all 79 updates in each cycle, including tails. At cycle 10 it is 49.08766794%; token-weighting all examples instead gives 47.4622%. Do not confuse this scalar loss share with gradient share or beneficial influence. Cycle-10 source mean NLLs are about 5.333 real and 0.0596 synthetic: the model confidently predicts many of its own targets, while real continuations remain surprising. Real rows are 1% over the epoch, not necessarily in every batch. This observation motivates measuring actual gradients and held-out consequences.',kind='plot')
    chart(s,makechart('loss',[1],[TEAL],'Real share of mean batch loss (%)',0,55))
    callouts(s,'Cycle 10: real loss share','49.1%','Real targets stay surprising.\nSelf-generated targets become\neasy to predict.',TEAL,[('Real examples: 1% of rows',TEAL),('Cycle 1 loss share: 6.5%',GRAY)],'Ratio of mean batch-loss terms. Corpus-token-weighted cycle-10 share: 47.5%.')
    replace(s+'_legendhead','What this percentage measures')

    s=new(12,'Three quantities. Three different questions.',40,'For equal 64-target examples, L_R = (1/B) sum over real rows of their mean continuation NLL, with the analogous synthetic term. Source proportions are already in those terms. Gradient vectors add: g = grad L_R + grad L_S. The plotted norm of their sum is not the sum of norms. SGD subtracts 0.01 times the combined gradient. Finally, unseen-real loss depends on direction and local curvature, not just training loss or vector magnitude. Nearly half the loss does not mean nearly half the gradient or accuracy gain.')
    for y,eq,meaning,c in [(92,'L = LR + LS','Loss: how surprising are the training targets?',NAVY),(177,'g = gR + gS','Gradient: how do the sources move the weights?',PURPLE),(262,'ΔNLL on unseen real text','Effect: does that movement help generalization?',TEAL)]:
        oid=textbox(s,f'eq{y}',eq,33,y,650,43,28,False,c,'Georgia')
        if y<200:
            for at,ch in enumerate(eq):
                if ch in 'RS':S.append({'updateTextStyle':{'objectId':oid,'textRange':{'type':'FIXED_RANGE','startIndex':at,'endIndex':at+1},'style':{'baselineOffset':'SUBSCRIPT','fontSize':{'magnitude':19,'unit':'PT'}},'fields':'baselineOffset,fontSize'}})
        textbox(s,f'meaning{y}',meaning,33,y+44,650,32,19,color=GRAY)
    footer(s,'Source-weighted terms already include the batch proportions. Never multiply by 1% a second time.')

    s=new(13,'Does the real contribution actually help?',45,'At each original SGD update, measure a fixed 128-pair real-validation probe using temporary copies from identical preupdate tensors. Q_M is the actual mixed update, Q_S synthetic alone, Q_R real alone. Real conditional effect = NLL(Q_M)-NLL(Q_S); synthetic conditional effect = NLL(Q_M)-NLL(Q_R). Below zero improves probe predictions. Real inclusion helps all 563 updates containing real examples. Synthetic inclusion worsens 778/790 in this 99% trajectory. Curves are means across all 79 updates including zero real contributions in absent-source batches. Effects do not add because of nonlinear interaction. Probe results are not complete-test PPL and no held-out gradients train the model.',kind='plot')
    chart(s,makechart('effects99',[1,2],[TEAL,RED],'Conditional ΔNLL / update',-.055,.055))
    callouts(s,'99% synthetic: local effects','563 / 563','Real inclusion improves\nthe probe in every update\ncontaining real examples.',TEAL,[('Real inclusion: below 0 helps',TEAL),('Synthetic inclusion: above 0 hurts',RED)],'Same 128 held-out real pairs. Conditional effects, averaged across updates; they do not add.')
    replace(s+'_legendhead','Measured intervention, not loss share')

    s=new(14,'Let each source act for a whole epoch',35,'We restore M2, M5 and M9 of the historical 99% SGD arm and replay cycle 3, 6 and 10. Each case starts three independent copies with identical saved weights, RNG, order and exact targets. Both-source replay passes terminal/probe/held-out discrepancies <=1e-5 before interpreting ablations. Keep the original batch denominator; real-only is not renormalized into full-strength real training. 100 real and 9,900 saved synthetic targets; do not regenerate. Each branch takes 79 scheduled positions including zero-movement absent-source batches. These are one-epoch forks, not three complete ten-cycle trajectories.')
    textbox(s,'checkpoints','Restore M2, M5 and M9',33,83,650,38,25,True,PURPLE)
    textbox(s,'both','Both',41,163,192,35,26,True,PURPLE)
    textbox(s,'real','Real only',270,163,205,35,26,True,TEAL)
    textbox(s,'synthetic','Synthetic only',491,163,208,35,26,True,RED)
    textbox(s,'explain','Same saved targets · same batch order\nRemove source terms, retain the original denominator\nThree independently evolving copies for one epoch',33,237,650,106,21,bullets=True)
    footer(s,'100 real + 9,900 synthetic rows. Replay verified before interpreting the source removals.')

    for n,c,cols in [(15,3,[1]),(16,3,[1,3]),(17,3,[1,3,2]),(18,6,[1,3,2]),(19,10,[1,3,2])]:
        s=new(n,f'Cycle {c}: remove a source, change the path',20 if n in [15,16] else 30,f'Whole-epoch fork at cycle {c}, step 0 is the common restored checkpoint. All curves are fixed 128-pair real-probe PPL after each update. Purple both-source replay, dotted dark grey synthetic-only, dashed light grey real-only. The three branches diverge independently, so their endpoint or arithmetic mean is not a prediction of the combined path. Original source weights are retained. Complete-test endpoints show the same ordering. Cycle 6 and 10 both-source late drops include one real example in the final 16-example batch; real-only already improves strongly before that tail. One seed and deliberately selected cases.',kind='plot')
        cid=makechart(f'fork{c}',cols,[{1:PURPLE,2:'#8899AA',3:'#384957'}[i] for i in cols],'128-pair probe perplexity',30,340,'Update within epoch',0,79,dashed={j:('MEDIUM_DASHED' if col==2 else 'DOTTED' if col==3 else 'SOLID') for j,col in enumerate(cols)},points=0)
        chart(s,cid)
        epc={r['source']:float(r['probe_ppl_after']) for r in endpoints if int(r['cycle'])==c}
        keys=[(f'Both             {epc["both"]:.2f}',PURPLE)]
        if 3 in cols:keys.append((f'Synthetic only   {epc["synthetic"]:.2f}','#384957'))
        if 2 in cols:keys.append((f'Real only        {epc["real"]:.2f}','#8899AA'))
        headline='Shared start → three branches'
        desc='Same original source weights.\nRemoving real terms worsens\nthe one-epoch outcome.' if n>=17 else 'First replay the actual epoch.\nThen remove one source\nfrom the saved training set.'
        if n==18:desc='Real-only improves well\nbefore the final batch.\nThe correction is directional.'
        if n==19:desc='One real example in the\n16-row final batch creates\na sharp correction.'
        callouts(s,headline,f'{epc["both"]:.2f}',desc,PURPLE,keys,'Fixed real probe, not full-test PPL. Same 30–340 axes in all three cases.')
        replace(s+'_legendhead','Branch endpoints · probe PPL')

    s=new(20,'Deterioration is visible beyond perplexity',25,'Output metrics use 256 fixed real-validation prompts, greedy decoding and 64 generated tokens at every checkpoint. Left is mean within-output empirical token entropy; lower means each output concentrates on fewer token types. Right is mean within-output repeated four-gram percentage; higher means repeated phrases inside a continuation. Neither is predictive entropy or factual accuracy. The real-only model can already repeat under deterministic decoding. Pooled distinct-n and pooled entropy answer different questions; see backup. Show 0/99/100 to keep the main figure readable.')
    for j,arm in enumerate(['real','mixed99','synthetic']):textbox(s,f'legend{j}',{'real':'0% synthetic','mixed99':'99% synthetic','synthetic':'100% synthetic'}[arm],35+j*222,76,217,24,15,True,{'real':TEAL,'mixed99':PURPLE,'synthetic':RED}[arm])
    chart(s,makechart('mean_within_entropy_nats',[1,3,2],[TEAL,PURPLE,RED],'Within-output entropy (nats)',0,4.2),18,101,280,'entropy')
    chart(s,makechart('mean_within_repeated_4gram_rate',[1,3,2],[TEAL,PURPLE,RED],'Repeated 4-grams within output (%)',0,100),369,101,280,'repetition')
    footer(s,'Same 256 prompts. Left: token variety inside an output. Right: phrase repetition inside an output.')

    s=new(21,'A small real fraction can carry a large correction',35,'Conclude with measured scope. Self-generated deterministic targets become easy, leaving little training loss while real targets remain informative. Real inclusion produces a beneficial local held-out effect; whole-epoch source removal reinforces that directional explanation in the selected cases. This is not a universal threshold, permanent stabilization or proof of a stationary point. Tail batches and order matter, especially at 99%. New seeds, longer runs, fixed-size tail handling and stochastic decoding are the next controlled tests. Existing checkpoints, configurations, exact data/provenance and analyses are saved in the repository and private recovery storage.',kind='conclusion',exemplar='iib26_s18')
    replace(s+'_title','What explains the gap?')
    replace(s+'_points','Synthetic targets become easy to predict.\nReal targets remain a strong corrective signal.\nRemoving that signal changes the next epoch.')
    style(s+'_points',25,False,WHITE)
    replace(s+'_limits','Not a universal 1% rule. Not proof of permanent stability.');style(s+'_limits',23,True,MINT)
    replace(s+'_fine','One seed, ten cycles; 16-example tails matter.\nNext: tail handling, new seeds, longer horizons and decoding.');style(s+'_fine',16,False,LIGHT)

    # Skipped backup evidence: optional discussion, not part of the ten-minute talk.
    s=new(22,'From reproduction to mechanism',0,'Historical roadmap, not a comparison across identical settings. Week 1 original-pretrained OPT reset; Week 2 distinguishes continued weights from fixed real-trained S0 and matches shared starts; Week 3 retains final epoch 5, then varies real fraction; FineWeb fresh data and matched completions remove prompt/loss asymmetries; SmolLM2 replicates the mixture effect; Week 5 constant-LR SGD and source diagnostics lead to whole-epoch ablations.',backup=True)
    table(s,'timeline',[['Stage','Question resolved'],['WikiText / OPT','Reset policy, selected epoch and overfitting confounders'],['Matched FineWeb','Fresh prompts; real versus generated continuations'],['Two small models','Fraction effect recurs; absolute PPL is tokenizer-specific'],['Constant-LR SGD','Measure loss, gradient and held-out effects separately'],['Whole-epoch forks','Test source removal along independently evolving paths']],33,93,654,250,[189,465])
    footer(s,'The full report preserves all earlier experiments; the main talk selects the clearest matched evidence.')

    s=new(23,'Two models show a similar fraction effect',0,'AdamW/cosine/five-beam sweeps. Baseline-relative PPL is descriptive; OPT and SmolLM2 have different tokenizers, native banks and pretraining. OPT high-fraction runs use RTX PRO 4500 while its main controls use 4000; SmolLM2 all 4000. This plot is not the new SGD experiment and does not isolate the optimizer.',kind='plot',backup=True)
    chart(s,makechart('adam_endpoints',[1,2],[GRAY,TEAL],'Cycle-10 PPL / baseline PPL',0,19,'Synthetic training targets (%)',0,100))
    callouts(s,'Earlier recipe','Two models','Near-total synthetic training\ndeteriorates much more than\nreal-heavy mixtures.',TEAL,[('OPT-125M',GRAY),('SmolLM2-135M',TEAL)],'AdamW + cosine LR + five beams. Baseline-relative comparison; not absolute model quality.')
    replace(s+'_legendhead','Model legend · baseline-relative PPL')

    s=new(24,'Lower-fraction SGD curves, expanded',0,'Linear scale reveals differences hidden by the full-range logarithmic plot. 0/50/80% end at 17.72566/18.58579/19.70688. The 80% arm peaks at 24.17467 in cycle 2, then partly recovers. Same scientific settings and fixed real test as the main sweep.',kind='plot',backup=True)
    chart(s,makechart('sgd_linear',[1,5,4],[TEAL,BLUE,AMBER],'Test perplexity',17,25))
    callouts(s,'Linear-axis detail','17–25','80% has an early peak.\n50% remains closer\nto real-only training.',AMBER,[('0%      17.73',TEAL),('50%     18.59',BLUE),('80%     19.71',AMBER)],'Same full-test metric. Expanded vertical scale is explicitly different from the main reveals.')

    s=new(25,'Gradient opposition is not zero movement',0,'Gradient norms are measured after source weighting, averaging all updates, with absent components zero. At cycle10 99% real=.297041, synthetic=.485282, combined=.533666. The combined norm can exceed the synthetic norm despite negative alignment. Relative to perpendicular same-strength vectors, the mean combined norm is 12.2% smaller. Geometry alone does not tell us held-out benefit; the separate probe interventions do.',kind='plot',backup=True)
    chart(s,makechart('norms99',[1,2,3],[TEAL,RED,PURPLE],'Mean source-weighted gradient norm',0,1))
    callouts(s,'99%: weighted gradient norms','0.534','The total can be larger\nthan synthetic alone, while\npointing in a better direction.',PURPLE,[('Real component',TEAL),('Synthetic component',RED),('Actual combined gradient',PURPLE)],'Norms do not add. Source proportions already included; gradient size does not establish benefit.')
    replace(s+'_legendhead','Measured gradient norms')

    s=new(26,'Source directions increasingly oppose',0,'Median cosine alignment among defined source pairs. Negative means opposition, zero perpendicular, positive alignment. 50/80 sources oppose in all790 updates. 99% sources oppose538/563 both-source updates;227 updates contain no real rows, so alignment undefined. Do not interpret a cosine as a held-out improvement or as complete gradient cancellation.',kind='plot',backup=True)
    chart(s,makechart('cosines',[1,2,3],[BLUE,AMBER,PURPLE],'Median real/synthetic gradient cosine',-1,.2))
    callouts(s,'Angle, not a loss share','cos(gR, gS)','Negative alignment\nmeans the vectors oppose\nin parameter space.',PURPLE,[('50% synthetic',BLUE),('80% synthetic',AMBER),('99% synthetic',PURPLE)],'Absent or near-zero source gradients have undefined alignment; no artificial zero angle.')
    replace(s+'_legendhead','Mixture legend')

    s=new(27,'Final batches expose an order effect',0,'The last original batch contains16 not128 examples. One real example there is6.25% of the batch, with eight times the per-example weight of a full batch. At cycles6/10 these tail updates make large corrections. In late99% cycles full batches still increase probe NLL on average; tail updates bring down the endpoints. This is evidence of sensitivity, not proof that all real benefit is confined to the tail. Whole-epoch real-only branches improve before the final batch.',backup=True)
    textbox(s,'big','1 real example in 16 = 6.25%',33,90,650,45,29,True,TEAL)
    table(s,'tail',[['99% historical path','Before final update','After final update'],['Cycle 6 · actual mixed','210.45','148.38'],['Cycle 10 · actual mixed','216.64','170.45']],33,175,654,122,[274,190,190])
    textbox(s,'why','A small batch amplifies each example’s weight.\nTest fixed-size tails and new row orders next.',33,316,650,58,21)
    footer(s,'Fixed 128-pair real probe. Per-update counterfactuals are local; they are not whole-run replacements.')

    s=new(28,'An average is not the combined trajectory',0,'Orange is the pointwise arithmetic mean of the separately evolved real-only and synthetic-only branch perplexities for cycle6. It is a visual reference only. Independent weights diverge after the first update; gradients are re-evaluated at different parameters; PPL is exponential in NLL. Therefore the average of endpoint PPLs is neither a gradient prediction, a model ensemble nor an interaction measure.',kind='plot',backup=True)
    chart(s,makechart('fork6',[1,2,3,4],[PURPLE,'#8899AA','#384957',AMBER],'128-pair probe perplexity',30,340,'Update within epoch',0,79,dashed={1:'MEDIUM_DASHED',2:'DOTTED'},points=0))
    callouts(s,'Cycle 6 · reference curve','176.34','Mean of the two branch PPLs.\nActual mixed endpoint:\n148.38.',AMBER,[('Both',PURPLE),('Real only / synthetic only',GRAY),('Arithmetic mean of branch PPLs',AMBER)],'The orange curve is descriptive, not a prediction of combined optimization.')
    replace(s+'_legendhead','Independent branch trajectories')

    s=new(29,'Decoding changes the training targets',0,'Separate OPT study with AdamW/cosine. Training-target generation decoder changes: five beams, greedy, pure multinomial sampling at temperature1 without truncation. End PPL real28.05 /beam342.01 /greedy390.99 /sampling31.88. Greedy and sampling ran4500; historical five-beam/real4000. Generation probes retained the same five-beam decoder in this study; do not confuse their metrics with the actual D10 training-target statistics. Sampling is a candidate control, not a guarantee against collapse.',backup=True)
    table(s,'decode',[['OPT training-target decoder','Cycle-10 test PPL'],['Real targets','28.05'],['Five deterministic beams','342.01'],['Greedy','390.99'],['Pure sampling','31.88']],33,104,654,210,[440,214])
    textbox(s,'point','The generator is part of the experimental protocol.',33,332,650,42,22,True,TEAL)
    footer(s,'Separate historical experiment: AdamW / cosine LR, with an explicitly recorded GPU difference.')

    s=new(30,'Pooled phrase diversity is a different metric',0,'Distinct-2/3 pool unique token ngrams divided by all within-output ngram occurrences across256 outputs. Denominators256*63 and256*62. A lower value indicates fewer unique phrases across the collection; not necessarily more repetition in each output. Pooled token entropy can rise while within-output repetition worsens because topic/vocabulary variation between outputs differs from repetition within one. These are greedy probe outputs, not model predictive entropy or factual accuracy.',backup=True)
    for j,(lab,c) in enumerate(zip(['0%','99%','100%'],[TEAL,PURPLE,RED])):textbox(s,f'leg{j}',lab+' synthetic',35+j*222,76,217,24,15,True,c)
    chart(s,makechart('distinct_2',[1,3,2],[TEAL,PURPLE,RED],'Pooled distinct-2 (%)',0,80),18,101,280,'d2')
    chart(s,makechart('distinct_3',[1,3,2],[TEAL,PURPLE,RED],'Pooled distinct-3 (%)',0,90),369,101,280,'d3')
    footer(s,'Same 256 prompts and 64-token outputs. No n-grams cross an output boundary.')

    s=new(31,'Whole-epoch forks: complete-test endpoints',0,'All values here use the complete fixed real test, not the128-pair diagnostic probe. Replays pass every1e-5 tolerance but are not bitwise identical. Full-test endpoint order agrees with probe trajectories. Original100real/9900synthetic strengths retained,79scheduled positions, exact saved synthetic targets, independent one-epoch branches.',backup=True)
    erows=[['Fork epoch','Both','Real only','Synthetic only']]
    for c in [3,6,10]:erows.append([str(c)]+[f'{float(next(r["test_ppl"] for r in endpoints if int(r["cycle"])==c and r["source"]==src)):.2f}' for src in ['both','real','synthetic']])
    table(s,'endpoints',erows,33,119,654,169,[160,164,165,165])
    textbox(s,'replay','Both-source replay verified before ablation.',33,318,650,40,23,True,PURPLE)
    footer(s,'Same 7,303 test pairs / 467,392 target tokens. The main fork graphs use a smaller probe.')

    s=new(32,'What we can claim—and what comes next',0,'References and boundaries. Shumailov2024 https://doi.org/10.1038/s41586-024-07566-y and https://arxiv.org/abs/2305.17493. Seddik2024 https://arxiv.org/abs/2404.05090 studies conditions under which real-data mixing limits collapse, not a universal empirical1% threshold. Drayson etal2025 https://arxiv.org/abs/2502.15654 examines decoding and detection. Villalobos2024 https://arxiv.org/abs/2211.04325 forecasts useful text supply under assumptions. All empirical numbers in this deck are our saved runs, not paper measurements.',backup=True)
    textbox(s,'limits','One seed, finite horizon, deterministic generation.\nHeld-out source documents can still overlap pretraining.\nLoss share, gradient geometry and benefit are distinct.',33,91,654,106,21,bullets=True)
    textbox(s,'refs','Read next',33,231,650,30,22,True,TEAL)
    textbox(s,'refsbody','Shumailov et al. (2024) · recursive collapse\nSeddik et al. (2024) · real/synthetic mixtures\nDrayson et al. (2025) · decoding and detection',33,273,650,85,19)
    footer(s,'Exact citations, protocol caveats and data paths are in the speaker notes. Full report and repo linked there.')

    assert len(slides)==32 and len(set(slides))==32
    assert len(trajectory)==55
    assert abs(get(trajectory,'mixed99',10,'test_perplexity')-164.4125178)<1e-4
    assert abs(float(loss[-1]['real_loss_share_percent'])-49.08766794)<1e-7
    assert all(len([r for r in forks if int(r['cycle'])==c and r['source']==src])==79 for c in [3,6,10] for src in ['both','real','synthetic'])
    manifest={'presentation_id':deck['presentationId'],'backing_spreadsheet_id':a.sheet_id,'source_revision':deck['revisionId'],'source_snapshot_sha256':hashlib.sha256(a.template_json.read_bytes()).hexdigest(),'input_hashes':sources,'slide_ids':slides,'main_slides':21,'skipped_backup_slides':11,'main_seconds':sum(x['seconds'] for x in outline if not x['backup']),'chart_ids':chartids,'chart_tab_id':sid,'charts_editable':True,'axis_policy':'Main fraction curves store explicitly labelled log10(PPL); raw PPL callouts. All source ablation panels share the same linear axes.','historical_evidence_modified':False}
    for name,obj in [('sheet-requests',C),('slide-requests',S),('notes',notes),('outline',outline),('data',blocks),('manifest',manifest),('geometry',geometry)]:
        (a.out/f'{name}.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'slides':len(slides),'charts':len(chartids),'slide_requests':len(S),'sheet_requests':len(C),'main_seconds':manifest['main_seconds']},ensure_ascii=True))


if __name__=='__main__':main()
