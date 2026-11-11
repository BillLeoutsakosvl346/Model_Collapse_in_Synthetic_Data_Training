"""Standalone offline figures from the frozen 77-row seven-fraction table."""
import argparse
import shutil
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

ROOT=next((p for p in Path(__file__).resolve().parents if (p/'.git').exists()),Path.cwd())
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
SOURCE=args.data.resolve();OUT=args.out.resolve()
if OUT==SOURCE or SOURCE.is_relative_to(OUT) or (OUT.exists() and any(OUT.iterdir())):
    raise SystemExit('Choose a new, separate, empty output directory; source evidence is protected.')
OUT.mkdir(parents=True,exist_ok=True)
for name in ['metrics.json','input_provenance.json','probe_audit.json']:
    shutil.copyfile(SOURCE/name,OUT/name)
rows=json.loads((OUT/'metrics.json').read_text())
fractions=[0.,.5,.8,.95,.99,.999,1.]
colors={0.:'#0072B2',.5:'#E69F00',.8:'#D55E00',.95:'#B32043',.99:'#222222',.999:'#007F86',1.:'#CC79A7'}
markers={0.:'o',.5:'^',.8:'D',.95:'X',.99:'*',.999:'H',1.:'o'}
series={f:sorted([r for r in rows if r['synthetic_fraction']==f],key=lambda r:r['cycle']) for f in fractions}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.titlesize':16,
    'axes.labelsize':13,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':10.5,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
footer='Fixed 256-prompt output probe; real-text scores use frozen held-out continuations.\nSeed 42 · one epoch per fresh chunk · shared cosine LR 5e-6 → 1e-6\n95/99/99.9%: RTX PRO 4500; 0/50/80/100%: RTX PRO 4000 (Blackwell)'
def save(fig,name):
    for ext in ('png','pdf'):fig.savefig(OUT/f'{name}.{ext}',dpi=200,facecolor='white')
    plt.close(fig)
def lines(ax,key,subset=fractions,scale=1):
    for f in subset:
        rs=series[f]
        ax.plot([r['cycle'] for r in rs],[scale*r[key] for r in rs],label=f'{f*100:.12g}% synthetic',
            color=colors[f],marker=markers[f],markersize=5,lw=2.5 if f==.999 else 1.8,
            linestyle='--' if f==.999 else '-')
    ax.set_xticks(range(11));ax.set_xlim(-.15,10.25)
def single(name,title,key,ylabel,subset=fractions,scale=1):
    fig,ax=plt.subplots(figsize=(8.6,8.6));fig.subplots_adjust(left=.15,right=.97,bottom=.19,top=.78)
    fig.suptitle(title,y=.965,fontweight='bold',fontsize=17)
    lines(ax,key,subset,scale);ax.set_xlabel('Training cycle (0 = pretrained model)');ax.set_ylabel(ylabel)
    handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.55,.915),ncol=2,frameon=False)
    fig.text(.5,.035,footer,ha='center',fontsize=9,color='#555555',linespacing=1.6)
    save(fig,name)
single('01_test_perplexity','Fixed real-test continuation perplexity','test_perplexity','Corpus perplexity (lower is better)')
single('02_test_perplexity_lower_fractions','Closer view: 0%, 50%, 80% and 95% synthetic','test_perplexity','Corpus perplexity (lower is better)',[0.,.5,.8,.95])
def family(name,title,panels):
    fig,axes=plt.subplots(2,1,figsize=(8.6,8.6));fig.subplots_adjust(left=.15,right=.97,bottom=.17,top=.78,hspace=.50)
    fig.suptitle(title,y=.965,fontweight='bold',fontsize=17)
    for ax,(key,ylabel,subtitle,scale) in zip(axes,panels):
        lines(ax,key,scale=scale);ax.set_ylabel(ylabel);ax.set_title(subtitle,fontsize=13,loc='left')
    axes[-1].set_xlabel('Training cycle (0 = pretrained model)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.55,.915),ncol=2,frameon=False)
    fig.text(.5,.035,footer,ha='center',fontsize=9,color='#555555',linespacing=1.6);save(fig,name)
family('03_entropy','Empirical token entropy: two different scopes',[
    ('token_entropy_nats','Nats','A  All 256 continuations pooled',1),
    ('mean_within_entropy_nats','Nats','B  Mean entropy within each 64-token output',1)])
family('04_distinct','Generated phrase diversity',[
    ('distinct_2','Distinct-2 (%)','A  Unique token pairs / all token pairs',100),
    ('distinct_3','Distinct-3 (%)','B  Unique token triplets / all token triplets',100)])
single('05_within_repetition','Repeated four-token phrases within outputs','mean_within_repeated_4gram_rate','Mean repeated 4-gram fraction (%)',scale=100)

corpora=['fineweb_edu','wikitext103','c4_en','pg19','ag_news']
baseline=series[0.][0]
endpoints={f:series[f][-1] for f in fractions}
ratios=np.asarray([[endpoints[f]['secondary_'+c+'_perplexity']/baseline['secondary_'+c+'_perplexity'] for c in corpora] for f in fractions])
fig,ax=plt.subplots(figsize=(8.6,8.6));fig.subplots_adjust(left=.16,right=.90,bottom=.29,top=.83)
fig.suptitle('Secondary held-out corpora at cycle 10',y=.96,fontweight='bold',fontsize=17)
im=ax.imshow(np.log2(ratios),cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-.10,vcenter=0,vmax=math.log2(ratios.max())),aspect='auto')
ax.grid(False);ax.set_xticks(range(5),['FineWeb','WikiText-103','C4','PG-19','AG News'],rotation=30,ha='right')
ax.set_yticks(range(7),[f'{f*100:.12g}%' for f in fractions]);ax.set_ylabel('Synthetic training-target fraction')
for i,f in enumerate(fractions):
    for j,c in enumerate(corpora):
        ax.text(j,i,f'{ratios[i,j]:.2f}×\nPPL {endpoints[f]["secondary_"+c+"_perplexity"]:.1f}',ha='center',va='center',fontsize=10,
            color='white' if ratios[i,j]>3 else 'black')
bar=fig.colorbar(im,ax=ax,fraction=.045,pad=.035);bar.set_label('log₂(final perplexity / initial perplexity)',fontsize=10)
fig.text(.5,.045,'Each cell: ratio to the same corpus’s pretrained baseline, then absolute PPL.\n512-token full-text evaluation; separate from the 129-position completion task.\n'+footer,ha='center',fontsize=8.7,color='#555555',linespacing=1.5)
save(fig,'06_secondary_corpora')

fig,ax=plt.subplots(figsize=(8.6,8.6));fig.subplots_adjust(left=.15,right=.97,bottom=.21,top=.87)
steps=np.arange(791);lr=1e-6+(5e-6-1e-6)*.5*(1+np.cos(np.pi*steps/790))
ax.plot(steps/79,lr/1e-6,color='#374151',lw=2.5,label='Shared global cosine schedule')
ax.scatter(range(11),[r['learning_rate_end']/1e-6 for r in series[0.]],color='#007F86',zorder=3,label='Recorded cycle boundaries')
ax.set_xticks(range(11));ax.set_xlabel('Training cycle (79 optimizer updates per cycle)');ax.set_ylabel('Learning rate (×10⁻⁶)');ax.legend(loc='upper right')
fig.suptitle('Learning rate changes throughout the experiment',y=.95,fontweight='bold',fontsize=17)
fig.text(.5,.07,'AdamW moments and the schedule carry across cycles; they do not restart.\nAll seven conditions share this schedule: 5e-6 initially, 3e-6 at cycle 5, 1e-6 finally.\nA ten-cycle plateau under this schedule is not evidence of permanent stability.',ha='center',fontsize=10,color='#555555',linespacing=1.6)
save(fig,'07_learning_rate')

manifest={'created_with':'plot.py','analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'python':platform.python_version(),
    'matplotlib':matplotlib.__version__,'numpy':np.__version__,
    'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    'metric_rows':77,'fractions':fractions,'input_hashes':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['metrics.json','input_provenance.json','probe_audit.json']},
    'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*')) if p.suffix in ('.png','.pdf')}}
(OUT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'figures':len(manifest['outputs'])//2,'out':str(OUT),'endpoints':list(endpoints)}))
