"""Offline square-panel scientific figures; preserves all original publications."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS={'synthetic_100':'#d66a32','mixed_50':'#178b87','mixed_20':'#3763a5','continued':'#178b87','reset':'#7755a0','s0':'#626b73'}
LABELS={'synthetic_100':'100% synthetic','mixed_50':'50% synthetic','mixed_20':'20% synthetic','continued':'50/50 continued','reset':'50/50 reset'}


def square_grid(n,legend=None):
    plt.rcParams.update({'font.size':11,'axes.titlesize':11.5,'axes.labelsize':10.5,'legend.fontsize':9,'font.family':'DejaVu Sans'})
    fig,axs=plt.subplots(n//2,2,figsize=(8,4*(n//2)+.45),squeeze=False,layout='constrained')
    for ax in axs.flat:
        ax.set_box_aspect(1)
        ax.grid(alpha=.2)
    return fig,list(axs.flat)


def trajectory(ax,by,key,title,scale=1,shared=False):
    for arm,rows in by.items():
        start=1 if shared else 0
        ax.plot(range(start,11),[r[key]*scale for r in rows[start:]],'o-',ms=3,lw=1.7,color=COLORS[arm],label=LABELS[arm])
    common=next(iter(by.values()))
    if shared:ax.plot([0,1],[r[key]*scale for r in common[:2]],'o-',color=COLORS['s0'],label='Shared S0/S1',zorder=6)
    else:ax.scatter([0],[common[0][key]*scale],color=COLORS['s0'],marker='s',zorder=6,label='Shared S0')
    ax.set(title=title,xlabel='Synthetic-training cycle',xticks=[0,2,4,6,8,10])


def save(fig,axes,path):
    handles,labels=axes[0].get_legend_handles_labels()
    if handles:fig.legend(handles,labels,loc='outside upper center',ncol=3,fontsize=9)
    fig.savefig(path,dpi=220)
    plt.close(fig)


def make_figures(out,by,histories,seqs,within,shared=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    specs=[('test_perplexity','A  Real-test perplexity',1),('validation_perplexity','B  Real-validation perplexity',1),
           ('distinct_2','C  Pooled distinct-2 (%)',100),('distinct_3','D  Pooled distinct-3 (%)',100)]
    fig,axes=square_grid(4)
    for ax,(key,title,scale) in zip(axes,specs):trajectory(ax,by,key,title,scale,shared)
    save(fig,axes,out/'prediction_and_phrases.png')
    fig,axes=square_grid(2)
    for ax,(key,title,scale) in zip(axes,[('token_entropy_nats','E  Pooled token entropy (nats)',1),('mean_within_repeated_4gram_rate','F  Within-output repetition (%)',100)]):trajectory(ax,by,key,title,scale,shared)
    save(fig,axes,out/'entropy_and_repetition.png')
    fig,axes=square_grid(4)
    for arm in by:
        z=np.sort(seqs[arm][10]);axes[0].plot(z,np.arange(1,len(z)+1)/len(z),color=COLORS[arm],label=LABELS[arm])
    z=np.sort(next(iter(seqs.values()))[0]);axes[0].plot(z,np.arange(1,len(z)+1)/len(z),color=COLORS['s0'],ls='--',label='Shared S0')
    axes[0].set(title='A  Final real-test block loss',xlabel='Mean block NLL',ylabel='Cumulative fraction')
    arms=list(by);target=arms[-1]
    for arm in arms[:-1]:
        gap=seqs[arm][10]-seqs[target][10]
        axes[1].hist(gap,bins=40,histtype='step',lw=1.5,color=COLORS[arm],label=LABELS[arm])
    axes[1].axvline(0,color='#555',ls='--');axes[1].set(title='B  Paired final test-loss gaps',xlabel='Other arm − '+LABELS[target]+' NLL',ylabel='Test blocks');axes[1].legend(fontsize=8)
    for ax,key,title in [(axes[2],'validation_loss','C  Cycle 10: validation loss'),(axes[3],'train_loss','D  Cycle 10: training loss')]:
        for arm in by:
            h=[r for r in histories if r['arm']==arm and int(r['cycle'])==10]
            ax.plot([int(r['epoch']) for r in h],[float(r[key]) for r in h],'o-',color=COLORS[arm],label=LABELS[arm])
        ax.axvline(5,ls=':',color='#777');ax.set(title=title,xlabel='Epoch (5 retained)',ylabel='NLL',xticks=range(1,6))
    save(fig,axes,out/'real_text_and_training.png')
    fig,axes=square_grid(4)
    trajectory(axes[0],by,'repeated_4gram_rate','A  Pooled repeated 4-grams (%)',100,shared)
    trajectory(axes[1],by,'mean_within_entropy_nats','B  Within-output entropy (nats)',1,shared)
    for j,arm in enumerate(by):
        w=.75/len(by)
        axes[2].bar(np.arange(2)+(j-(len(by)-1)/2)*w,[100*by[arm][-1][key] for key in ('newline_token_50118_share','standalone_space_share')],width=w,color=COLORS[arm],label=LABELS[arm])
        z=np.sort(within[arm][10]);axes[3].plot(z*100,np.arange(1,len(z)+1)/len(z),color=COLORS[arm],label=LABELS[arm])
    axes[2].set_xticks([0,1],['Newline','Standalone\nspace']);axes[2].set(title='C  Final formatting tokens',ylabel='Share of probe tokens (%)')
    axes[3].set(title='D  Final within-output repetition',xlabel='Repeated 4-grams (%)',ylabel='Cumulative fraction')
    save(fig,axes,out/'repetition_and_formatting.png')
    fig,axes=square_grid(2)
    for ax,key,title in [(axes[0],'frozen_s0_nll_mean','E  Frozen-S0 continuation NLL: mean'),(axes[1],'frozen_s0_nll_p95','F  Frozen-S0 continuation NLL: P95')]:trajectory(ax,by,key,title,1,shared)
    save(fig,axes,out/'frozen_s0_scores.png')
    return {p.stem:str(p.resolve()) for p in sorted(out.glob('*.png'))}
