"""Full-training evaluation and auxiliary-token composition, from seed ledgers."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'figures'
OUT.mkdir(exist_ok=True)
with (HERE/'data/search_costs.csv').open() as f:
    rows=list(csv.DictReader(f))
SEEDS=(11,23,37)
TREES=('Uniform','UCT','AgentUCT')
ALL=('PPO','VinePPO',*TREES)
NAMES={m:('ASCT-'+m if m in TREES else m) for m in ALL}
COLORS={'PPO':'#555555','VinePPO':'#9B6A1C','Uniform':'#737A35','UCT':'#936490','AgentUCT':'#16577C'}
lookup={(r['method'],int(r['seed'])):r for r in rows}
def vals(m,key):return np.array([float(lookup[m,s][key]) for s in SEEDS])
ENV=('generator_4b_input_tokens','generator_4b_output_tokens','query_embedding_06b_input_tokens','reranker_06b_input_tokens')
for m in ALL:
    assert np.array_equal(sum(vals(m,k) for k in ENV),vals(m,'qwen_tokens'))
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':9,
                     'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none'})
def style(ax,labels=True,methods=TREES):
    ax.set_yticks(range(len(methods)),[NAMES[m] for m in methods] if labels else [])
    if labels:
        for label in ax.get_yticklabels()[1:]: label.set_fontweight('bold')
    ax.set_ylim(len(methods)-.5,-.5)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x',color='#e4e4e4',lw=.5)
    ax.set_axisbelow(True)
def save(fig,name):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text():continue
        b=t.get_window_extent(renderer)
        assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1,(name,t.get_text(),b)
    for ext in ('pdf','svg','png'):fig.savefig(OUT/f'{name}.{ext}',dpi=300)
    plt.close(fig)

fig=plt.figure(figsize=(5.5,6.4))
ax=fig.add_axes([.30,.635,.66,.25])
stack_colors=('#7893A7','#CCA768')
totalrows=[]
for y,m in enumerate(ALL):
    env=vals(m,'qwen_tokens')/1e6
    actor=vals(m,'auxiliary_actor_scoring_tokens')/1e6
    total=env+actor
    ax.barh(y,env.mean(),color=stack_colors[0],height=.52,label='Auxiliary environment' if y==0 else None)
    ax.barh(y,actor.mean(),left=env.mean(),color=stack_colors[1],height=.52,hatch='///',edgecolor='white',linewidth=.3,label='Auxiliary actor scoring' if y==0 else None)
    ax.errorbar(total.mean(),y,xerr=total.std(ddof=1),fmt='none',capsize=3,color='#444',clip_on=False)
    label='0 (no auxiliary evaluation)' if m=='PPO' else f'{total.mean():.2f}'
    ax.text(total.mean()+total.std(ddof=1)+5,y,label,va='center',fontsize=8.5,fontweight='bold' if m=='AgentUCT' else 'normal')
    if m=='VinePPO':
        ax.text(env.mean()/2,y,f'{env.mean():.2f}',ha='center',va='center',fontsize=8,color='white')
        ax.text(env.mean()+actor.mean()/2,y,f'{actor.mean():.2f}',ha='center',va='center',fontsize=8,bbox=dict(facecolor=stack_colors[1],edgecolor='none',pad=1.5))
    totalrows.append({'method':m,'environment_M':float(env.mean()),'auxiliary_actor_M':float(actor.mean()),'total_M':float(total.mean()),'total_sd_M':float(total.std(ddof=1))})
ax.set_yticks(range(5),[NAMES[m] for m in ALL]);[label.set_fontweight('bold') for label in ax.get_yticklabels()[2:]]
ax.set_ylim(4.6,-.6);ax.set_xlim(0,445);ax.set_xticks([0,100,200,300,400]);ax.set_xlabel('Total auxiliary token count (M)')
ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
fig.text(.05,.970,'(a) Auxiliary total and its two components',fontsize=10)
fig.legend(loc='upper center',bbox_to_anchor=(.63,.947),ncol=1,frameon=False,fontsize=8.5)
titles=['(b) Generator input (4B)','(c) Generator output (4B)',
        '(d) Query embedding (0.6B)','(e) Reranker input (0.6B)']
for idx,(key,title) in enumerate(zip(ENV,titles)):
    x=.30 if idx%2==0 else .70
    y=.345 if idx<2 else .080
    a=fig.add_axes([x,y,.26,.17])
    for j,m in enumerate(('VinePPO',*TREES)):
        v=vals(m,key)/1e6
        a.errorbar(v.mean(),j,xerr=v.std(ddof=1),fmt='D',ms=3.5,capsize=2,color=COLORS[m])
    style(a,labels=idx%2==0,methods=('VinePPO',*TREES))
    limits=[200,8,1,75]
    ticks=[[0,100,200],[0,4,8],[0,.5,1],[0,25,50,75]]
    a.set_xlim(0,limits[idx]);a.set_xticks(ticks[idx])
    a.set_xlabel('Environment tokens (M)',fontsize=8)
    a.set_title(title,loc='left',fontsize=8.5,pad=7)
    a.tick_params(labelsize=8)
save(fig,'auxiliary_token_composition')
(OUT/'auxiliary_total_data.json').write_text(json.dumps(totalrows,indent=2))
print(json.dumps(totalrows,indent=2))
