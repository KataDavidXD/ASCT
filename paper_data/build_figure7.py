"""ASCT mechanism diagram, based on v26 Sections 4, A.2 and A.3.

The numerical example is illustrative, inherited from the original Figure 7.
Run with Python + matplotlib. Exports PDF, SVG and PNG alongside this file.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

OUT = Path(__file__).resolve().parent/'figures'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix', 'font.size': 9, 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'svg.fonttype': 'none'})
INK, BLUE, GOLD = '#26313B', '#16577C', '#95631D'
GRAY, LIGHT, PALE = '#65717A', '#D5DFE5', '#EDF4F8'
W, H = 5.5, 4.65
fig = plt.figure(figsize=(W, H), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0,W), ylim=(0,H)); ax.axis('off')

def txt(x,y,s,size=9,align='center',color=INK,bold=False,**kwargs):
    return ax.text(x,y,s,ha=align,va='center',fontsize=size,color=color,
        fontweight='bold' if bold else 'normal',zorder=8,**kwargs)

def line(points,color=LIGHT,dash=False,lw=.8):
    ax.plot(*zip(*points),color=color,lw=lw,ls=(0,(3,2)) if dash else '-',zorder=1)

def arrow(a,b,color=INK,dash=False,lw=.9):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=7,
        linewidth=lw,color=color,linestyle=(0,(3,2)) if dash else '-',
        shrinkA=1.5,shrinkB=2,zorder=3))

def box(x,y,w,h,fill='white',edge=LIGHT):
    ax.add_patch(FancyBboxPatch((x,y),w,h,
        boxstyle='round,pad=0,rounding_size=0.035',fc=fill,ec=edge,lw=.8,zorder=0))

def node(x,y,r=.026,fill='white',edge=BLUE):
    ax.add_patch(Circle((x,y),r,fc=fill,ec=edge,lw=.8,zorder=4))

# ASCT: actual behavior and optimizer records are separated from auxiliary work.
txt(.10,4.47,'(a) ASCT: multi-step search supplies local actor credit',10.2,'left',BLUE,True)
box(.10,3.83,1.28,.43,fill=PALE)
txt(.74,4.115,'Frozen actor at '+r'$s$',9.3,bold=True)
txt(.74,3.93,r'$p_a=\pi_{\mathrm{old}}(a\mid s)$',9.6)
arrow((1.38,4.04),(1.77,4.04))
box(1.79,3.83,1.42,.43)
txt(2.50,4.115,r'Sample $a_t\sim p$',9.3,bold=True)
txt(2.50,3.93,r'Execute $a_t$; advance',9.1)
arrow((3.21,4.04),(4.22,4.04))
txt(3.71,4.20,r'$(s,a_t,p_{a_t})$',9.7)
box(4.24,3.83,1.16,.43,fill=PALE,edge=BLUE)
txt(4.82,4.04,'PPO update',9.7,bold=True,color=BLUE)

# State restoration and frozen probabilities have distinct destinations.
arrow((.52,3.83),(.52,3.38),BLUE,True)
txt(.94,3.59,'Restore '+r'$s$',8.8,color=BLUE)
line([(1.13,3.83),(1.13,3.70),(3.03,3.70),(3.03,3.38)],color=GOLD,dash=True)
arrow((3.03,3.49),(3.03,3.37),GOLD,True)
txt(2.18,3.71,r'Frozen probabilities $p$',8.8,color=GOLD,
    bbox={'facecolor':'white','edgecolor':'none','pad':1.2})
arrow((4.82,3.39),(4.82,3.83),BLUE,True)
txt(5.18,3.54,'Credit',8.8,color=BLUE)

txt(.12,3.23,'1  Execute alternatives',9.2,'left',BLUE,True)
txt(3.01,3.23,'2  Action values',9.2,color=BLUE,bold=True)
txt(4.77,3.23,'3  Center credit',9.2,color=BLUE,bold=True)

# A real tree topology with all legal roots and multiple downstream decisions.
root=(.27,2.49); node(*root,r=.065,fill=PALE)
txt(*root,r'$s$',9.5,color=BLUE)
for i,y in enumerate([2.88,2.49,2.10]):
    first=(.86,y); nxt=(1.34,y)
    arrow(root,first,BLUE,True); node(*first)
    txt(.63,y+.105,rf'$a_{i+1}$',9.2,color=BLUE)
    for dy in [-.105,.105]:
        mid=(1.38,y+dy); leaf=(1.98,y+dy)
        arrow(first,mid,BLUE,True); node(*mid)
        arrow(mid,leaf,BLUE,True); node(*leaf,r=.023,fill=GOLD,edge=GOLD)
txt(1.98,3.07,r'$U(w)$',9.0,color=GOLD)
txt(1.14,1.84,'Evidence-dependent choices',8.8,color=GRAY)
txt(1.14,1.66,'Stop / Continue; rerank',8.8,color=GRAY)
arrow((2.08,2.49),(2.37,2.49),BLUE,True)

# Illustrative table: values acquired by search, probabilities from the actor.
cols=[2.57,3.03,3.49]
for x,label,c in zip(cols,[r'$a$',r'$p_a$',r'$\widehat Q(a)$'],[INK,GOLD,BLUE]):
    txt(x,2.96,label,9.5,color=c)
line([(2.39,2.84),(3.70,2.84)],GRAY)
ax.add_patch(Rectangle((2.39,2.37),1.31,.25,fc='#DDEBF4',ec='none',zorder=0))
for i,(p,q,y) in enumerate([(.20,.40,2.72),(.30,.60,2.49),(.50,.90,2.26)]):
    for x,s in zip(cols,[rf'$a_{i+1}$',f'{p:.2f}',f'{q:.2f}']):
        txt(x,y,s,9.5,bold=i==1)
txt(3.04,2.03,r'Example: $a_t=a_2$',8.9,color=BLUE)
txt(3.04,1.76,r'$\widehat Q(a)=\frac{1}{|I_a|}\sum_{i\in I_a}U(w_i)$',9.7,color=BLUE)
arrow((3.72,2.49),(3.98,2.49),BLUE,True)

# A positive absolute outcome can have negative credit relative to the actor.
txt(4.77,2.94,'Actor reference',9,color=GOLD)
txt(4.77,2.72,r'$\widehat V=\sum_a p_a\widehat Q(a)$',10.1)
txt(4.77,2.48,r'$=0.71$',10.1)
box(4.03,1.93,1.39,.36,fill=PALE,edge=BLUE)
txt(4.725,2.11,r'$\widehat A(a_t)=\widehat Q(a_t)-\widehat V$',9.7,color=BLUE)
txt(4.77,1.76,r'$0.60-0.71=-0.11$',9.7,color=BLUE)

line([(.10,1.48),(5.40,1.48)])
txt(.12,1.31,'Update actor-sampled decisions; use branches only to estimate credit.',9.1,'left',bold=True)
txt(.12,1.10,'Next iteration: refresh actor states and credit.   Deployment: actor only.',9.0,'left')

# Small comparator uses the same update interface, different estimation unit.
line([(.10,.92),(5.40,.92)],color=GRAY)
txt(.12,.74,'(b) VinePPO: credit from adjacent MC state values',9.4,'left',bold=True)
node(.30,.39,.048,edge=GRAY); txt(.30,.39,r'$s$',9.3)
node(1.22,.39,.048,edge=GRAY); txt(1.22,.39,r"$s'$",9.3)
arrow((.36,.39),(1.15,.39))
txt(.75,.56,r'$a_t$',9.2)
for cx in [.30,1.22]:
    for dx in [-.16,0,.16]:
        arrow((cx,.33),(cx+dx,.13),GRAY,True,lw=.65)
txt(2.44,.45,r"$\widehat A_t=r_t+\widehat V(s')-\widehat V(s)$",10.0)
txt(2.44,.20,'Frozen-actor continuations',8.9,color=GRAY)
arrow((3.80,.39),(4.21,.39))
box(4.24,.19,1.16,.41)
txt(4.82,.395,'PPO update',9.2)

# Verify the numerical illustration and text geometry before exporting.
assert abs(sum(p*q for p,q in [(0.2,0.4),(0.3,0.6),(0.5,0.9)])-.71)<1e-12
fig.canvas.draw(); renderer=fig.canvas.get_renderer()
rects=[]
for artist in ax.texts:
    b=artist.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    assert b.x0>=0 and b.x1<=1 and b.y0>=0 and b.y1<=1,(artist.get_text(),b.bounds)
    rects.append((artist.get_text(),b))
overlaps=[]
for i,(a,ba) in enumerate(rects):
    for b,bb in rects[i+1:]:
        if ba.overlaps(bb): overlaps.append([a,b])
assert not overlaps,overlaps
for ext in ['pdf','svg','png']:
    fig.savefig(OUT/f'figure7_asct_mechanism.{ext}',dpi=360,facecolor='white')
(OUT/'figure7_audit.json').write_text(json.dumps({
    'source':'ASCT_ICLR2027_v26_submission.pdf, Sections 4, A.2-A.3 and original Figure 7',
    'size_inches':[W,H], 'text_bounds_pass':True, 'text_overlap_pairs':overlaps,
    'illustration_baseline':.71,'illustration_sampled_credit':-.11,
    'toy_values_are_not_empirical_results':True,
    'displayed_core_equations':['root-action mean utility','actor-weighted baseline','sampled-action centered credit'],
    'minimum_font_pt':min(a.get_fontsize() for a in ax.texts)
},indent=2),encoding='utf-8')
print(OUT/'figure7_asct_mechanism.pdf')
