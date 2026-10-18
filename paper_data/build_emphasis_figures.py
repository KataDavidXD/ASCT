from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
from plot_style import paper_font
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
HERE=Path(__file__).resolve().parent; OUT=HERE/'figures'
SEEDS=(11,23,37); TREES=('Uniform','UCT','AgentUCT'); METHODS=('PPO','VinePPO',*TREES)
NAMES={m:'ASCT-'+m if m in TREES else m for m in ('Base',*METHODS)}
COLORS={'Base':'#999999','PPO':'#555555','VinePPO':'#9B6A1C','Uniform':'#737A35','UCT':'#936490','AgentUCT':'#16577C'}
def read(name): return list(csv.DictReader((HERE/'data'/name).open(encoding='utf-8-sig')))
selected=[r for r in read('main_endpoints.csv') if r['setting']=='matched' and r['split']=='test']
u={m:np.array([float(r['utility_official']) for r in selected if r['method']==m]) for m in METHODS}
base=float(next(r['utility_official'] for r in selected if r['method']=='Base'))
plt.rcParams.update({'font.family':'serif','font.serif':[paper_font()],'font.size':9,'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none'})
def clean(ax):
    ax.spines[['top','right']].set_visible(False); ax.set_axisbelow(True)
def save(fig,name):
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text(): continue
        b=t.get_window_extent(renderer)
        assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1,(name,t.get_text(),b)
    for ext in ('pdf','png'): fig.savefig(OUT/f'{name}.{ext}',dpi=200)
    plt.close(fig)
fig, ax = plt.subplots(figsize=(5.5, 2.24))
fig.subplots_adjust(left=0.255, right=0.887, top=0.91, bottom=0.22)
ordered = ("Base", *METHODS)
for y, m in enumerate(ordered):
    values = np.array([base]) if m == "Base" else u[m]
    avg = values.mean()
    sd = values.std(ddof=1) if len(values) > 1 else 0.0
    # Full-height dashed mean guides; muted so data and SD intervals stay primary.
    ax.axvline(avg, color=COLORS[m], linestyle=(0, (4, 3)),
               linewidth=1.0 if m == "AgentUCT" else 0.75,
               alpha=0.55 if m == "AgentUCT" else 0.40, zorder=2)
    if len(values) > 1:
        ax.scatter(values, y + np.array([-0.10, 0, 0.10]), marker="o", s=12,
                   facecolor="white", edgecolor=COLORS[m], linewidth=0.8, zorder=4)
        ax.errorbar(avg, y, xerr=sd, fmt="none", ecolor=COLORS[m],
                    elinewidth=1.7 if m == "AgentUCT" else 1.1,
                    capsize=3, capthick=1.1, zorder=3)
    ax.scatter([avg], [y], marker="D", s=31 if m == "AgentUCT" else 22,
               facecolor=COLORS[m], edgecolor="white", linewidth=0.55, zorder=5)
    ax.text(1.055, y, f"{avg:.4f}", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=8.8,
            fontweight="bold" if m in TREES else "normal", color=COLORS[m])
ax.text(1.055, -0.84, "Mean", transform=ax.get_yaxis_transform(),
        ha="left", va="center", fontsize=8.8, color="#333333")
ax.set_yticks(range(len(ordered)), [NAMES[m] for m in ordered])
for label in ax.get_yticklabels()[3:]: label.set_fontweight("bold")
ax.set_ylim(5.6, -0.6)
ax.set_xlim(0.54, 0.65)
ax.set_xticks(np.arange(0.54, 0.651, 0.02))
ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
ax.set_xlabel(r"Mean test workflow utility, $\overline{U}_{\mathrm{test}}$", labelpad=4)
ax.tick_params(axis="y", length=0, pad=7)
clean(ax)
ax.grid(False, axis="x")
save(fig, "utility_errorbars")


METHODS=('Base','PPO','VinePPO',*TREES); LEVELS=('easy','medium','hard')
COUNTS={'easy':76,'medium':256,'hard':68}
MARKERS={'Base':'x','PPO':'o','VinePPO':'^','Uniform':'s','UCT':'v','AgentUCT':'D'}
summary=read('difficulty_summary.csv'); seed_rows=read('difficulty_by_seed.csv')
for r in summary:
    r['mean']=float(r['mean']); r['sd']=float(r['sd']) if r['sd'] else ''
for r in seed_rows: r['utility']=float(r['utility'])
fig,axes=plt.subplots(1,3,figsize=(6.4,2.6),sharey=True)
for ax,level in zip(axes,LEVELS):
    for x,m in enumerate(METHODS):
        s=next(r for r in summary if r['method']==m and r['level']==level)
        v=[r['utility'] for r in seed_rows if r['method']==m and r['level']==level]
        if m!='Base':ax.scatter(x+np.array([-.13,0,.13]),v,s=10,facecolors='none',edgecolors=COLORS[m],alpha=.55,lw=.75,zorder=3)
        ax.errorbar(x,s['mean'],yerr=None if s['sd']=='' else s['sd'],fmt=MARKERS[m],
            color=COLORS[m],ms=4.3,capsize=3,elinewidth=1.15,zorder=4)
    ax.set_title(f"{level.capitalize()} ($n={COUNTS[level]}$)",fontsize=9)
    ax.set_xticks(range(6),[NAMES[m] for m in METHODS],rotation=48,ha='right',fontsize=8)
    for label in ax.get_xticklabels()[3:]: label.set_fontweight('bold')
    ax.set_xlim(-.45,5.5);ax.set_ylim(.43,.71);ax.set_yticks([.45,.50,.55,.60,.65,.70])
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0].set_ylabel('Mean test workflow utility',fontsize=9)
fig.subplots_adjust(left=.10,right=.985,bottom=.37,top=.86,wspace=.12)
save(fig,'test_utility_by_difficulty')