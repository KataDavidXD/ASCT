"""Native paper-width alternatives, identical retained observations and projections."""
from pathlib import Path
import csv, json, hashlib
import numpy as np
import matplotlib
from plot_style import paper_font
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, FormatStrFormatter

HERE=Path(__file__).resolve().parent
DATA=HERE/'data'
OUT=HERE/'figures'
OUT.mkdir(exist_ok=True)
read=lambda name:list(csv.DictReader((DATA/name).open(encoding='utf-8-sig',newline='')))
curves=read('training_curve_means.csv')
cost=read('cost_by_seed.csv')
methods=['PPO','VinePPO','Uniform','UCT','AgentUCT']
names={m:('ASCT-'+m if m in methods[2:] else m) for m in methods}
colors=dict(zip(methods,['#555555','#9B6A1C','#737A35','#936490','#16577C']))
styles=['--',(0,(5,2,1,2)),'--','-.','-']
markers=['s','^','o','s','D']
plt.rcParams.update({'font.family':'serif','font.serif':[paper_font()],'font.size':10,
    'axes.labelsize':10,'axes.titlesize':10,'xtick.labelsize':10,'ytick.labelsize':10,
    'legend.fontsize':10,'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,
    'svg.fonttype':'none','axes.linewidth':.65})
N=np.geomspace(100000,3000000,501)
projections={}
for m in methods:
    rows=sorted([r for r in cost if r['method']==m],key=lambda r:int(r['seed']))
    assert [int(r['seed']) for r in rows]==[11,23,37]
    projections[m]=np.array([float(r['test_utility'])-1e-4*float(r['auxiliary_total_tokens'])/N for r in rows])
a=[r for r in cost if r['method']=='AgentUCT']
p=[r for r in cost if r['method']=='PPO']
cross=1e-4*np.mean([float(r['auxiliary_total_tokens']) for r in a])/(np.mean([float(r['test_utility']) for r in a])-np.mean([float(r['test_utility']) for r in p]))
assert abs(cross-360358)<1

def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#eeeeee',lw=.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=3,pad=2)

def training(ax,metric,title,ylabel,ylim):
    for m in methods[2:]:
        j=methods.index(m)
        rows=sorted([r for r in curves if r['method']==m],key=lambda r:int(r['iteration']))
        assert len(rows)==3
        y=np.array([float(r[metric+'_mean']) for r in rows])
        sd=np.array([float(r[metric+'_sd']) for r in rows])
        ax.plot([1,2,3],y,color=colors[m],ls=styles[j],marker=markers[j],ms=3.5,lw=1.3)
        ax.fill_between([1,2,3],y-sd,y+sd,color=colors[m],alpha=.13,lw=0)
        ax.errorbar([1,2,3],y,yerr=sd,fmt='none',color=colors[m],capsize=2.5,lw=.8)
    ax.set(title=title,ylim=ylim,xticks=[1,2,3],ylabel=ylabel)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    clean(ax)

def amortized(ax):
    for j,m in enumerate(methods):
        mean=projections[m].mean(0);sd=projections[m].std(0,ddof=1)
        ax.fill_between(N,mean-sd,mean+sd,color=colors[m],alpha=.085,lw=0)
        ax.plot(N,mean,color=colors[m],ls=styles[j],marker=markers[j],markevery=[0,135,237,339,500],ms=3,lw=1.5 if m=='AgentUCT' else 1.05)
    ax.set(xscale='log',xlim=(1e5,3e6),ylim=(.18,.645),ylabel=r'Mean $J_{\mathrm{deploy}}$')
    ax.set_xticks([1e5,5e5,3e6],['100k','500k','3M'])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([.2,.4,.6])
    ax.plot([cross,cross],[.18,projections['PPO'].mean()],color='#666666',ls=(0,(4,2)),lw=.7)
    ax.text(cross*1.1,.235,'360k',fontsize=10,color='#444444')
    ax.set_xlabel(r'Policy uses $N$ (log)')
    clean(ax)

def legend(fig):
    handles=[Line2D([],[],color=colors[m],ls=styles[j],marker=markers[j],ms=3.5,lw=1.3) for j,m in enumerate(methods)]
    # Three ASCT entries in the top row; two controls in the second row.
    order=[2,0,3,1,4]
    leg=fig.legend([handles[i] for i in order],[names[methods[i]] for i in order],loc='upper center',bbox_to_anchor=(.52,1),ncol=3,frameon=False,handlelength=1.6,columnspacing=.9,handletextpad=.4,labelspacing=.35)
    for t in leg.get_texts():
        if t.get_text().startswith('ASCT-'):t.set_fontweight('bold')


namespace=globals()
out=HERE/'figures'
width=5.5*.485
for name in ['training_half','amortized_half']:
    fig=plt.figure(figsize=(width,2.85))
    if name=='training_half':
        ax1=fig.add_axes([.20,.51,.77,.235]);ax2=fig.add_axes([.20,.155,.77,.235])
        namespace['training'](ax1,'rollout_u',r'(a) Mean rollout $U_{\mathrm{train}}$','',(.51,.58))
        namespace['training'](ax2,'j_search',r'(b) Mean $J_{\mathrm{search}}$','',(.477,.543))
        ax1.set_yticks([.52,.55,.58]);ax2.set_yticks([.48,.51,.54])
        ax2.set_xlabel('Training iteration')
        legend_methods=methods[2:];ncol=1
    else:
        ax=fig.add_axes([.205,.155,.75,.59])
        namespace['amortized'](ax);ax.set_ylabel('');ax.set_title(r'Mean $J_{\mathrm{deploy}}$')
        ax.set_yticks([.2,.3,.4,.5,.6])
        legend_methods=methods;ncol=2
    handles=[]
    for m in legend_methods:
        j=methods.index(m)
        handles.append(Line2D([],[],color=namespace['colors'][m],ls=namespace['styles'][j],marker=namespace['markers'][j],ms=3.5,lw=1.3))
    legend=fig.legend(handles,[namespace['names'][m] for m in legend_methods],loc='upper center',bbox_to_anchor=(.50,1.005),ncol=ncol,frameon=False,handlelength=1.2,columnspacing=.5,handletextpad=.2,labelspacing=.15)
    for t in legend.get_texts():
        if t.get_text().startswith('ASCT-'):t.set_fontweight('bold')
    fig.canvas.draw()
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            bb=text.get_window_extent(fig.canvas.get_renderer())
            assert bb.x0>=-1 and bb.y0>=-1 and bb.x1<=fig.bbox.width+1 and bb.y1<=fig.bbox.height+1,(name,text.get_text(),bb.bounds)
    for ext in ['pdf','png','svg']:fig.savefig(out/f'{name}.{ext}',dpi=240)
    plt.close(fig)

audit=json.loads((HERE/'combined_figure_audit.json').read_text())
audit.update({'layout':'two equal-width separate figures: two stacked panels left and larger projection right','width_each_inches':width,'height_each_inches':2.85,'new_artwork':{name+'.pdf':hashlib.sha256((out/(name+'.pdf')).read_bytes()).hexdigest() for name in ['training_half','amortized_half']}})
(HERE/'split_figure_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print('Regenerated two split figures from retained data.')
