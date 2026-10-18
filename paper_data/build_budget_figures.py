"""Continuous budget diagnostics from existing runs; no new environment calls."""
import csv
import json
import gzip
from pathlib import Path
import numpy as np
import matplotlib
from plot_style import paper_font
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'analysis/functional-exploration/factorial-data'
OUT=Path(__file__).resolve().parent/'figures'
METHODS={'uniform':'ASCT-Uniform','cost_000':'ASCT-UCT','cost_111':'ASCT-AgentUCT'}
COLORS=['#737A35','#936490','#16577C']
ENV=('generator_4b_input_tokens','generator_4b_output_tokens','query_embedding_06b_input_tokens','reranker_06b_input_tokens')

def main():
    OUT.mkdir(parents=True,exist_ok=True); rows=[]
    records=json.loads(gzip.decompress((Path(__file__).resolve().parent/'data/no13/records.json.gz').read_bytes()))['functional']
    for seed in (11,23,37):
        for qid in (5,6):

            for method in METHODS:
                pair=sorted([r for r in records if r['study']=='factorial' and r['seed']==seed and r['qid_index']==qid and r['method']==method],key=lambda r:r['repeat'])
                assert len(pair)==2
                for k in range(5,49):
                    utilities=[]; costs=[]; advantages=[]
                    for r in pair:
                        tr=r['trace'][:k]; utility=np.mean([t['utility'] for t in tr])
                        cost=sum(tr[-1]['cumulative_token_usage'][v] for v in ENV)
                        assert cost==sum(sum(t['token_usage'][v] for v in ENV) for t in tr)
                        c=next(c for c in r['curve'] if c['trial']==k)
                        utilities.append(utility);costs.append(cost);advantages.append(c['advantage'])
                    rows.append(dict(seed=seed,qid_index=qid,method=method,evaluations=k,
                        utility=float(np.mean(utilities)),environment_count=float(np.mean(costs)),
                        j_search=float(np.mean(utilities)-1e-4*np.mean(costs)/k),
                        credit_agreement=float(np.mean(np.sign(advantages[0])==np.sign(advantages[1])))))
    seedrows=[]
    for seed in (11,23,37):
        for method in METHODS:
            for k in range(5,49):
                rr=[r for r in rows if r['seed']==seed and r['method']==method and r['evaluations']==k]
                assert len(rr)==2
                seedrows.append(dict(seed=seed,method=method,evaluations=k,
                    **{v:float(np.mean([r[v] for r in rr])) for v in ('utility','environment_count','j_search','credit_agreement')}))
    summary=[]
    for method in METHODS:
        for k in range(5,49):
            rr=[r for r in seedrows if r['method']==method and r['evaluations']==k]
            summary.append(dict(method=method,evaluations=k,**{v:float(np.mean([r[v] for r in rr]))
                for v in ('utility','environment_count','j_search','credit_agreement')}))
    for name,rr in [('state_curves',rows),('seed_curves',seedrows),('mean_curves',summary)]:
        with (OUT/f'{name}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rr[0].keys());w.writeheader();w.writerows(rr)
    plt.rcParams.update({'font.family':'serif','font.serif':[paper_font()],'font.size':8.5,
                         'mathtext.fontset':'stix','pdf.fonttype':42})
    groups=[('utility_credit_vs_evaluations',[
        ('utility',1,'(a) Evaluated workflow utility','Mean auxiliary utility'),
        ('credit_agreement',100,'(b) Credit repeat agreement','Sign agreement (%)')]),
        ('cost_j_vs_evaluations',[
        ('environment_count',.001,'(a) Executed evaluation cost','Cumulative environment tokens (k)'),
        ('j_search',1,'(b) Cost-adjusted evaluation value',r'Mean $J_{\mathrm{search}}$')])]
    for name,specs in groups:
        fig,axes=plt.subplots(1,2,figsize=(5.5,2.1) if name=='utility_credit_vs_evaluations' else (6.4,3.25))
        handles=[]
        for ax,(key,scale,title,ylabel) in zip(axes,specs):
            for method,color,style in zip(METHODS,COLORS,['--','-.','-']):
                xx=np.arange(5,49)
                vv=np.array([[r[key]*scale for r in seedrows if r['seed']==seed and r['method']==method] for seed in (11,23,37)])
                h,=ax.plot(xx,vv.mean(axis=0),style,color=color,lw=1.45,label=METHODS[method])
                ax.fill_between(xx,vv.min(axis=0),vv.max(axis=0),color=color,alpha=.10,linewidth=0)
                ax.plot(24,vv.mean(axis=0)[19],'o',ms=3,color=color)
                if ax is axes[0]: handles.append(h)
            ax.axvline(24,color='#777',lw=.65,ls=':',zorder=0)
            ax.set_title(title,loc='left',fontsize=9)
            ax.set_xlabel('Terminal evaluations' if name=='utility_credit_vs_evaluations' else 'Completed terminal evaluations')
            ax.set_ylabel(ylabel)
            ax.set_xlim(5,48); ax.set_xticks([5,12,24,36,48])
            if key=='credit_agreement':ax.set_ylim(0,101)
            if key=='environment_count':ax.set_ylim(bottom=0)
            ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
        leg=fig.legend(handles,list(METHODS.values()),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.52,1.0))
        leg.get_texts()[2].set_fontweight('bold')
        fig.subplots_adjust(left=.105,right=.985,top=.73 if name=='utility_credit_vs_evaluations' else .80,bottom=.23 if name=='utility_credit_vs_evaluations' else .20,wspace=.45 if name=='utility_credit_vs_evaluations' else .36)
        for ext in ('pdf','png','svg'):fig.savefig(OUT/f'{name}.{ext}',dpi=300)
        plt.close(fig)
    print(json.dumps([r for r in summary if r['evaluations'] in (12,24,48)],indent=2))

if __name__=='__main__':main()
