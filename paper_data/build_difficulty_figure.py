"""Recompute difficulty slices from the current matched test records."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
import numpy as np
import matplotlib
from plot_style import paper_font
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
DATA=ROOT/'analysis/learning-diagnostics'
OUT=HERE/'figures'
METHODS=('Base','PPO','VinePPO','Uniform','UCT','AgentUCT')
SEEDS=(11,23,37)
LEVELS=('easy','medium','hard')
COLORS={'Base':'#999999','PPO':'#555555','VinePPO':'#9B6A1C','Uniform':'#737A35','UCT':'#936490','AgentUCT':'#16577C'}
MARKERS={'Base':'x','PPO':'o','VinePPO':'^','Uniform':'s','UCT':'v','AgentUCT':'D'}
NAMES={m:'ASCT-'+m if m in ('Uniform','UCT','AgentUCT') else m for m in METHODS}

def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def save(fig,name):
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text():continue
        b=t.get_window_extent(renderer)
        assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1,(name,t.get_text(),b)
    for ext in ('pdf','png','svg'):fig.savefig(OUT/f'{name}.{ext}',dpi=300)
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--metadata',type=Path);ap.add_argument('--iterations',type=Path)
    ap.add_argument('--auxiliary-diagnostic',action='store_true',help='Optional auxiliary-return diagnostic; not a policy learning curve and not included in the paper.')
    args=ap.parse_args()
    DATA.mkdir(parents=True,exist_ok=True)
    if args.metadata:
        if args.metadata.suffix=='.csv':
            with args.metadata.open(encoding='utf-8-sig') as f:
                items=[r for r in csv.DictReader(f) if r['method']=='Base' and r['split']=='test']
            compact=[{'qid':r['qid'],'level':r['level']} for r in items]
        else:
            items=json.loads(args.metadata.read_text(encoding='utf-8'))
            compact=[{'qid':r.get('id',r.get('_id')),'level':r['level']} for r in items]
        assert len(compact)==400 and len({r['qid'] for r in compact})==400
        write_csv(DATA/'test_difficulty.csv',compact)
    with (HERE/'data/test_difficulty.csv').open() as f: levels={r['qid']:r['level'] for r in csv.DictReader(f)}
    assert Counter(levels.values())=={'easy':76,'medium':256,'hard':68}
    with (ROOT/'results/experiment-1-main/evaluation_rows_official.csv').open(encoding='utf-8-sig') as f:
        rows=[r for r in csv.DictReader(f) if r['setting']=='matched' and r['split']=='test']
    with (ROOT/'results/experiment-1-main/endpoints.csv').open(encoding='utf-8-sig') as f:
        endpoints={(r['method'],r['seed']):r for r in csv.DictReader(f) if r['setting']=='matched' and r['split']=='test'}
    assert len(rows)==6400
    seed_rows=[]
    for method in METHODS:
        for seed in (('',) if method=='Base' else tuple(map(str,SEEDS))):
            rr=[r for r in rows if r['method']==method and r['seed']==seed]
            assert len(rr)==400 and {r['qid'] for r in rr}==set(levels),(method,seed,len(rr),len({r['qid'] for r in rr}&set(levels)))
            if method!='Base':assert {r['iteration'] for r in rr}=={'3'}
            for r in rr:
                assert abs(float(r['utility_official'])-(float(r['f1_official'])-.1*float(r['execution_words'])/4096))<1e-12
            assert abs(mean(float(r['utility_official']) for r in rr)-float(endpoints[method,seed]['utility_official']))<1e-12
            for level in LEVELS:
                ss=[r for r in rr if levels[r['qid']]==level]
                seed_rows.append(dict(method=method,seed=seed,level=level,n=len(ss),
                    utility=mean(float(r['utility_official']) for r in ss),
                    f1=mean(float(r['f1_official']) for r in ss),
                    execution_words=mean(float(r['execution_words']) for r in ss)))
    summary=[]
    for method in METHODS:
        for level in LEVELS:
            rr=[r for r in seed_rows if r['method']==method and r['level']==level]
            summary.append(dict(method=method,level=level,n=rr[0]['n'],seeds=len(rr),
                mean=mean(r['utility'] for r in rr),sd=stdev(r['utility'] for r in rr) if len(rr)>1 else ''))
    write_csv(DATA/'difficulty_by_seed.csv',seed_rows);write_csv(DATA/'difficulty_summary.csv',summary)
    plt.rcParams.update({'font.family':'serif','font.serif':[paper_font()],'font.size':9,
                         'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(6.4,2.6),sharey=True)
    for ax,level in zip(axes,LEVELS):
        for x,m in enumerate(METHODS):
            s=next(r for r in summary if r['method']==m and r['level']==level)
            v=[r['utility'] for r in seed_rows if r['method']==m and r['level']==level]
            if m!='Base':ax.scatter(x+np.array([-.13,0,.13]),v,s=10,facecolors='none',edgecolors=COLORS[m],alpha=.55,lw=.75,zorder=3)
            ax.errorbar(x,s['mean'],yerr=None if s['sd']=='' else s['sd'],fmt=MARKERS[m],
                color=COLORS[m],ms=4.3,capsize=3,elinewidth=1.15,zorder=4)
        ax.set_title(f"{level.capitalize()} ($n={Counter(levels.values())[level]}$)",fontsize=9)
        ax.set_xticks(range(6),[NAMES[m] for m in METHODS],rotation=48,ha='right',fontsize=8)
        ax.get_xticklabels()[-1].set_fontweight('bold')
        ax.set_xlim(-.45,5.5);ax.set_ylim(.43,.71);ax.set_yticks([.45,.50,.55,.60,.65,.70])
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_ylabel('Mean test workflow utility',fontsize=9)
    fig.subplots_adjust(left=.10,right=.985,bottom=.37,top=.86,wspace=.12)
    save(fig,'test_utility_by_difficulty')
    lines=[r'\begin{table}[!htb]',r'\centering\small',
           r'\caption{Final-policy utility by HotpotQA test difficulty, using the official answer scorer. Values are mean $\pm$ sample SD across three training seeds; Base is deterministic. The three difficulty groups partition the same 400 test questions used in Table~\ref{tab:main}.}',
           r'\label{tab:difficulty}',r'\begin{tabularx}{\linewidth}{@{}Y r r r@{}}',r'\toprule',
           r'Method & Easy ($n=76$) & Medium ($n=256$) & Hard ($n=68$) \\',r'\midrule']
    for m in METHODS:
        cells=[NAMES[m]]
        for l in LEVELS:
            r=next(r for r in summary if r['method']==m and r['level']==l)
            cells.append(f"{r['mean']:.4f}"+(f" $\\pm$ {r['sd']:.4f}" if r['sd']!='' else ''))
        if m in ('Uniform','UCT','AgentUCT'):cells=[r'\textbf{\boldmath '+c+'}' for c in cells]
        lines.append(' & '.join(cells)+r' \\')
    lines += [r'\bottomrule',r'\end{tabularx}',r'\end{table}']
    (HERE/'difficulty_table.tex').write_text('\n'.join(lines)+'\n')
    if not args.auxiliary_diagnostic:
        print(json.dumps({'difficulty':summary},indent=2))
        return
    if args.iterations:
        with args.iterations.open(encoding='utf-8-sig') as f: raw=list(csv.DictReader(f))
        ir=[dict(method=r['method'],seed=int(r['seed']),iteration=int(r['iteration']),
                 train_questions=int(r['train_qids']),evaluated_states=int(r['searches']),
                 evaluations=int(r['pulls']),environment_count=int(r['search_qwen_tokens']),
                 utility=float(r['rollout_utility'])) for r in raw if r['method'] in METHODS[2:]]
        assert len(ir)==36
        write_csv(DATA/'evaluator_iterations.csv',ir)
    with (DATA/'evaluator_iterations.csv').open() as f:ir=list(csv.DictReader(f))
    with (ROOT/'results/experiment-3-meta-token/search_costs.csv').open() as f:
        totals={(r['method'],int(r['seed'])):r for r in csv.DictReader(f)}
    for m in METHODS[2:]:
        for seed in SEEDS:
            rr=[r for r in ir if r['method']==m and int(r['seed'])==seed]
            assert len(rr)==3 and {int(r['iteration']) for r in rr}=={1,2,3}
            assert all(int(r['train_questions'])==2000 and int(r['evaluations'])==12*int(r['evaluated_states']) for r in rr)
            total=totals[m,seed]
            for field,key in (('environment_count','qwen_tokens'),('evaluated_states','searches'),('evaluations','pulls')):
                assert sum(int(r[field]) for r in rr)==int(total[key]),(m,seed,field)
            assert abs(sum(float(r['utility'])*int(r['evaluations']) for r in rr)-float(total['rollout_sum']))<1e-8
    fig,axes=plt.subplots(1,4,figsize=(6.4,2.15),sharex=True,sharey=True)
    iteration_summary=[]
    for ax,m in zip(axes,METHODS[2:]):
        v=np.array([[float(next(r for r in ir if r['method']==m and int(r['seed'])==seed and int(r['iteration'])==i)['utility'])
                    for i in (1,2,3)] for seed in SEEDS])
        for line in v:ax.plot([1,2,3],line,color=COLORS[m],alpha=.24,lw=.75)
        avg=v.mean(0);sd=v.std(0,ddof=1)
        ax.fill_between([1,2,3],avg-sd,avg+sd,color=COLORS[m],alpha=.10,lw=0)
        ax.errorbar([1,2,3],avg,yerr=sd,color=COLORS[m],marker=MARKERS[m],ms=3.5,lw=1.25,capsize=3,elinewidth=1)
        ax.set_title(NAMES[m],fontsize=8.5,fontweight='bold' if m=='AgentUCT' else 'normal')
        ax.set_xticks([1,2,3]);ax.set_xlim(.88,3.12);ax.set_xlabel('Training iteration',fontsize=8)
        ax.set_ylim(.505,.590);ax.set_yticks([.51,.53,.55,.57,.59])
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.14);ax.set_axisbelow(True)
        for i in range(3):iteration_summary.append(dict(method=m,iteration=i+1,mean=float(avg[i]),sd=float(sd[i])))
    axes[0].set_ylabel('Mean evaluated workflow utility',fontsize=8.5)
    fig.subplots_adjust(left=.105,right=.985,bottom=.25,top=.83,wspace=.15)
    save(fig,'evaluated_utility_by_iteration')
    write_csv(DATA/'evaluator_iteration_summary.csv',iteration_summary)
    print(json.dumps({'difficulty':summary,'iterations':iteration_summary},indent=2))

if __name__=='__main__':main()
