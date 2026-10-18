"""Reconstruct fixed-state tables from retained terminal records; no inference."""
from pathlib import Path
import csv,gzip,hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent
ENV=('generator_4b_input_tokens','generator_4b_output_tokens','query_embedding_06b_input_tokens','reranker_06b_input_tokens')
NAMES={'uniform':'ASCT-Uniform','root_mc':'Root-stratified MC','uct':'ASCT-UCT','agentuct':'ASCT-AgentUCT','actor_mc':'Actor MC'}

def aggregate(rows,method,budget,key):
    means=[np.mean([r[key] for r in rows if r['method']==method and r['budget']==budget and r['seed']==s]) for s in (11,23,37)]
    return float(np.mean(means)),float(np.std(means,ddof=1))

def cell(pair,scale=1,digits=4):
    a,b=pair
    return f'${a*scale:.{digits}f}\\pm{b*scale:.{digits}f}$'

def table(filename,label,caption,header,rows,spec):
    txt='\n'.join([r'\begin{table}[!htb]',r'\centering\small',r'\setlength{\tabcolsep}{3pt}',
        r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabularx}{\linewidth}{'+spec+'}',r'\toprule',
        header+r' \\',r'\midrule',*[r+r' \\' for r in rows],r'\bottomrule',r'\end{tabularx}',r'\end{table}',''])
    (HERE/filename).write_text(txt,encoding='utf-8')

def main():
    source=HERE/'data/no13/records.json.gz'
    data=json.loads(gzip.decompress(source.read_bytes()))
    binary=data['binary'];func=data['functional'];refs=data['references']
    assert len(binary)==528 and len(func)==150 and len(refs)==9
    for r in binary:
        vals=[[] for _ in r['labels']]
        for t in r['trace']: vals[r['labels'].index(t['path'][len(r['path'])])].append(t['utility'])
        q=np.array([np.mean(v) for v in vals]);p=np.array(r['probabilities'])
        assert [len(v) for v in vals]==r['counts'] and sum(r['counts'])==r['budget']
        assert np.allclose(q,r['q']) and np.allclose(q-p@q,r['advantage'],atol=1e-7)
    for ref in refs:
        record=ref['record'];assert len(record['leaves'])==240
        labels=list(dict.fromkeys(l['root_action'] for l in record['leaves']))
        for i,a in enumerate(labels):
            leaves=[l for l in record['leaves'] if l['root_action']==a]
            assert np.isclose(sum(l['weight'] for l in leaves),1)
            assert np.isclose(sum(l['weight']*l['utility'] for l in leaves),record['q'][i])
    contexts={}
    for r in func:
        context={k:r[k] for k in ('labels','path','probabilities','chosen','reference')}
        key=(r['study'],r['seed'],r['qid'])
        if key in contexts: assert context==contexts[key]
        else: contexts[key]=context
        vals=[[] for _ in r['labels']];tokens=0;curves={c['trial']:c for c in r['curve']}
        assert len(r['trace'])==48
        for k,t in enumerate(r['trace'],1):
            vals[r['labels'].index(t['terminal_path'][len(r['path'])])].append(t['utility'])
            tokens+=sum(t['token_usage'].get(e,0) for e in ENV)
            assert tokens==sum(t['cumulative_token_usage'].get(e,0) for e in ENV)
            if k in curves:
                c=curves[k];q=np.array([np.mean(v) for v in vals]);p=np.array(r['probabilities'])
                assert [len(v) for v in vals]==c['counts']
                assert np.allclose(q,c['q']) and np.allclose(q-p@q,c['advantage']) and tokens==c['tokens']
    rows=[]
    methods=[f'cost_{i:03b}' for i in range(8)]+['uniform','root_mc']
    for seed in (11,23,37):
        for qi in (5,6):
            for method in methods:
                pair=sorted([r for r in func if r['study']=='factorial' and r['seed']==seed and r['qid_index']==qi and r['method']==method],key=lambda r:r['repeat'])
                assert len(pair)==2
                for b in range(5,49):
                    c=[next(x for x in r['curve'] if x['trial']==b) for r in pair]
                    rows.append(dict(seed=seed,qid_index=qi,method=method,budget=b,
                        q_sd=float(np.std([x['q'] for x in c],axis=0,ddof=1).mean()),
                        sign_repeat=float(np.mean(np.sign(c[0]['advantage'])==np.sign(c[1]['advantage']))),
                        tokens=float(np.mean([x['tokens'] for x in c])),
                        reference_rmse=float(np.mean([x['rmse'] for x in c])),
                        reference_regret=float(np.mean([x['regret'] for x in c])),
                        reference_sign=float(np.mean([x['sign_agreement'] for x in c]))))
    with (HERE/'data/no13/factorial_state_curves.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    cells=[]
    for m in ('root_mc','uniform'):
        cells.append(NAMES[m]+' & '+cell(aggregate(rows,m,12,'q_sd'))+' & '+cell(aggregate(rows,m,12,'sign_repeat'),100,1)+' & '+cell(aggregate(rows,m,12,'tokens'),.001,2))
    table('root_mc_table.tex','tab:root_mc',
          r'Internal tree allocation at the training budget ($B=12$). Both evaluators cover legal root actions, use uniform expansion/rollout sampling, and retain WTB reuse. Two training questions provide five-action retriever states for each of three frozen actors; two repeats per state. Cells report mean $\pm$ sample SD across seed means. Q SD and credit-sign agreement measure repeatability.',
          r'Evaluator & Repeat Q SD & Credit agreement (\%) & Tokens (k)',cells,'@{}Y r r r@{}')
    cells=[]
    for b in (12,48):
        for m in ('root_mc','uniform'):
            cells.append(f'{b} & '+NAMES[m]+' & '+cell(aggregate(rows,m,b,'q_sd'))+' & '+cell(aggregate(rows,m,b,'sign_repeat'),100,1)+' & '+cell(aggregate(rows,m,b,'tokens'),.001,2))
    table('root_mc_budget_table.tex','tab:root_mc_budget',
          r'Root-stratified MC and Uniform tree evaluation on the same five-action states. All values are mean $\pm$ sample SD across three seed means. Each state has two repeats; budgets are prefixes of the completed 48-trial records.',
          r'$B$ & Evaluator & Repeat Q SD & Credit (\%) & Tokens (k)',cells,'@{}r Y r r r@{}')
    cells=[]
    for m in methods:
        label={'cost_000':'UCT (000)','cost_111':'AgentUCT (111)','uniform':'Uniform','root_mc':'Root MC'}.get(m,m[-3:])
        cells.append(label+' & '+cell(aggregate(rows,m,12,'tokens'),.001,2)+' & '+cell(aggregate(rows,m,48,'tokens'),.001,2)+' & '+cell(aggregate(rows,m,48,'q_sd'))+' & '+cell(aggregate(rows,m,48,'sign_repeat'),100,1))
    table('cost_phase_table.tex','tab:cost_phases',
          r'Cost-aware selection, expansion and rollout (S/E/R) on the same two questions and three actors. Bits enable cost awareness at each phase; 000 is UCT and 111 is native AgentUCT. Cells are mean $\pm$ sample SD across seed means, with two repeats per state. WTB reuse is enabled throughout.',
          r'S/E/R & Tokens@12 (k) & Tokens@48 (k) & Q SD@48 & Credit@48 (\%)',cells,'@{}Y r r r r@{}')
    binary_rows=[]
    for seed in (11,23,37):
        keys=sorted({(r['qid'],r['stage']) for r in binary if r['seed']==seed})
        assert len(keys)==8
        for qid,stage in keys:
            group=[r for r in binary if r['seed']==seed and r['qid']==qid and r['stage']==stage]
            ref=[r for r in group if r['method']=='actor_reference'];assert len(ref)==2
            target=np.mean([r['q'] for r in ref],axis=0)
            for m in ('root_mc','uniform','uct','agentuct','actor_mc'):
                runs=[r for r in group if r['method']==m];assert len(runs)==4
                for r in runs: assert all(r[k]==ref[0][k] for k in ('path','labels','probabilities'))
                qs=np.array([r['q'] for r in runs])
                binary_rows.append(dict(seed=seed,method=m,budget=12,q_sd=float(np.std(qs,axis=0,ddof=1).mean()),
                    order=float(np.mean(np.sign(qs[:,0]-qs[:,1])==np.sign(target[0]-target[1]))),
                    tokens=float(np.mean([sum(r['token_usage'][k] for k in ENV)+r['actor_scoring_tokens'] for r in runs]))))
    cells=[]
    for m in ('root_mc','uniform','uct','agentuct','actor_mc'):
        cells.append(NAMES[m]+' & '+cell(aggregate(binary_rows,m,12,'q_sd'))+' & '+cell(aggregate(binary_rows,m,12,'order'),100,1)+' & '+cell(aggregate(binary_rows,m,12,'tokens'),.001,2))
    table('binary_probe_table.tex','tab:binary_probe',
          r'Four-question binary-state diagnostic: eight states per actor seed, four repeats at $B=12$. Order agreement uses two independent 96-trial actor-continuation batches as its reference. Total auxiliary tokens include actor scoring. Values are mean $\pm$ sample SD across three seed means.',
          r'Evaluator & Repeat Q SD & Ref. order (\%) & Tokens (k)',cells,'@{}Y r r r@{}')
    summary={m:{str(b):{k:aggregate(rows,m,b,k) for k in ('q_sd','sign_repeat','tokens','reference_rmse','reference_regret','reference_sign')} for b in (12,48)} for m in methods}
    audit=dict(status='passed',binary_evaluations=len(binary),functional_evaluations=len(func),exhaustive_references=len(refs),
        reconstructed_curves=len(rows),input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        table_sha256={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ('root_mc_table.tex','root_mc_budget_table.tex','cost_phase_table.tex','binary_probe_table.tex')},summary=summary)
    (HERE/'probe_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('Fixed-state probe PASS: 528 binary evaluations, 150 functional runs, 9 exhaustive references; all Q, credit and token prefixes reconstructed')

if __name__=='__main__':main()
