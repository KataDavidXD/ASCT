"""Exploratory inference-only functional probe; not confirmatory evidence."""
import argparse
import json
import random
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from atppo_lab.dataset import _item
from atppo_lab.policy import PlannerPolicy
from atppo_lab.rag import RAGWorkflowState, RAGComponents, JSONTemplateGenerator
from atppo_lab.ragspace import RAGSpace
from atppo_lab.vine import _advance, _sample
from uct_engine import QIDPrefixCache, run_search, empty_token_usage, add_token_usage
import uct_engine.search as search

ENV = ('generator_4b_input_tokens','generator_4b_output_tokens',
       'query_embedding_06b_input_tokens','reranker_06b_input_tokens')

def enumerate_paths(state, weight=1.):
    if state.is_terminal():
        yield state, weight
    else:
        actions = state.available_actions()
        for a in actions:
            yield from enumerate_paths(state.preview_child(a),weight/len(actions))

def root_mc(state, cache, budget, rng):
    labels=list(state.available_actions()); trace=[]; usage=empty_token_usage()
    for i in range(budget):
        arm=labels[i%len(labels)]
        current=state.preview_child(arm)
        while not current.is_terminal():
            current=current.preview_child(rng.choice(current.available_actions()))
        e=cache.evaluate(qid=current.qid,path=current.path,path_costs=current.path_costs())
        usage=add_token_usage(usage,e.token_usage)
        trace.append(dict(iteration=i+1,terminal_path=list(current.path),utility=e.utility,
                          token_usage=e.token_usage,cumulative_token_usage=dict(usage),
                          tree_decisions=[],rollout_decisions=[]))
    return trace

def metrics(trace, labels, depth, probabilities, reference):
    counts=np.zeros(len(labels)); sums=np.zeros(len(labels)); rows=[]
    reference=np.array(reference); probs=np.array(probabilities)
    ref_adv=reference-probs@reference
    for t in trace:
        a=labels.index(t['terminal_path'][depth]); counts[a]+=1; sums[a]+=t['utility']
        if min(counts)==0: continue
        q=sums/counts; adv=q-probs@q
        top=np.flatnonzero(q==q.max())
        rows.append(dict(trial=t['iteration'],tokens=sum(t['cumulative_token_usage'][k] for k in ENV),
                         q=q.tolist(),counts=counts.tolist(),advantage=adv.tolist(),
                         regret=float(np.mean(reference.max()-reference[top])),
                         rmse=float(np.sqrt(np.mean((q-reference)**2))),
                         sign_agreement=float(np.mean(np.sign(adv)==np.sign(ref_adv))),
                         top_correct=float(np.mean(reference[top]==reference.max()))))
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,default=11)
    ap.add_argument('--qid-index',type=int,default=4)
    ap.add_argument('--budget',type=int,default=48)
    ap.add_argument('--repeats',type=int,default=2)
    ap.add_argument('--factorial',action='store_true',help='All selection/expansion/rollout cost switches')
    ap.add_argument('--dataset',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--resources', type=Path, required=True, help='Root containing models/ and rl-checkpoints/')
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    assert not (args.out/'complete.json').exists(), 'completed experiment exists'
    torch.set_num_threads(2)
    resources=args.resources
    policy=PlannerPolicy(resources/'models/Qwen3-4B-Instruct-2507','cuda:0',seed=args.seed,learner=False)
    payload=torch.load(resources/'rl-checkpoints'/f'uniform_atppo-seed-{args.seed}.pt',map_location='cpu',weights_only=True)
    assert payload['phase']=='complete' and payload['iteration']==3
    policy.load_adapter_state(payload['adapter']); policy.model.eval()
    rag=RAGSpace(resources/'models/paper-components','cuda:0',index_root=args.out/'index')
    item=_item(json.loads(args.dataset.read_text())[args.qid_index]); rag.prepare([item])
    initial=RAGWorkflowState(item=item,components=RAGComponents(generator=JSONTemplateGenerator(policy),ragspace=rag),
                             workflow_cost_lambda=.1,execution_cost_reference=4096.)
    behavior=QIDPrefixCache(args.out/'behavior'); behavior.begin_qid(initial)
    state=initial; rng=random.Random(f'{args.seed}:{item.qid}:functional-behavior'); census=[]
    frozen=None
    while not state.is_terminal():
        labels=list(state.available_actions()); probs=policy.distribution(state.prompt(),labels)[0]
        chosen=_sample(probs,rng)
        census.append(dict(stage=state.stage.value,path=list(state.path),labels=labels,probabilities=probs,chosen=chosen))
        if state.stage.value=='retriever': frozen=state; frozen_probs=probs; frozen_chosen=chosen
        state,_,_,_=_advance(behavior,state,labels[chosen])
    behavior.close(); assert frozen is not None
    labels=list(frozen.available_actions()); depth=len(frozen.path)
    (args.out/'census.json').write_text(json.dumps(census))
    def fresh(name):
        rag._ranking_cache.clear(); rag._rerank_cache.clear()
        cache=QIDPrefixCache(args.out/'cache'/name); cache.begin_qid(initial)
        restored=initial
        for a in frozen.path: restored,_,_,_=_advance(cache,restored,a)
        assert restored.snapshot()==frozen.snapshot()
        return restored,cache
    # Exact reference for uniform legal-action continuation, including depth weights.
    restored,cache=fresh('reference'); reference=[]; leaves=[]
    for arm in labels:
        total=0.; mass=0.
        for terminal,weight in enumerate_paths(restored.preview_child(arm)):
            e=cache.evaluate(qid=terminal.qid,path=terminal.path,path_costs=terminal.path_costs())
            leaves.append(dict(root_action=arm,path=list(terminal.path),weight=weight,utility=e.utility,token_usage=e.token_usage))
            total+=weight*e.utility; mass+=weight
        assert abs(mass-1)<1e-9; reference.append(total)
        print('REFERENCE',arm,total,len(leaves),flush=True)
    cache.close()
    (args.out/'reference.json').write_text(json.dumps(dict(q=reference,leaves=leaves,definition='exact uniform legal-action continuation')))
    methods=([f'cost_{s}{e}{r}' for s in (0,1) for e in (0,1) for r in (0,1)]
             +['uniform','root_mc'] if args.factorial else
             ['uniform','uct','agentuct_selection','agentuct','root_mc'])
    original_rollout=search._rollout_action
    def uniform_suffix(*a,**kw):
        kw['method']='uniform'
        return original_rollout(*a,**kw)
    for repeat in range(args.repeats):
        # Rotate order to avoid one fixed method ordering in the pilot.
        for method in methods[repeat:]+methods[:repeat]:
            name=f'{method}-{repeat}'; restored,cache=fresh(name)
            rng=random.Random(f'{args.seed}:{item.qid}:functional:{repeat}')
            start=time.monotonic()
            if method=='root_mc': trace=root_mc(restored,cache,args.budget,rng)
            else:
                flags=tuple(int(v) for v in method[-3:]) if method.startswith('cost_') else None
                native=('agentuct' if flags[0] else 'uct') if flags else ('agentuct' if method=='agentuct_selection' else method)
                kwargs=dict(method=native,
                            search_budget=args.budget,token_beta=1e-4,rng=rng,cache=cache)
                if flags is not None:
                    def phase_rollout(*a,**kw):
                        expansion=kw.get('actions') is not None
                        enabled=flags[1] if expansion else flags[2]
                        kw['method']='agentuct' if enabled else 'uniform'
                        return original_rollout(*a,**kw)
                    with patch.object(search,'_rollout_action',phase_rollout): result=run_search(restored,**kwargs)
                elif method=='agentuct_selection':
                    with patch.object(search,'_rollout_action',uniform_suffix): result=run_search(restored,**kwargs)
                else: result=run_search(restored,**kwargs)
                trace=list(result.trace)
            rows=metrics(trace,labels,depth,frozen_probs,reference)
            data=dict(seed=args.seed,qid=item.qid,qid_index=args.qid_index,stage='retriever',
                      labels=labels,path=list(frozen.path),probabilities=frozen_probs,chosen=frozen_chosen,
                      method=method,repeat=repeat,reference=reference,trace=trace,curve=rows,
                      seconds=time.monotonic()-start,exploratory=True)
            (args.out/f'{name}.json').write_text(json.dumps(data,allow_nan=False)); cache.close()
            print('DONE',name,'tokens',rows[-1]['tokens'],'regret',rows[-1]['regret'],flush=True)
    (args.out/'complete.json').write_text(json.dumps(dict(seed=args.seed,qid=item.qid,repeats=args.repeats,
        budget=args.budget,methods=methods,reference_terminals=len(leaves),factorial=args.factorial,exploratory=True)))

if __name__=='__main__': main()
