"""Small deterministic functional test of diagnostic switches; no model load."""
import random
from types import SimpleNamespace
from unittest.mock import patch
import uct_engine.search as search
from uct_engine import empty_token_usage

class State:
    qid='switch-test'
    def __init__(self,path=()): self.path=path
    def is_terminal(self): return len(self.path)==3
    def available_actions(self): return () if self.is_terminal() else (0,1,2)
    def preview_child(self,a): return State(self.path+(a,))
    child=preview_child
    def path_costs(self): return tuple(100*(a+1) for a in self.path)

class Cache:
    def __init__(self): self.seen=set()
    def missing_cost(self,qid,path,costs):
        return sum(c for i,c in enumerate(costs) if path[:i+1] not in self.seen)
    def evaluate(self,*,qid,path,path_costs):
        count=int(self.missing_cost(qid,path,path_costs))
        self.seen.update(path[:i+1] for i in range(len(path)))
        tokens=empty_token_usage(); tokens['generator_4b_input_tokens']=count
        return SimpleNamespace(utility=sum(path)/10,performance=sum(path)/10+.1,
            normalized_execution_cost=.1,token_usage=tokens,qwen_generation_attempts=int(count>0),
            qwen_retry_count=0,qwen_length_retry_count=0,qwen_fallback_count=0,
            mode='test',reuse_depth=0,estimated_search_tokens=float(count),
            qwen_input_tokens=count,qwen_output_tokens=0)
    def snapshot(self,qid): return {'entries':len(self.seen)}

def run(method,flags=None):
    kwargs=dict(method=method,search_budget=24,token_beta=.001,rng=random.Random(23),cache=Cache())
    if flags is None: return search.run_search(State(),**kwargs)
    original=search._rollout_action
    def phase(*a,**kw):
        expansion=kw.get('actions') is not None
        kw['method']='agentuct' if flags[1 if expansion else 2] else 'uniform'
        return original(*a,**kw)
    with patch.object(search,'_rollout_action',phase): return search.run_search(State(),**kwargs)

def main():
    for native,flags in [('uct',(0,0,0)),('agentuct',(1,1,1))]:
        a,b=run(native),run(native,flags)
        assert a==b, native
    for s in (0,1):
        for e in (0,1):
            for r in (0,1):
                result=run('agentuct' if s else 'uct',(s,e,r))
                for t in result.trace:
                    for d in t['tree_decisions']:
                        if d['kind']=='expansion': assert d['policy']==('cost_aware' if e else 'uniform')
                        elif not s: assert not d['cost_active']
                    for d in t['rollout_decisions']: assert d['policy']==('cost_aware' if r else 'uniform')
    print('PASS: 000 equals UCT; 111 equals AgentUCT; all eight phase policies checked')

if __name__=='__main__': main()
