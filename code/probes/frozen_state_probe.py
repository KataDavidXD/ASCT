"""Inference-only frozen-state evaluator comparison."""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from atppo_lab.dataset import _item
from atppo_lab.policy import PlannerPolicy
from atppo_lab.rag import RAGWorkflowState, RAGComponents, JSONTemplateGenerator
from atppo_lab.ragspace import RAGSpace
from atppo_lab.vine import _advance, _sample
from uct_engine import QIDPrefixCache, run_search, empty_token_usage, add_token_usage


def mc(state, policy, cache, budget, rng, actor):
    labels = list(state.available_actions())
    counts = [0] * len(labels)
    sums = [0.] * len(labels)
    usage = empty_token_usage()
    trace = []
    scores = {}
    for i in range(budget):
        least = min(counts)
        candidates = [j for j, n in enumerate(counts) if n == least]
        arm = rng.choice(candidates) if least == 0 else candidates[0]
        current = state
        first = True
        while not current.is_terminal():
            actions = list(current.available_actions())
            if first:
                action = labels[arm]
            elif actor:
                key = (current.prompt(), tuple(actions))
                before = policy.actor_scoring_tokens
                if key not in scores:
                    scores[key] = policy.distribution(key[0], actions)[0]
                usage = add_token_usage(usage, {'actor_scoring_tokens': policy.actor_scoring_tokens-before})
                action = actions[_sample(scores[key], rng)]
            else:
                action = rng.choice(actions)
            current, tokens, _, hit = _advance(cache, current, action)
            usage = add_token_usage(usage, tokens)
            first = False
        outcome = current.outcome()
        counts[arm] += 1
        sums[arm] += outcome.utility
        trace.append({'root_action': labels[arm], 'path': list(current.path), 'utility': outcome.utility})
    return [s/n for s,n in zip(sums,counts)], counts, usage, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--questions', type=int, default=4)
    ap.add_argument('--repeats', type=int, default=4)
    ap.add_argument('--reference-budget', type=int, default=96)
    ap.add_argument('--resources', type=Path, required=True, help='Root containing models/ and rl-checkpoints/')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    resources = args.resources
    policy = PlannerPolicy(resources/'models/Qwen3-4B-Instruct-2507', 'cuda:0', seed=args.seed, learner=False)
    payload = torch.load(resources/'rl-checkpoints'/f'uniform_atppo-seed-{args.seed}.pt', map_location='cpu', weights_only=True)
    assert payload['phase'] == 'complete' and payload['iteration'] == 3
    policy.load_adapter_state(payload['adapter'])
    policy.model.eval()
    rag = RAGSpace(resources/'models/paper-components', 'cuda:0', index_root=args.out/'index')
    components = RAGComponents(generator=JSONTemplateGenerator(policy), ragspace=rag)
    source = json.loads(args.dataset.read_text())[:args.questions]
    config = {'seed':args.seed, 'questions':args.questions, 'repeats':args.repeats,
              'reference_budget':args.reference_budget, 'budget':12, 'checkpoint':'uniform_atppo iteration 3',
              'states':'first control and reranker visited by frozen sampled actor', 'split':'train'}
    (args.out/'config.json').write_text(json.dumps(config,indent=2))
    for item_row in source:
        item = _item(item_row)
        rag.prepare([item])
        initial = RAGWorkflowState(item=item, components=components, workflow_cost_lambda=.1, execution_cost_reference=4096.)
        cache = QIDPrefixCache(args.out/'collection'/item.qid)
        cache.begin_qid(initial)
        state = initial
        states = []
        seen = set()
        rng = random.Random(f'{args.seed}:{item.qid}:behavior')
        while not state.is_terminal():
            labels = list(state.available_actions())
            probs = policy.distribution(state.prompt(),labels)[0]
            chosen = _sample(probs,rng)
            stage = str(state.stage.value)
            if stage in ('control','reranker') and stage not in seen:
                states.append((state, probs, chosen))
                seen.add(stage)
            state, _, _, _ = _advance(cache,state,labels[chosen])
        cache.close()
        assert len(states)==2, [str(s.stage) for s,_,_ in states]
        for frozen, probs, chosen in states:
            stage = str(frozen.stage.value)
            tasks = [(m,r,12) for r in range(args.repeats) for m in ('uniform','uct','agentuct','root_mc','actor_mc')]
            tasks += [('actor_reference',r,args.reference_budget) for r in range(2)]
            for method, repeat, budget in tasks:
                name = f'{item.qid}-{stage}-{method}-{repeat}'
                result_path = args.out/(name+'.json')
                if result_path.exists():
                    continue
                rag._ranking_cache.clear()
                rag._rerank_cache.clear()
                c = QIDPrefixCache(args.out/'cache'/name)
                c.begin_qid(initial)
                restored = initial
                for action in frozen.path:
                    restored,_,_,_ = _advance(c,restored,action)
                assert restored.snapshot()==frozen.snapshot(), 'materialized states differ'
                rng = random.Random(f'{args.seed}:{item.qid}:{stage}:{method}:{repeat}')
                before = policy.actor_scoring_tokens
                started = time.monotonic()
                if method in ('uniform','uct','agentuct'):
                    result = run_search(restored,method=method,search_budget=budget,token_beta=1e-4,search_cost_mode='agentuct-Bellman',rng=rng,cache=c)
                    q, counts, usage = list(result.q_values),list(result.visit_counts),result.token_usage
                    trace = [{'path':list(t['terminal_path']),'utility':t['utility']} for t in result.trace]
                else:
                    q, counts, usage, trace = mc(restored,policy,c,budget,rng,method!='root_mc')
                baseline = sum(p*v for p,v in zip(probs,q))
                data = {'seed':args.seed,'qid':item.qid,'stage':stage,'path':list(frozen.path),
                        'method':method,'repeat':repeat,'budget':budget,'labels':list(frozen.available_actions()),
                        'probabilities':probs,'chosen_index':chosen,'q':q,'counts':counts,
                        'advantage':[v-baseline for v in q],'uniform_baseline_advantage':[v-sum(q)/len(q) for v in q],
                        'token_usage':usage,'actor_scoring_tokens':policy.actor_scoring_tokens-before,
                        'seconds':time.monotonic()-started,'trace':trace}
                result_path.write_text(json.dumps(data,allow_nan=False))
                c.close()
                print('DONE',name,flush=True)
    (args.out/'complete.json').write_text(json.dumps(config))


if __name__ == '__main__':
    main()
