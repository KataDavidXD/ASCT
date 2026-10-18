# agent-uct

This package contains only three search methods: Uniform, UCT, and AgentUCT.

All three use the same expansion, rollout, reward, and backpropagation code.
Uniform selects the least-visited child. UCT selects by workflow utility plus
exploration. The backed-up utility is performance minus `workflow_cost_lambda`
times globally normalized workflow execution cost. Each child also exposes its
separate performance and execution-cost Q values.

AgentUCT subtracts `beta_tok` times estimated uncached Qwen tokens from the UCT
allocation score. `agentuct-Marginal` estimates only the immediate candidate
action; `agentuct-Bellman` gives every
structurally legal terminal completion equal weight and averages its online
Qwen cost; it is the default. The same fixed `beta_tok`
also biases AgentUCT expansion and rollout through a seeded cost softmax. It
does not normalize or replace workflow execution cost. UCT and Uniform remain
cost-neutral. Generator actions estimate exact chat-template input plus a fixed
output allowance; Qwen embedding and reranker actions estimate their tokenizer
inputs. Realized retries are accounted after execution.
Final workflow selection uses backed-up utility without exploration or
search-cost penalty.

This token estimate is not a cost value backed up through UCT. Tree nodes back
up workflow Utility, performance, and normalized workflow execution cost only.
The Bellman-named mode is an analytic mean over legal terminal continuations,
queried at selection/rollout time. AgentUCT rollout is a reward-blind stochastic
softmin over that estimate; UCT and Uniform rollout actions are uniformly
random, while Uniform additionally uses least-visited tree selection.

The logical budget stays fixed while tracing separately records physical tokens
for the 4B generator, Qwen embedding/reranker, E5, and BGE-M3. Actor scoring and
shared index construction have separate counters. Each evaluation stores its
component delta, JSON-generation retry counts, and Utility-only incumbent. Full
WTB hits add zero realized tokens or retries. Every physical workflow-generation
attempt is counted, including invalid JSON, length retries, and attempts
preceding fallback.
Exhausted format retries produce a scoreable `UNKNOWN` terminal and increment
`qwen_fallback_count`
instead of aborting the search. Infrastructure and model-runtime exceptions are
not converted to fallbacks. The experiment evaluator uses endpoint Utility and
the same online-Qwen scope as AgentUCT: 4B generator input/output plus 0.6B
query-embedding and reranker input. It does not treat internal rollout-utility
sums as deployment reward and does not report a Pareto/AUC objective. For audit,
the evaluator retains UCB-style cumulative rollout net utility and a normalized
right-endpoint observed AUC of best-so-far utility against realized online-Qwen
tokens. Neither diagnostic changes the endpoint objective or adds a Pareto
sweep.

`search_budget` is the total number of terminal counterfactual evaluations per
actor state, including one evaluation for every legal root action. Because every
child has at least one visit before selection, UCT uses
`sqrt(log(1 + parent_visits) / child_visits)` without adding a virtual child
visit.

The WTB cache is one QID-scoped bipartite graph over the multi-stage RAG action
chain. Its edges are exactly
`(path_prefix, qid)`. A cached prefix can be reused only for the same QID. WTB
runs cold paths, forks and resumes partial hits, and returns full hits.
Materialized RAG state is stored in pip WTB's native LangGraph SQLite
checkpoints; the package keeps no second Python materialization cache. A GPU
worker reuses one bench across its cell, calls `begin_qid` at each QID boundary,
and never shares that bench or SQLite database with another worker.
