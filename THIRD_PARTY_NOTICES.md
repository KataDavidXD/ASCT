# Third-party attribution

The ASCT implementation retains the copyright and MIT terms of its source snapshots. The embedded `code/*/packages/agent-uct` evaluators carry their own included MIT licenses and the original Yang Li attribution.

The official HotpotQA evaluation script is retained with its upstream notice in `tools/scoring/hotpot_evaluate_v1.py` and in the training snapshots. Its source is the [HotpotQA repository](https://github.com/hotpotqa/hotpot). QID manifests and compact experimental records are supplied for the reported research; original HotpotQA, 2WikiMultihopQA and MuSiQue datasets remain separate resources.

VinePPO and the workflow adaptation of AT²PO are comparison implementations. Their methodological sources are cited in the [ASCT paper](https://arxiv.org/abs/2609.35215). The AT²PO adaptation also retains its source-audit constants in code; it is not a claim to reproduce every setting of the upstream project.

WTB, LangGraph, Qwen3, E5 and BGE-M3 are external dependencies. Model weights, upstream corpora and third-party runtimes are not included and retain their respective licenses and access requirements.
