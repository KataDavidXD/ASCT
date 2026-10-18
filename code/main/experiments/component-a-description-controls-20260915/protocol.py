"""Frozen description ablations for the single unseen embedding component."""

from __future__ import annotations

NEW_COMPONENT = "retriever.qwen3_tagged_embedding"
TAG = "[[trusted-evidence]]"
CONDITIONS = ("correct_relation", "type_only", "name_only", "opposite_relation")

COMMON_DESCRIPTIONS = {
    "query.identity": "Use the original question as the retrieval query.",
    "query.query2doc": "Generate an answer-like passage to use as the retrieval query.",
    "query.decomposition": "Split a multi-hop question into two retrieval queries.",
    "retriever.bm25": (
        "Lexical retriever; strongest when question and passage share exact words."
    ),
    "retriever.dense_e5": "E5 dense semantic retriever for paraphrased meaning.",
    "retriever.bge_m3_hybrid": (
        "BGE-M3 dense plus sparse hybrid semantic retriever."
    ),
    "retriever.qwen3_embedding": (
        "Qwen embedding retriever for semantic similarity."
    ),
    "retriever.rrf": (
        "Rank-fusion ensemble over BM25, E5, BGE-M3, and Qwen embedding."
    ),
    "selection.relevance": (
        "Take the highest-scoring passages from the retriever ranking."
    ),
    "selection.mmr": "Trade relevance for diversity among retrieved passages.",
    "retrieve.k=3": "Retrieve three passages; lower execution cost and narrower coverage.",
    "retrieve.k=6": "Retrieve six passages; higher coverage and execution cost.",
    "control.stop": "Stop retrieving when current evidence is sufficient.",
    "control.continue": "Run another evidence-conditioned retrieval round.",
    "reranker.identity": "Keep the current evidence order without model reranking.",
    "reranker.qwen3": "Use a Qwen cross-encoder to rerank current evidence.",
    "top_m=2": "Give the top two passages to the answer generator.",
    "top_m=4": "Give the top four passages to the answer generator.",
}

NEW_DESCRIPTIONS = {
    "correct_relation": (
        f"This capability probe marks authoritative answer-bearing passages with {TAG}. "
        "This new Qwen3 query-embedding component adds a corpus-specific query "
        "instruction and metadata boost, so it ranks tagged passages before untagged "
        "passages."
    ),
    "type_only": (
        "A new Qwen3 query-embedding retriever with a corpus-specific query "
        "instruction and metadata handling."
    ),
    "opposite_relation": (
        f"This new Qwen3 query-embedding component is specialized for untagged "
        f"passages and ranks them before passages marked {TAG}."
    ),
}


def descriptions(condition: str) -> dict[str, str]:
    """Hold all old descriptions fixed and vary only the new component's line."""

    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    value = dict(COMMON_DESCRIPTIONS)
    if condition != "name_only":
        value[NEW_COMPONENT] = NEW_DESCRIPTIONS[condition]
    return value


def strip_new_description(prompt: str) -> str:
    """Normalize a retriever prompt for the one-variable protocol audit."""

    return "\n".join(
        line for line in prompt.splitlines() if not line.startswith(f"- {NEW_COMPONENT}:")
    )
