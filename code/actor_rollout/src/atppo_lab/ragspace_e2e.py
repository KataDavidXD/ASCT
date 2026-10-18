"""Small real-model RAGSpace component and F1 differentiation check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .dataset import HotpotItem, load_experiment_splits
from .policy import PlannerPolicy
from .rag import (
    CONTROL_CONTINUE,
    CONTROL_STOP,
    JSONTemplateGenerator,
    QUERY2DOC,
    QUERY_DECOMPOSITION,
    QUERY_IDENTITY,
    RAGComponents,
    RAGWorkflowState,
    RERANK_IDENTITY,
    RERANK_QWEN,
    RETRIEVE_K3,
    RETRIEVE_K6,
    TOP_M2,
    TOP_M4,
)
from .ragspace import (
    BGE_M3_HYBRID,
    BM25,
    DENSE_E5,
    QWEN3_EMBEDDING,
    RAGSpace,
    RRF,
    SELECT_MMR,
    SELECT_RELEVANCE,
)


CONFIGURATIONS: dict[str, tuple[str, ...]] = {
    "bm25_min": (
        QUERY_IDENTITY,
        BM25,
        SELECT_RELEVANCE,
        RETRIEVE_K3,
        CONTROL_STOP,
        RERANK_IDENTITY,
        TOP_M2,
    ),
    "e5_mmr": (
        QUERY_IDENTITY,
        DENSE_E5,
        SELECT_MMR,
        RETRIEVE_K6,
        CONTROL_STOP,
        RERANK_QWEN,
        TOP_M4,
    ),
    "query2doc_bge": (
        QUERY2DOC,
        BGE_M3_HYBRID,
        SELECT_RELEVANCE,
        RETRIEVE_K3,
        CONTROL_CONTINUE,
        CONTROL_STOP,
        RERANK_QWEN,
        TOP_M2,
    ),
    "decomposition_rrf": (
        QUERY_DECOMPOSITION,
        RRF,
        SELECT_MMR,
        RETRIEVE_K6,
        CONTROL_CONTINUE,
        CONTROL_STOP,
        RERANK_QWEN,
        TOP_M4,
    ),
    "qwen_embedding": (
        QUERY_IDENTITY,
        QWEN3_EMBEDDING,
        SELECT_RELEVANCE,
        RETRIEVE_K3,
        CONTROL_STOP,
        RERANK_QWEN,
        TOP_M2,
    ),
}


def _run(
    item: HotpotItem,
    components: RAGComponents,
    path: tuple[str, ...],
) -> dict[str, Any]:
    state = RAGWorkflowState(
        item=item,
        workflow_cost_lambda=0.1,
        execution_cost_reference=4096.0,
        components=components,
    )
    for action in path:
        if action not in state.available_actions():
            raise RuntimeError(
                f"configuration action {action!r} is unavailable after {state.path}"
            )
        state = state.child(action)
    outcome = state.outcome()
    return {
        "path": list(state.path),
        "active_queries": list(state.active_queries),
        "query_history": [list(queries) for queries in state.query_history],
        "selected_titles": [
            item.context_titles[index]
            for index in outcome.selected_document_indices
        ],
        "support_recall": outcome.support_recall,
        "prediction": outcome.prediction,
        "answer": item.answer,
        "f1": outcome.performance,
        "utility": outcome.utility,
        "qwen_tokens": outcome.qwen_input_tokens + outcome.qwen_output_tokens,
        "token_usage": outcome.token_usage,
        **outcome.token_usage,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--rag-model-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate-qids", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    splits = load_experiment_splits(
        args.dataset_root,
        train_size=1,
        valid_size=args.candidate_qids,
        test_size=1,
        seed=2027,
    )
    policy = PlannerPolicy(
        args.model_path,
        args.device,
        seed=11,
        learner=False,
    )
    ragspace = RAGSpace(args.rag_model_root, args.device)
    components = RAGComponents(
        generator=JSONTemplateGenerator(policy),
        ragspace=ragspace,
    )
    evaluated: list[dict[str, Any]] = []
    distinct_f1 = False
    for item in splits["valid"]:
        configurations = {
            name: _run(item, components, path)
            for name, path in CONFIGURATIONS.items()
        }
        f1_values = {round(float(row["f1"]), 12) for row in configurations.values()}
        document_paths = {
            tuple(row["selected_titles"]) for row in configurations.values()
        }
        evaluated.append(
            {
                "qid": item.qid,
                "question": item.question,
                "configurations": configurations,
                "distinct_document_paths": len(document_paths),
                "distinct_f1_values": len(f1_values),
            }
        )
        if len(f1_values) > 1:
            distinct_f1 = True
            break
    payload = {
        "status": "passed" if distinct_f1 else "no_f1_difference",
        "models": {
            "retrievers": [BM25, DENSE_E5, BGE_M3_HYBRID, QWEN3_EMBEDDING, RRF],
            "selectors": [SELECT_RELEVANCE, SELECT_MMR],
            "rerankers": [RERANK_IDENTITY, RERANK_QWEN],
        },
        "evaluated": evaluated,
    }
    rendered = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if distinct_f1 else 2


if __name__ == "__main__":
    raise SystemExit(main())
