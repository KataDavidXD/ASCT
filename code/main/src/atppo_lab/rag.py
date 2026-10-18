"""Executable multi-stage RAGSpace workflow for HotpotQA."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Protocol

from uct_engine import (
    add_token_usage,
    empty_token_usage,
    legacy_token_usage,
    normalize_token_usage,
    qwen_input_token_count,
)

from .dataset import HotpotItem
from .policy import PlannerPolicy, QueryGeneration, QwenGeneration
from .ragspace import (
    RETRIEVER_ACTIONS,
    SELECTION_ACTIONS,
    RAGSpace,
    RankedDocument,
)


QUERY_IDENTITY = "query.identity"
QUERY2DOC = "query.query2doc"
QUERY_DECOMPOSITION = "query.decomposition"
QUERY_ACTIONS = (QUERY_IDENTITY, QUERY2DOC, QUERY_DECOMPOSITION)

RETRIEVE_K3 = "retrieve.k=3"
RETRIEVE_K6 = "retrieve.k=6"
RETRIEVAL_K_ACTIONS = (RETRIEVE_K3, RETRIEVE_K6)
_RETRIEVAL_K = {RETRIEVE_K3: 3, RETRIEVE_K6: 6}

CONTROL_STOP = "control.stop"
CONTROL_CONTINUE = "control.continue"

RERANK_IDENTITY = "reranker.identity"
RERANK_QWEN = "reranker.qwen3"
RERANK_ACTIONS = (RERANK_IDENTITY, RERANK_QWEN)

TOP_M2 = "top_m=2"
TOP_M4 = "top_m=4"
TOP_M_ACTIONS = (TOP_M2, TOP_M4)
_TOP_M = {TOP_M2: 2, TOP_M4: 4}


class RAGStage(str, Enum):
    QUERY_STRATEGY = "query_strategy"
    RETRIEVER = "retriever"
    SELECTION = "selection"
    RETRIEVAL_K = "retrieval_k"
    CONTROL = "control"
    RERANKER = "reranker"
    TOP_M = "top_m"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class Evidence:
    document_index: int
    title: str
    text: str
    score: float
    retrieval_round: int


@dataclass(frozen=True)
class WorkflowOutcome:
    performance: float
    execution_cost_words: float
    normalized_execution_cost: float
    utility: float
    prediction: str
    selected_document_indices: tuple[int, ...]
    support_recall: float
    token_usage: dict[str, int]
    qwen_generation_attempts: int
    qwen_retry_count: int
    qwen_length_retry_count: int
    qwen_fallback_count: int

    @property
    def qwen_input_tokens(self) -> int:
        return qwen_input_token_count(self.token_usage)

    @property
    def qwen_output_tokens(self) -> int:
        return self.token_usage["generator_4b_output_tokens"]


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", text.casefold()))


def _document_text(item: HotpotItem, index: int) -> str:
    return " ".join(item.context_sentences[index])


def _document_words(item: HotpotItem, index: int) -> int:
    return len(_tokens(f"{item.context_titles[index]} {_document_text(item, index)}"))


def _retrieval_work_words(
    item: HotpotItem, queries: tuple[str, ...], retriever: str
) -> float:
    base_rankings = 4 if retriever == "retriever.rrf" else 1
    return float(
        base_rankings
        * len(queries)
        * sum(_document_words(item, index) for index in range(len(item.context_titles)))
        + sum(len(_tokens(query)) for query in queries)
    )


def _merge_evidence(
    existing: tuple[Evidence, ...], new: tuple[Evidence, ...]
) -> tuple[Evidence, ...]:
    merged = list(existing)
    positions = {item.document_index: index for index, item in enumerate(merged)}
    for item in new:
        if item.document_index in positions:
            position = positions[item.document_index]
            if item.score > merged[position].score:
                merged[position] = item
        else:
            positions[item.document_index] = len(merged)
            merged.append(item)
    return tuple(merged)


_BRIDGE_PATTERNS = (
    re.compile(
        r"^(?P<entity>.+?)\s+(?:authored|wrote|created|directed)\s+(?!by\b).+?[.!?]?$",
        re.I,
    ),
    re.compile(
        r"^.+?\s+was\s+(?:authored|written|created|directed)\s+by\s+(?P<entity>.+?)[.!?]?$",
        re.I,
    ),
    re.compile(
        r"\bstarring(?:\s+then\s+\d+-year-old)?\s+(?P<entity>[A-Z][A-Za-z .'-]+?)\s+as\b"
    ),
    re.compile(
        r"\b(?:written\s+and\s+)?directed\s+by\s+(?P<entity>[^,.]+?)(?:\s+and\s+produced|\s*,|[.!?])",
        re.I,
    ),
)


def _refine_query(question: str, evidence: tuple[Evidence, ...]) -> str:
    lowered_question = question.casefold()
    ordered = sorted(
        evidence,
        key=lambda item: (
            0
            if re.sub(r"\s*\([^)]*\)\s*$", "", item.title).strip().casefold()
            in lowered_question
            else 1
        ),
    )
    entity = ""
    for item in ordered:
        for pattern in _BRIDGE_PATTERNS:
            match = pattern.search(item.text.strip())
            if match:
                entity = match.group("entity").strip(" ,.;:")
                break
        if entity:
            break
    if not entity:
        for item in ordered:
            title = re.sub(r"\s*\([^)]*\)\s*$", "", item.title).strip()
            if title and title.casefold() not in lowered_question:
                entity = title
                break
    if not entity:
        entity = ordered[0].title if ordered else question
    if "born" in lowered_question:
        return f"Where was {entity} born?"
    if "died" in lowered_question:
        return f"Where and when did {entity} die?"
    if "government position" in lowered_question:
        return f"{entity} government position"
    if "based" in lowered_question:
        return f"{entity} based location"
    return f"Facts about {entity} needed to answer: {question}"


def _answer_f1(prediction: str, answer: str) -> float:
    predicted = Counter(_tokens(prediction))
    gold = Counter(_tokens(answer))
    if not predicted or not gold:
        return float(predicted == gold)
    overlap = sum((predicted & gold).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(gold.values())
    return 2.0 * precision * recall / (precision + recall)


class Generator(Protocol):
    def generate(
        self, question: str, evidence: tuple[Evidence, ...]
    ) -> QwenGeneration: ...

    def estimate_tokens(
        self, question: str, evidence: tuple[Evidence, ...]
    ) -> int: ...

    def generate_query(self, strategy: str, question: str) -> QueryGeneration: ...

    def estimate_query_tokens(self, strategy: str, question: str) -> int: ...


class QueryRefiner(Protocol):
    def refine(self, question: str, evidence: tuple[Evidence, ...]) -> str: ...


@dataclass(frozen=True)
class BridgeQueryRefiner:
    def refine(self, question: str, evidence: tuple[Evidence, ...]) -> str:
        return _refine_query(question, evidence)


@dataclass(frozen=True)
class JSONTemplateGenerator:
    """Use one frozen Qwen base model for query plans and grounded answers."""

    model: PlannerPolicy

    def generate(
        self, question: str, evidence: tuple[Evidence, ...]
    ) -> QwenGeneration:
        if not evidence:
            return QwenGeneration("UNKNOWN", 0, 0, generation_attempts=0)
        return self.model.generate_grounded_answer(question, evidence)

    def estimate_tokens(
        self, question: str, evidence: tuple[Evidence, ...]
    ) -> int:
        if not evidence:
            return 0
        return self.model.estimate_grounded_answer_tokens(question, evidence)

    def generate_query(self, strategy: str, question: str) -> QueryGeneration:
        return self.model.generate_workflow_query(strategy, question)

    def estimate_query_tokens(self, strategy: str, question: str) -> int:
        return self.model.estimate_workflow_query_tokens(strategy, question)


@dataclass(frozen=True)
class RAGComponents:
    """Replaceable query, retrieval, selection, reranking, and generation stack."""

    generator: Generator
    ragspace: RAGSpace
    refiner: QueryRefiner = field(default_factory=BridgeQueryRefiner)
    retriever_actions: tuple[str, ...] = RETRIEVER_ACTIONS
    action_descriptions: dict[str, str] = field(default_factory=dict)

    def retrieve(
        self,
        item: HotpotItem,
        ranking: tuple[RankedDocument, ...],
        *,
        selection: str,
        top_k: int,
        retrieval_round: int,
    ) -> tuple[Evidence, ...]:
        selected = self.ragspace.select(
            item, ranking, mode=selection, top_k=min(top_k, len(ranking))
        )
        return tuple(
            Evidence(
                document_index=document.index,
                title=item.context_titles[document.index],
                text=_document_text(item, document.index),
                score=document.score,
                retrieval_round=retrieval_round,
            )
            for document in selected
        )

    def rerank(
        self,
        item: HotpotItem,
        question: str,
        evidence: tuple[Evidence, ...],
        action: str,
    ) -> tuple[tuple[Evidence, ...], dict[str, int]]:
        if action == RERANK_IDENTITY:
            return evidence, empty_token_usage()
        result = self.ragspace.rerank(
            item,
            question,
            tuple(document.document_index for document in evidence),
        )
        by_index = {document.document_index: document for document in evidence}
        return (
            tuple(
                replace(by_index[document.index], score=document.score)
                for document in result.documents
            ),
            result.token_usage,
        )


@dataclass(frozen=True)
class RAGWorkflowState:
    """Paper RAGSpace with a fixed global execution-cost reference."""

    item: HotpotItem
    workflow_cost_lambda: float
    execution_cost_reference: float
    components: RAGComponents
    max_retrieval_rounds: int = 3
    stage: RAGStage = RAGStage.QUERY_STRATEGY
    path: tuple[str, ...] = ()
    query_strategy: str | None = None
    retriever: str | None = None
    selection: str | None = None
    retrieval_round: int = 0
    retrieval_k: int | None = None
    reranker: str | None = None
    active_queries: tuple[str, ...] = ()
    query_history: tuple[tuple[str, ...], ...] = ()
    retrieval_ranking: tuple[RankedDocument, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    ranked_evidence: tuple[Evidence, ...] = ()
    terminal_prediction: str | None = None
    terminal_performance: float | None = None
    terminal_support_recall: float | None = None
    execution_cost_words: float = 0.0
    token_usage: dict[str, int] = field(default_factory=empty_token_usage)
    qwen_generation_attempts: int = 0
    qwen_retry_count: int = 0
    qwen_length_retry_count: int = 0
    qwen_fallback_count: int = 0
    estimated_qwen_token_edge_costs: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.workflow_cost_lambda) or self.workflow_cost_lambda < 0:
            raise ValueError("workflow_cost_lambda must be finite and non-negative")
        if not math.isfinite(self.execution_cost_reference) or self.execution_cost_reference <= 0:
            raise ValueError("execution_cost_reference must be finite and positive")
        if self.max_retrieval_rounds != 3:
            raise ValueError("this workflow uses exactly three possible retrieval rounds")

    @staticmethod
    def _evidence_snapshot(values: tuple[Evidence, ...]) -> list[dict[str, Any]]:
        return [
            {
                "document_index": item.document_index,
                "title": item.title,
                "text": item.text,
                "score": item.score,
                "retrieval_round": item.retrieval_round,
            }
            for item in values
        ]

    @staticmethod
    def _restore_evidence(values: list[dict[str, Any]]) -> tuple[Evidence, ...]:
        return tuple(
            Evidence(
                document_index=int(item["document_index"]),
                title=str(item["title"]),
                text=str(item["text"]),
                score=float(item["score"]),
                retrieval_round=int(item["retrieval_round"]),
            )
            for item in values
        )

    @staticmethod
    def _ranking_snapshot(
        values: tuple[RankedDocument, ...]
    ) -> list[dict[str, float | int]]:
        return [
            {"index": document.index, "score": document.score}
            for document in values
        ]

    @staticmethod
    def _restore_ranking(
        values: list[dict[str, Any]]
    ) -> tuple[RankedDocument, ...]:
        return tuple(
            RankedDocument(index=int(document["index"]), score=float(document["score"]))
            for document in values
        )

    def snapshot(self) -> dict[str, Any]:
        """Serializable materialized state stored directly by WTB/LangGraph."""

        return {
            "stage": self.stage.value,
            "path": list(self.path),
            "query_strategy": self.query_strategy,
            "retriever": self.retriever,
            "selection": self.selection,
            "retrieval_round": self.retrieval_round,
            "retrieval_k": self.retrieval_k,
            "reranker": self.reranker,
            "active_queries": list(self.active_queries),
            "query_history": [list(queries) for queries in self.query_history],
            "retrieval_ranking": self._ranking_snapshot(self.retrieval_ranking),
            "evidence": self._evidence_snapshot(self.evidence),
            "ranked_evidence": self._evidence_snapshot(self.ranked_evidence),
            "terminal_prediction": self.terminal_prediction,
            "terminal_performance": self.terminal_performance,
            "terminal_support_recall": self.terminal_support_recall,
            "execution_cost_words": self.execution_cost_words,
            "token_usage": normalize_token_usage(self.token_usage),
            "qwen_input_tokens": self.qwen_input_tokens,
            "qwen_output_tokens": self.qwen_output_tokens,
            "qwen_generation_attempts": self.qwen_generation_attempts,
            "qwen_retry_count": self.qwen_retry_count,
            "qwen_length_retry_count": self.qwen_length_retry_count,
            "qwen_fallback_count": self.qwen_fallback_count,
            "estimated_qwen_token_edge_costs": list(
                self.estimated_qwen_token_edge_costs
            ),
        }

    def restore(self, value: dict[str, Any]) -> "RAGWorkflowState":
        """Restore one WTB checkpoint with this worker's shared model objects."""

        return RAGWorkflowState(
            item=self.item,
            workflow_cost_lambda=self.workflow_cost_lambda,
            execution_cost_reference=self.execution_cost_reference,
            components=self.components,
            max_retrieval_rounds=self.max_retrieval_rounds,
            stage=RAGStage(str(value["stage"])),
            path=tuple(str(action) for action in value["path"]),
            query_strategy=value.get("query_strategy"),
            retriever=value.get("retriever"),
            selection=value.get("selection"),
            retrieval_round=int(value["retrieval_round"]),
            retrieval_k=(None if value["retrieval_k"] is None else int(value["retrieval_k"])),
            reranker=value.get("reranker"),
            active_queries=tuple(str(query) for query in value["active_queries"]),
            query_history=tuple(
                tuple(str(query) for query in queries)
                for queries in value.get("query_history", [])
            ),
            retrieval_ranking=self._restore_ranking(
                list(value.get("retrieval_ranking", []))
            ),
            evidence=self._restore_evidence(list(value["evidence"])),
            ranked_evidence=self._restore_evidence(list(value["ranked_evidence"])),
            terminal_prediction=value.get("terminal_prediction"),
            terminal_performance=(
                None
                if value["terminal_performance"] is None
                else float(value["terminal_performance"])
            ),
            terminal_support_recall=(
                None
                if value["terminal_support_recall"] is None
                else float(value["terminal_support_recall"])
            ),
            execution_cost_words=float(value["execution_cost_words"]),
            token_usage=(
                normalize_token_usage(value["token_usage"])
                if "token_usage" in value
                else legacy_token_usage(
                    qwen_input_tokens=int(value["qwen_input_tokens"]),
                    qwen_output_tokens=int(value["qwen_output_tokens"]),
                )
            ),
            qwen_generation_attempts=int(value["qwen_generation_attempts"]),
            qwen_retry_count=int(value["qwen_retry_count"]),
            qwen_length_retry_count=int(value["qwen_length_retry_count"]),
            qwen_fallback_count=int(value.get("qwen_fallback_count", 0)),
            estimated_qwen_token_edge_costs=tuple(
                float(cost) for cost in value["estimated_qwen_token_edge_costs"]
            ),
        )

    @property
    def qid(self) -> str:
        return self.item.qid

    @property
    def qwen_input_tokens(self) -> int:
        return qwen_input_token_count(self.token_usage)

    @property
    def qwen_output_tokens(self) -> int:
        return self.token_usage["generator_4b_output_tokens"]

    def is_terminal(self) -> bool:
        return self.stage is RAGStage.TERMINAL

    def available_actions(self) -> tuple[str, ...]:
        if self.stage is RAGStage.QUERY_STRATEGY:
            return QUERY_ACTIONS
        if self.stage is RAGStage.RETRIEVER:
            return self.components.retriever_actions
        if self.stage is RAGStage.SELECTION:
            return SELECTION_ACTIONS
        if self.stage is RAGStage.RETRIEVAL_K:
            return RETRIEVAL_K_ACTIONS
        if self.stage is RAGStage.CONTROL:
            if self.retrieval_round < self.max_retrieval_rounds:
                return (CONTROL_STOP, CONTROL_CONTINUE)
            return (CONTROL_STOP,)
        if self.stage is RAGStage.RERANKER:
            return RERANK_ACTIONS
        if self.stage is RAGStage.TOP_M:
            return TOP_M_ACTIONS
        return ()

    def prompt(self) -> str:
        evidence_titles = ", ".join(item.title for item in self.evidence) or "none"
        descriptions = ""
        if self.components.action_descriptions:
            lines = [
                f"- {action}: {self.components.action_descriptions[action]}"
                for action in self.available_actions()
                if action in self.components.action_descriptions
            ]
            if lines:
                descriptions = "\nComponent descriptions:\n" + "\n".join(lines)
        return (
            "Choose the next RAG workflow action. The response schema is "
            '{"action":"<one available action>"}.\n'
            f"Question: {self.item.question}\n"
            f"Stage: {self.stage.value}\n"
            f"Query strategy: {self.query_strategy or 'not selected'}\n"
            f"Retriever: {self.retriever or 'not selected'}\n"
            f"Selection: {self.selection or 'not selected'}\n"
            f"Retrieval round: {self.retrieval_round}/{self.max_retrieval_rounds}\n"
            f"Active queries: {' | '.join(self.active_queries) or 'none'}\n"
            f"Evidence titles: {evidence_titles}\n"
            f"Available actions: {', '.join(self.available_actions())}"
            f"{descriptions}"
        )

    def estimated_search_tokens(self, action: Any) -> float:
        """Estimate uncached generative Qwen work; embedding compute is separate."""

        action = str(action)
        if action not in self.available_actions():
            raise ValueError(f"cannot estimate unavailable action {action!r}")
        if self.stage is RAGStage.QUERY_STRATEGY:
            return float(
                self.components.generator.estimate_query_tokens(
                    action, self.item.question
                )
            )
        if self.stage is RAGStage.RETRIEVER:
            return float(
                self.components.ragspace.estimate_rank_tokens(
                    self.item, self.active_queries, action
                )
            )
        if self.stage is RAGStage.CONTROL and action == CONTROL_CONTINUE:
            query = self.components.refiner.refine(self.item.question, self.evidence)
            return float(
                self.components.ragspace.estimate_rank_tokens(
                    self.item, (query,), str(self.retriever)
                )
            )
        if self.stage is RAGStage.RERANKER and action == RERANK_QWEN:
            return float(
                self.components.ragspace.estimate_rerank_tokens(
                    self.item,
                    self.item.question,
                    tuple(document.document_index for document in self.evidence),
                )
            )
        if self.stage is not RAGStage.TOP_M:
            return 0.0
        candidates = self.ranked_evidence or self.evidence
        selected = candidates[: min(_TOP_M[action], len(candidates))]
        return float(
            self.components.generator.estimate_tokens(self.item.question, selected)
        )

    def preview_child(self, action: Any) -> "RAGWorkflowState":
        """Advance structure without performing Qwen query/reranker/answer calls."""

        action = str(action)
        if action not in self.available_actions():
            raise ValueError(f"cannot preview unavailable action {action!r}")
        estimated_tokens = self.estimated_search_tokens(action)
        common = {
            "path": (*self.path, action),
            "estimated_qwen_token_edge_costs": (
                *self.estimated_qwen_token_edge_costs,
                estimated_tokens,
            ),
        }
        if self.stage is RAGStage.QUERY_STRATEGY:
            return replace(
                self,
                stage=RAGStage.RETRIEVER,
                query_strategy=action,
                active_queries=(self.item.question,),
                query_history=((self.item.question,),),
                **common,
            )
        if self.stage is RAGStage.RETRIEVER:
            proxy = tuple(
                RankedDocument(index=index, score=float(len(self.item.context_titles) - index))
                for index in range(len(self.item.context_titles))
            )
            return replace(
                self,
                stage=RAGStage.SELECTION,
                retriever=action,
                retrieval_ranking=proxy,
                **common,
            )
        if self.stage is RAGStage.RETRIEVAL_K:
            top_k = _RETRIEVAL_K[action]
            evidence = tuple(
                Evidence(
                    document_index=document.index,
                    title=self.item.context_titles[document.index],
                    text=_document_text(self.item, document.index),
                    score=document.score,
                    retrieval_round=1,
                )
                for document in self.retrieval_ranking[:top_k]
            )
            return replace(
                self,
                stage=RAGStage.CONTROL,
                retrieval_round=1,
                retrieval_k=top_k,
                evidence=evidence,
                **common,
            )
        if self.stage is RAGStage.CONTROL and action == CONTROL_CONTINUE:
            query = self.components.refiner.refine(self.item.question, self.evidence)
            seen = {document.document_index for document in self.evidence}
            ordered_indices = [
                *(
                    document.index
                    for document in self.retrieval_ranking
                    if document.index not in seen
                ),
                *(document.index for document in self.retrieval_ranking if document.index in seen),
            ]
            proxy = tuple(
                RankedDocument(index=index, score=float(len(ordered_indices) - position))
                for position, index in enumerate(ordered_indices)
            )
            evidence = tuple(
                Evidence(
                    document_index=document.index,
                    title=self.item.context_titles[document.index],
                    text=_document_text(self.item, document.index),
                    score=document.score,
                    retrieval_round=self.retrieval_round + 1,
                )
                for document in proxy[: int(self.retrieval_k)]
            )
            return replace(
                self,
                retrieval_round=self.retrieval_round + 1,
                active_queries=(query,),
                query_history=(*self.query_history, (query,)),
                retrieval_ranking=proxy,
                evidence=_merge_evidence(self.evidence, evidence),
                **common,
            )
        if self.stage is RAGStage.RERANKER:
            return replace(
                self,
                stage=RAGStage.TOP_M,
                reranker=action,
                ranked_evidence=self.evidence,
                **common,
            )
        if self.stage is RAGStage.TOP_M:
            return replace(self, stage=RAGStage.TERMINAL, **common)
        return self.child(action)

    def child(self, action: Any) -> "RAGWorkflowState":
        action = str(action)
        if action not in self.available_actions():
            raise ValueError(f"unavailable RAG action {action!r} at {self.stage.value}")
        edge_execution_cost = 0.0
        estimated_tokens = self.estimated_search_tokens(action)
        updates: dict[str, Any] = {}
        if self.stage is RAGStage.QUERY_STRATEGY:
            generated = self.components.generator.generate_query(
                action, self.item.question
            )
            edge_execution_cost = float(sum(len(_tokens(query)) for query in generated.queries))
            updates.update(
                stage=RAGStage.RETRIEVER,
                query_strategy=action,
                active_queries=generated.queries,
                query_history=(generated.queries,),
                token_usage=add_token_usage(
                    self.token_usage,
                    {
                        "generator_4b_input_tokens": generated.input_tokens,
                        "generator_4b_output_tokens": generated.output_tokens,
                    },
                ),
                qwen_generation_attempts=(
                    self.qwen_generation_attempts + generated.generation_attempts
                ),
                qwen_retry_count=self.qwen_retry_count + generated.retry_count,
                qwen_length_retry_count=(
                    self.qwen_length_retry_count + generated.length_retry_count
                ),
                qwen_fallback_count=self.qwen_fallback_count + generated.fallback_count,
            )
        elif self.stage is RAGStage.RETRIEVER:
            result = self.components.ragspace.rank(
                self.item, self.active_queries, action
            )
            edge_execution_cost = _retrieval_work_words(
                self.item, self.active_queries, action
            )
            updates.update(
                stage=RAGStage.SELECTION,
                retriever=action,
                retrieval_ranking=result.documents,
                token_usage=add_token_usage(self.token_usage, result.token_usage),
            )
        elif self.stage is RAGStage.SELECTION:
            if action == "selection.mmr":
                edge_execution_cost = float(
                    sum(
                        _document_words(self.item, index)
                        for index in range(len(self.item.context_titles))
                    )
                )
            updates.update(stage=RAGStage.RETRIEVAL_K, selection=action)
        elif self.stage is RAGStage.RETRIEVAL_K:
            top_k = _RETRIEVAL_K[action]
            evidence = self.components.retrieve(
                self.item,
                self.retrieval_ranking,
                selection=str(self.selection),
                top_k=top_k,
                retrieval_round=1,
            )
            updates.update(
                stage=RAGStage.CONTROL,
                retrieval_round=1,
                retrieval_k=top_k,
                evidence=evidence,
            )
        elif self.stage is RAGStage.CONTROL and action == CONTROL_CONTINUE:
            query = self.components.refiner.refine(self.item.question, self.evidence)
            result = self.components.ragspace.rank(
                self.item, (query,), str(self.retriever)
            )
            evidence = self.components.retrieve(
                self.item,
                result.documents,
                selection=str(self.selection),
                top_k=int(self.retrieval_k),
                retrieval_round=self.retrieval_round + 1,
            )
            edge_execution_cost = _retrieval_work_words(
                self.item, (query,), str(self.retriever)
            )
            updates.update(
                retrieval_round=self.retrieval_round + 1,
                active_queries=(query,),
                query_history=(*self.query_history, (query,)),
                retrieval_ranking=result.documents,
                evidence=_merge_evidence(self.evidence, evidence),
                token_usage=add_token_usage(self.token_usage, result.token_usage),
            )
        elif self.stage is RAGStage.CONTROL:
            updates["stage"] = RAGStage.RERANKER
        elif self.stage is RAGStage.RERANKER:
            ranked, reranker_token_usage = self.components.rerank(
                self.item, self.item.question, self.evidence, action
            )
            if action == RERANK_QWEN:
                edge_execution_cost = float(
                    sum(_document_words(self.item, item.document_index) for item in self.evidence)
                )
            updates.update(
                stage=RAGStage.TOP_M,
                reranker=action,
                ranked_evidence=ranked,
                token_usage=add_token_usage(
                    self.token_usage, reranker_token_usage
                ),
            )
        elif self.stage is RAGStage.TOP_M:
            selected = self.ranked_evidence[: _TOP_M[action]]
            generated = self.components.generator.generate(
                self.item.question, selected
            )
            prediction = generated.text
            chosen_titles = {item.title for item in selected}
            supporting = set(self.item.supporting_titles)
            edge_execution_cost = float(
                sum(_document_words(self.item, item.document_index) for item in selected)
            )
            updates.update(
                stage=RAGStage.TERMINAL,
                ranked_evidence=selected,
                terminal_prediction=prediction,
                terminal_performance=_answer_f1(prediction, self.item.answer),
                terminal_support_recall=(
                    len(chosen_titles & supporting) / max(1, len(supporting))
                ),
                token_usage=add_token_usage(
                    self.token_usage,
                    {
                        "generator_4b_input_tokens": generated.input_tokens,
                        "generator_4b_output_tokens": generated.output_tokens,
                    },
                ),
                qwen_generation_attempts=(
                    self.qwen_generation_attempts + generated.generation_attempts
                ),
                qwen_retry_count=self.qwen_retry_count + generated.retry_count,
                qwen_length_retry_count=(
                    self.qwen_length_retry_count + generated.length_retry_count
                ),
                qwen_fallback_count=self.qwen_fallback_count + generated.fallback_count,
            )
        return replace(
            self,
            path=(*self.path, action),
            execution_cost_words=self.execution_cost_words + edge_execution_cost,
            estimated_qwen_token_edge_costs=(
                *self.estimated_qwen_token_edge_costs,
                estimated_tokens,
            ),
            **updates,
        )

    def path_costs(self) -> tuple[float, ...]:
        """Per-edge generative Qwen-token estimates; not workflow utility cost."""

        return self.estimated_qwen_token_edge_costs

    def outcome(self) -> WorkflowOutcome:
        if not self.is_terminal():
            raise ValueError("workflow outcome requires a terminal state")
        if (
            self.terminal_prediction is None
            or self.terminal_performance is None
            or self.terminal_support_recall is None
        ):
            raise ValueError("terminal RAG state is not materialized")
        normalized_cost = self.execution_cost_words / self.execution_cost_reference
        utility = self.terminal_performance - self.workflow_cost_lambda * normalized_cost
        return WorkflowOutcome(
            performance=self.terminal_performance,
            execution_cost_words=self.execution_cost_words,
            normalized_execution_cost=normalized_cost,
            utility=utility,
            prediction=self.terminal_prediction,
            selected_document_indices=tuple(
                item.document_index for item in self.ranked_evidence
            ),
            support_recall=self.terminal_support_recall,
            token_usage=normalize_token_usage(self.token_usage),
            qwen_generation_attempts=self.qwen_generation_attempts,
            qwen_retry_count=self.qwen_retry_count,
            qwen_length_retry_count=self.qwen_length_retry_count,
            qwen_fallback_count=self.qwen_fallback_count,
        )


__all__ = [
    "BridgeQueryRefiner",
    "CONTROL_CONTINUE",
    "CONTROL_STOP",
    "Evidence",
    "Generator",
    "JSONTemplateGenerator",
    "QUERY_ACTIONS",
    "RAGComponents",
    "RAGStage",
    "RAGWorkflowState",
    "WorkflowOutcome",
]
