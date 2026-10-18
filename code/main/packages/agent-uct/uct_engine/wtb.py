"""WTB checkpoint reuse indexed only by workflow path prefix and QID."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, TypedDict

from .tokens import (
    empty_token_usage,
    normalize_token_usage,
    qwen_input_token_count,
    subtract_token_usage,
)

QUERY_ACTIONS = ("query.identity", "query.query2doc", "query.decomposition")
RETRIEVER_ACTIONS = (
    "retriever.bm25",
    "retriever.dense_e5",
    "retriever.bge_m3_hybrid",
    "retriever.qwen3_embedding",
    "retriever.rrf",
)
SELECTION_ACTIONS = ("selection.relevance", "selection.mmr")
RETRIEVE_ACTIONS = ("retrieve.k=3", "retrieve.k=6")
CONTROL_ACTIONS = ("control.stop", "control.continue")
RERANK_ACTIONS = ("reranker.identity", "reranker.qwen3")
TOP_M_ACTIONS = ("top_m=2", "top_m=4")
_END = "_end"
_MAX_STEPS = 9


class _WorkflowState(TypedDict, total=False):
    qid: str
    path: list[str]
    rag_state: dict[str, Any]
    config_key: list[str]
    _variant_config: dict[str, str]


@dataclass(frozen=True)
class _PrefixEntry:
    execution_id: str
    checkpoint_id: str


@dataclass(frozen=True)
class CacheEvaluation:
    performance: float
    normalized_execution_cost: float
    utility: float
    mode: str
    reuse_depth: int
    estimated_search_tokens: float
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


class QIDPrefixCache:
    """Fork/resume native WTB states through `(path_prefix, qid)` edges."""

    def __init__(
        self,
        directory: str | Path,
        *,
        extra_retriever_actions: tuple[str, ...] = (),
    ) -> None:
        from langgraph.graph import END, StateGraph
        from wtb.sdk import WorkflowProject, WTBTestBench

        self._root: Any | None = None
        self._qid: str | None = None
        if len(set(extra_retriever_actions)) != len(extra_retriever_actions):
            raise ValueError("extra retriever actions must be unique")
        if set(extra_retriever_actions) & set(RETRIEVER_ACTIONS):
            raise ValueError("extra retriever actions must not replace defaults")
        self._retriever_actions = (*RETRIEVER_ACTIONS, *extra_retriever_actions)
        cache_dir = Path(directory)
        cache_dir.mkdir(parents=True, exist_ok=True)

        def apply_action(action: str):
            def apply(state: _WorkflowState) -> dict[str, object]:
                if action == _END:
                    return {
                        "qid": state["qid"],
                        "path": list(state["path"]),
                        "rag_state": dict(state["rag_state"]),
                    }
                root = self._active_root(str(state["qid"]))
                current = root.restore(dict(state["rag_state"]))
                if current.qid != state["qid"] or tuple(current.path) != tuple(
                    state["path"]
                ):
                    raise RuntimeError(
                        "WTB checkpoint RAG state differs from its QID/path"
                    )
                child = current.child(action)
                return {
                    "qid": child.qid,
                    "path": list(child.path),
                    "rag_state": child.snapshot(),
                }

            return apply

        def graph_factory():
            graph = StateGraph(_WorkflowState)
            stages = tuple(f"step_{index}" for index in range(_MAX_STEPS))
            for stage in stages:
                graph.add_node(stage, apply_action(_END))
            graph.set_entry_point(stages[0])
            for source, destination in zip(stages[:-1], stages[1:], strict=True):
                graph.add_edge(source, destination)
            graph.add_edge(stages[-1], END)
            return graph

        self._bench = WTBTestBench.create(
            mode="development", data_dir=str(cache_dir / "wtb")
        )
        self._project = WorkflowProject(
            name="atppo-qid-rag-workflow", graph_factory=graph_factory
        )
        actions = (
            *QUERY_ACTIONS,
            *self._retriever_actions,
            *SELECTION_ACTIONS,
            *RETRIEVE_ACTIONS,
            *CONTROL_ACTIONS,
            *RERANK_ACTIONS,
            *TOP_M_ACTIONS,
            _END,
        )
        for index in range(_MAX_STEPS):
            stage = f"step_{index}"
            for action in actions:
                self._project.register_variant(stage, action, apply_action(action))
        self._bench.register_project(self._project)
        self._entries: dict[tuple[tuple[Hashable, ...], str], _PrefixEntry] = {}
        self._stats = {"full_miss": 0, "partial_resume": 0, "full_hit": 0}

    def begin_qid(self, root_state: Any) -> None:
        """Bind one empty RAG root and discard only the previous QID's local index."""

        if not str(root_state.qid) or tuple(root_state.path):
            raise ValueError("WTB cache requires one QID's empty-path RAG root")
        if not callable(getattr(root_state, "snapshot", None)) or not callable(
            getattr(root_state, "restore", None)
        ):
            raise TypeError("RAG root must provide snapshot() and restore()")
        self._root = root_state
        self._qid = str(root_state.qid)
        self._entries.clear()
        self._stats = {"full_miss": 0, "partial_resume": 0, "full_hit": 0}

    def _active_root(self, qid: str) -> Any:
        if self._root is None or self._qid is None:
            raise RuntimeError("WTB cache has no active QID; call begin_qid() first")
        if qid != self._qid:
            raise ValueError("WTB cache cannot access another QID")
        return self._root

    @staticmethod
    def _status(execution: Any) -> str:
        value = getattr(execution, "status", None)
        return str(getattr(value, "value", value)).lower()

    @staticmethod
    def _checkpoint_id(checkpoint: Any) -> str:
        return str(getattr(checkpoint, "id", ""))

    def _variant_config(self, path: tuple[Hashable, ...]) -> dict[str, str]:
        actions = [str(action) for action in path]
        if len(actions) < 7 or actions[0] not in QUERY_ACTIONS:
            raise ValueError("RAG path must begin with one query strategy")
        if actions[1] not in self._retriever_actions:
            raise ValueError("RAG path requires one retriever")
        if actions[2] not in SELECTION_ACTIONS:
            raise ValueError("RAG path requires relevance or MMR selection")
        if actions[3] not in RETRIEVE_ACTIONS:
            raise ValueError("RAG path requires retrieval k=3 or k=6")
        position = 4
        control_count = 0
        stopped = False
        while position < len(actions) and actions[position] in CONTROL_ACTIONS:
            control_count += 1
            stopped = actions[position] == "control.stop"
            position += 1
            if stopped:
                break
        if not stopped or not 1 <= control_count <= 3:
            raise ValueError("RAG path requires stop within three control decisions")
        if position >= len(actions) or actions[position] not in RERANK_ACTIONS:
            raise ValueError("RAG path requires a reranker after control")
        position += 1
        if position != len(actions) - 1 or actions[position] not in TOP_M_ACTIONS:
            raise ValueError("RAG path must end with one top_m action")
        if len(actions) > _MAX_STEPS:
            raise ValueError("RAG path exceeds the three-round workflow")
        padded = [*actions, *([_END] * (_MAX_STEPS - len(actions)))]
        return {f"step_{index}": action for index, action in enumerate(padded)}

    def _longest_prefix(self, qid: str, path: tuple[Hashable, ...]) -> int:
        depth = 0
        for index in range(1, len(path) + 1):
            if (path[:index], qid) not in self._entries:
                break
            depth = index
        return depth

    def _checkpoints(
        self,
        execution_id: str,
        *,
        qid: str,
        path: tuple[Hashable, ...],
    ) -> dict[int, Any]:
        selected: dict[int, Any] = {}
        for checkpoint in self._bench.get_checkpoints(execution_id):
            state = getattr(checkpoint, "state_values", None)
            if not isinstance(state, Mapping) or str(state.get("qid", "")) != qid:
                continue
            prefix = tuple(state.get("path", ()))
            depth = len(prefix)
            rag_state = state.get("rag_state")
            if (
                not 1 <= depth <= len(path)
                or prefix != path[:depth]
                or not isinstance(rag_state, Mapping)
            ):
                continue
            previous = selected.get(depth)
            if previous is None or int(checkpoint.step) > int(previous.step):
                selected[depth] = checkpoint
        return selected

    def _native_state(self, entry: _PrefixEntry, *, path: tuple[Hashable, ...]) -> Any:
        if self._qid is None:
            raise RuntimeError("WTB cache has no active QID")
        root = self._active_root(self._qid)
        checkpoint = next((
            item for item in self._bench.get_checkpoints(entry.execution_id)
            if self._checkpoint_id(item) == entry.checkpoint_id
        ), None)
        if checkpoint is None:
            raise RuntimeError("WTB native checkpoint for cached prefix is unavailable")
        values = checkpoint.state_values
        state = root.restore(dict(values["rag_state"]))
        if state.qid != self._qid or tuple(state.path) != path:
            raise RuntimeError("WTB restored state crossed its QID/path boundary")
        return state

    def missing_cost(
        self,
        qid: str,
        path: tuple[Hashable, ...],
        path_costs: tuple[float, ...],
    ) -> float:
        self._active_root(qid)
        if len(path) != len(path_costs):
            raise ValueError("one search evaluation cost is required per action")
        return math.fsum(path_costs[self._longest_prefix(qid, path) :])

    def materialized_state(
        self,
        qid: str,
        path: tuple[Hashable, ...],
    ) -> Any:
        """Return the exact RAG child state already stored by WTB."""

        self._active_root(qid)
        entry = self._entries.get((path, qid))
        if entry is None:
            raise KeyError("RAG prefix has not been materialized by search")
        return self._native_state(entry, path=path)

    def advance(self, state: Any, action: str) -> tuple[Any, bool]:
        """Execute one real edge through WTB, or restore its native checkpoint.

        Actor-driven continuations need observed queries/evidence before choosing
        the next action. Existing full-path tree evaluation remains unchanged.
        The bool indicates an already materialized edge (zero new model work).
        """
        root = self._active_root(state.qid)
        if action not in state.available_actions():
            raise ValueError("cannot advance an unavailable workflow action")
        path = (*state.path, action)
        if len(path) > _MAX_STEPS:
            raise ValueError("workflow prefix exceeds the supported horizon")
        entry = self._entries.get((path, state.qid))
        if entry is not None:
            self._stats["full_hit"] += 1
            return self._native_state(entry, path=path), True
        padded = [*path, *([_END] * (_MAX_STEPS - len(path)))]
        variants = {f"step_{i}": str(value) for i, value in enumerate(padded)}
        if not state.path:
            execution = self._bench.run(
                project=self._project.name,
                initial_state={
                    "qid": state.qid, "path": [], "rag_state": root.snapshot(),
                    "config_key": list(path), "_variant_config": variants,
                },
                variant_config=variants,
            )
            mode = "full_miss"
        else:
            source = self._entries.get((tuple(state.path), state.qid))
            if source is None:
                raise ValueError("advance requires a WTB-materialized parent")
            fork = self._bench.fork(
                source.execution_id, checkpoint_id=source.checkpoint_id,
                new_initial_state={"config_key": list(path), "_variant_config": variants},
            )
            execution = self._bench.resume(fork.fork_execution_id)
            mode = "partial_resume"
        if self._status(execution) != "completed":
            raise RuntimeError(f"WTB prefix execution failed: {self._status(execution)}")
        # Keep the checkpoint immediately after the real edge, before padded
        # no-op nodes. A subsequent fork must resume at the NEXT action slot.
        candidates = [
            item for item in self._bench.get_checkpoints(str(execution.id))
            if isinstance(getattr(item, "state_values", None), Mapping)
            and str(item.state_values.get("qid", "")) == state.qid
            and tuple(item.state_values.get("path", ())) == path
        ]
        checkpoint = min(candidates, key=lambda item: int(item.step), default=None)
        if checkpoint is None:
            raise RuntimeError("WTB checkpoint missing for the executed prefix")
        entry = _PrefixEntry(str(execution.id), self._checkpoint_id(checkpoint))
        self._entries[(path, state.qid)] = entry
        self._stats[mode] += 1
        return self._native_state(entry, path=path), False

    def evaluate(
        self,
        *,
        qid: str,
        path: tuple[Hashable, ...],
        path_costs: tuple[float, ...],
    ) -> CacheEvaluation:
        root = self._active_root(qid)
        variants = self._variant_config(path)
        if len(path_costs) != len(path) or any(
            not math.isfinite(cost) or cost < 0 for cost in path_costs
        ):
            raise ValueError("search evaluation costs must be finite and non-negative")

        reuse_depth = self._longest_prefix(qid, path)
        if reuse_depth == len(path):
            self._stats["full_hit"] += 1
            terminal = self._native_state(self._entries[(path, qid)], path=path)
            outcome = terminal.outcome()
            return CacheEvaluation(
                performance=outcome.performance,
                normalized_execution_cost=outcome.normalized_execution_cost,
                utility=outcome.utility,
                mode="full_hit",
                reuse_depth=reuse_depth,
                estimated_search_tokens=0.0,
                token_usage=empty_token_usage(),
                qwen_generation_attempts=0,
                qwen_retry_count=0,
                qwen_length_retry_count=0,
                qwen_fallback_count=0,
            )

        reused_token_usage = empty_token_usage()
        reused_qwen_generation_attempts = 0
        reused_qwen_retry_count = 0
        reused_qwen_length_retry_count = 0
        reused_qwen_fallback_count = 0
        if reuse_depth == 0:
            execution = self._bench.run(
                project=self._project.name,
                initial_state={
                    "qid": qid,
                    "path": [],
                    "rag_state": root.snapshot(),
                    "config_key": list(path),
                    "_variant_config": variants,
                },
                variant_config=variants,
            )
            mode = "full_miss"
        else:
            source = self._entries[(path[:reuse_depth], qid)]
            reused = self._native_state(source, path=path[:reuse_depth])
            reused_token_usage = normalize_token_usage(reused.token_usage)
            reused_qwen_generation_attempts = int(reused.qwen_generation_attempts)
            reused_qwen_retry_count = int(reused.qwen_retry_count)
            reused_qwen_length_retry_count = int(reused.qwen_length_retry_count)
            reused_qwen_fallback_count = int(reused.qwen_fallback_count)
            fork = self._bench.fork(
                source.execution_id,
                checkpoint_id=source.checkpoint_id,
                new_initial_state={
                    "config_key": list(path),
                    "_variant_config": variants,
                },
            )
            execution = self._bench.resume(fork.fork_execution_id)
            mode = "partial_resume"
        if self._status(execution) != "completed":
            raise RuntimeError(
                f"WTB execution failed with status {self._status(execution)}"
            )

        execution_id = str(execution.id)
        checkpoints = self._checkpoints(execution_id, qid=qid, path=path)
        for depth in range(reuse_depth + 1, len(path) + 1):
            checkpoint = checkpoints.get(depth)
            if checkpoint is None:
                raise RuntimeError(f"WTB checkpoint missing for prefix depth {depth}")
            self._entries[(path[:depth], qid)] = _PrefixEntry(
                execution_id=execution_id,
                checkpoint_id=self._checkpoint_id(checkpoint),
            )

        terminal = self._native_state(self._entries[(path, qid)], path=path)
        outcome = terminal.outcome()
        token_usage = subtract_token_usage(outcome.token_usage, reused_token_usage)
        qwen_generation_attempts = (
            int(outcome.qwen_generation_attempts)
            - reused_qwen_generation_attempts
        )
        qwen_retry_count = int(outcome.qwen_retry_count) - reused_qwen_retry_count
        qwen_length_retry_count = (
            int(outcome.qwen_length_retry_count) - reused_qwen_length_retry_count
        )
        qwen_fallback_count = (
            int(outcome.qwen_fallback_count) - reused_qwen_fallback_count
        )
        if any(
            value < 0
            for value in (
                *token_usage.values(),
                qwen_generation_attempts,
                qwen_retry_count,
                qwen_length_retry_count,
                qwen_fallback_count,
            )
        ):
            raise RuntimeError("WTB resumed Qwen counters moved backwards")
        self._stats[mode] += 1
        return CacheEvaluation(
            performance=outcome.performance,
            normalized_execution_cost=outcome.normalized_execution_cost,
            utility=outcome.utility,
            mode=mode,
            reuse_depth=reuse_depth,
            estimated_search_tokens=math.fsum(path_costs[reuse_depth:]),
            token_usage=token_usage,
            qwen_generation_attempts=qwen_generation_attempts,
            qwen_retry_count=qwen_retry_count,
            qwen_length_retry_count=qwen_length_retry_count,
            qwen_fallback_count=qwen_fallback_count,
        )

    def snapshot(self, qid: str) -> dict[str, object]:
        self._active_root(qid)
        return {"stats": dict(self._stats)}

    def close(self) -> None:
        self._bench.close()


__all__ = ["CacheEvaluation", "QIDPrefixCache"]
