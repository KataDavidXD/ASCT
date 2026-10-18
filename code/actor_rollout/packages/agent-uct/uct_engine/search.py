"""Uniform, UCT, and cost-aware AgentUCT over one QID-scoped WTB cache."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Hashable

from .core import SearchState, TreeNode
from .tokens import (
    add_token_usage,
    empty_token_usage,
    generator_token_count,
    normalize_token_usage,
    qwen_input_token_count,
)
from .wtb import QIDPrefixCache

SEARCH_METHODS = ("uniform", "uct", "agentuct")
SEARCH_COST_MODES = ("agentuct-Marginal", "agentuct-Bellman")
DEFAULT_SEARCH_COST_MODE = "agentuct-Bellman"


@dataclass(frozen=True)
class SearchResult:
    actions: tuple[Hashable, ...]
    q_values: tuple[float, ...]
    q_performance: tuple[float, ...]
    q_normalized_execution_costs: tuple[float, ...]
    visit_counts: tuple[int, ...]
    search_budget: int
    search_cost_mode: str
    token_beta: float
    initial_estimated_search_tokens: tuple[float, ...]
    best_terminal_path: tuple[Hashable, ...]
    best_terminal_performance: float
    best_terminal_normalized_execution_cost: float
    best_terminal_utility: float
    token_usage: dict[str, int]
    qwen_input_tokens: int
    qwen_output_tokens: int
    qwen_tokens: int
    qwen_generation_attempts: int
    qwen_retry_count: int
    qwen_length_retry_count: int
    qwen_fallback_count: int
    successful_qwen_generations: int
    cumulative_evaluation_utility: float
    mean_evaluation_utility: float
    trace: tuple[dict[str, object], ...]
    cache: dict[str, object]


def _terminal_previews(state: SearchState) -> tuple[SearchState, ...]:
    if state.is_terminal():
        return (state,)
    output: list[SearchState] = []
    for action in state.available_actions():
        output.extend(_terminal_previews(state.preview_child(action)))
    return tuple(output)


def _missing_tokens(
    state: SearchState,
    cache: QIDPrefixCache,
) -> float:
    return cache.missing_cost(state.qid, state.path, state.path_costs())


def _immediate_token_estimate(
    state: SearchState,
    action: Hashable,
    cache: QIDPrefixCache,
) -> float:
    child = state.preview_child(action)
    return max(0.0, _missing_tokens(child, cache) - _missing_tokens(state, cache))


def _bellman_continuation_token_estimate(
    state: SearchState,
    action: Hashable,
    cache: QIDPrefixCache,
) -> float:
    base = _missing_tokens(state, cache)
    terminals = _terminal_previews(state.preview_child(action))
    return math.fsum(
        max(0.0, _missing_tokens(terminal, cache) - base)
        for terminal in terminals
    ) / len(terminals)


def _search_token_estimate(
    state: SearchState,
    action: Hashable,
    *,
    search_cost_mode: str,
    cache: QIDPrefixCache,
) -> float:
    if search_cost_mode == "agentuct-Marginal":
        return _immediate_token_estimate(state, action, cache)
    if search_cost_mode == "agentuct-Bellman":
        return _bellman_continuation_token_estimate(state, action, cache)
    raise ValueError(f"unknown search cost mode: {search_cost_mode}")


def _choose_child(
    node: TreeNode,
    *,
    method: str,
    token_beta: float,
    exploration_constant: float,
    search_cost_mode: str,
    cache: QIDPrefixCache,
) -> tuple[TreeNode, dict[str, object]]:
    selected: TreeNode | None = None
    selected_score = float("-inf")
    selected_cost = 0.0
    selected_exploration = 0.0
    selected_base_score = float("-inf")
    utility_only: TreeNode | None = None
    utility_only_score = float("-inf")
    minimum_cost = float("inf")
    maximum_cost = float("-inf")
    for action in node.state.available_actions():
        child = node.children[action]
        exploration = exploration_constant * math.sqrt(
            math.log(node.visits + 1.0) / child.visits
        )
        if method == "uniform":
            base_score = -float(child.visits)
            cost = 0.0
            score = -float(child.visits)
        else:
            base_score = child.q + exploration
            cost = (
                _search_token_estimate(
                    node.state,
                    action,
                    search_cost_mode=search_cost_mode,
                    cache=cache,
                )
                if method == "agentuct"
                else 0.0
            )
            score = base_score - token_beta * cost
        minimum_cost = min(minimum_cost, cost)
        maximum_cost = max(maximum_cost, cost)
        if base_score > utility_only_score:
            utility_only = child
            utility_only_score = base_score
        if score > selected_score:
            selected = child
            selected_score = score
            selected_cost = cost
            selected_exploration = exploration
            selected_base_score = base_score
    if selected is None:
        raise RuntimeError("fully expanded search node has no child")
    return selected, {
        "kind": "selection",
        "path": list(node.state.path),
        "selected_action": selected.action_from_parent,
        "utility_only_action": (
            None if utility_only is None else utility_only.action_from_parent
        ),
        "selected_q_utility": selected.q,
        "selected_exploration": selected_exploration,
        "selected_estimated_tokens": selected_cost,
        "minimum_estimated_tokens": minimum_cost,
        "maximum_estimated_tokens": maximum_cost,
        "selected_utility_only_score": selected_base_score,
        "best_utility_only_score": utility_only_score,
        "net_score": selected_score,
        "cost_active": (
            method == "agentuct"
            and token_beta > 0
            and maximum_cost > minimum_cost
        ),
        "cost_changed_choice": selected is not utility_only,
    }


def _sample_weighted(weights: tuple[float, ...], rng: random.Random) -> int:
    total = math.fsum(weights)
    threshold = rng.random() * total
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if threshold <= cumulative:
            return index
    return len(weights) - 1


def _rollout_action(
    state: SearchState,
    *,
    method: str,
    token_beta: float,
    search_cost_mode: str,
    rng: random.Random,
    cache: QIDPrefixCache,
    actions: tuple[Hashable, ...] | None = None,
) -> tuple[Hashable, dict[str, object]]:
    candidates = state.available_actions() if actions is None else actions
    if method == "agentuct":
        costs = tuple(
            _search_token_estimate(
                state,
                action,
                search_cost_mode=search_cost_mode,
                cache=cache,
            )
            for action in candidates
        )
        penalties = tuple(token_beta * cost for cost in costs)
        minimum = min(penalties)
        weights = tuple(math.exp(-(penalty - minimum)) for penalty in penalties)
        index = _sample_weighted(weights, rng)
        probabilities = tuple(weight / math.fsum(weights) for weight in weights)
        policy = "cost_aware"
    else:
        costs = tuple(0.0 for _ in candidates)
        index = rng.randrange(len(candidates))
        probabilities = tuple(1.0 / len(candidates) for _ in candidates)
        policy = "uniform"
    return candidates[index], {
        "path": list(state.path),
        "selected_action": candidates[index],
        "policy": policy,
        "selected_probability": probabilities[index],
        "selected_estimated_tokens": costs[index],
        "minimum_estimated_tokens": min(costs),
        "maximum_estimated_tokens": max(costs),
        "cost_active": (
            method == "agentuct"
            and token_beta > 0
            and max(costs) > min(costs)
        ),
    }


def run_search(
    root_state: SearchState,
    *,
    method: str,
    search_budget: int,
    token_beta: float,
    rng: random.Random,
    cache: QIDPrefixCache,
    search_cost_mode: str = DEFAULT_SEARCH_COST_MODE,
    exploration_constant: float = 1.4,
) -> SearchResult:
    """Run one search after the actor action has already been sampled.

    ``search_budget`` is the total number of terminal evaluations. Every root
    action receives one rollout before the remaining budget is allocated.
    """

    if method not in SEARCH_METHODS:
        raise ValueError(f"search method must be one of {SEARCH_METHODS}")
    if root_state.is_terminal():
        raise ValueError("search root must be non-terminal")
    if not root_state.qid:
        raise ValueError("search root qid must be non-empty")
    if not math.isfinite(token_beta) or token_beta < 0:
        raise ValueError("token_beta must be finite and non-negative")
    if search_cost_mode not in SEARCH_COST_MODES:
        raise ValueError(f"search_cost_mode must be one of {SEARCH_COST_MODES}")

    root_actions = root_state.available_actions()
    if not root_actions:
        raise ValueError("search root has no legal actions")
    if search_budget < len(root_actions):
        raise ValueError("search_budget must cover every legal root action")
    initial_token_estimates = (
        tuple(
            _search_token_estimate(
                root_state,
                action,
                search_cost_mode=search_cost_mode,
                cache=cache,
            )
            for action in root_actions
        )
        if method == "agentuct"
        else tuple(0.0 for _ in root_actions)
    )
    root = TreeNode(root_state)
    trace: list[dict[str, object]] = []
    best: tuple[float, tuple[Hashable, ...], float, float] | None = None
    cumulative_qwen_input_tokens = 0
    cumulative_qwen_output_tokens = 0
    cumulative_token_usage = empty_token_usage()
    cumulative_qwen_generation_attempts = 0
    cumulative_qwen_retry_count = 0
    cumulative_qwen_length_retry_count = 0
    cumulative_qwen_fallback_count = 0
    successful_qwen_generations = 0
    cumulative_evaluation_utility = 0.0
    iterations = search_budget
    for iteration in range(1, iterations + 1):
        node = root
        tree_decisions: list[dict[str, object]] = []
        rollout_decisions: list[dict[str, object]] = []
        while not node.state.is_terminal() and not node.unexpanded_actions():
            node, decision = _choose_child(
                node,
                method=method,
                token_beta=token_beta,
                exploration_constant=exploration_constant,
                search_cost_mode=search_cost_mode,
                cache=cache,
            )
            tree_decisions.append(decision)

        selection_depth = len(node.state.path)

        expanded = False
        unexpanded = node.unexpanded_actions()
        if not node.state.is_terminal() and unexpanded:
            expanded_action, expansion_decision = _rollout_action(
                node.state,
                method=method,
                token_beta=token_beta,
                search_cost_mode=search_cost_mode,
                rng=rng,
                cache=cache,
                actions=unexpanded,
            )
            expansion_decision["kind"] = "expansion"
            tree_decisions.append(expansion_decision)
            node = node.add_child(
                expanded_action,
                node.state.preview_child(expanded_action),
            )
            expanded = True

        terminal = node.state
        while not terminal.is_terminal():
            action, rollout_decision = _rollout_action(
                terminal,
                method=method,
                token_beta=token_beta,
                search_cost_mode=search_cost_mode,
                rng=rng,
                cache=cache,
            )
            rollout_decision["kind"] = "rollout"
            rollout_decisions.append(rollout_decision)
            terminal = terminal.preview_child(action)

        evaluated = cache.evaluate(
            qid=terminal.qid,
            path=terminal.path,
            path_costs=terminal.path_costs(),
        )
        current: TreeNode | None = node
        while current is not None:
            current.update(
                utility=evaluated.utility,
                performance=evaluated.performance,
                execution_cost=evaluated.normalized_execution_cost,
            )
            current = current.parent
        candidate_best = (
            evaluated.utility,
            terminal.path,
            evaluated.performance,
            evaluated.normalized_execution_cost,
        )
        if best is None or candidate_best[0] > best[0]:
            best = candidate_best
        cumulative_token_usage = add_token_usage(
            cumulative_token_usage, evaluated.token_usage
        )
        cumulative_qwen_input_tokens = qwen_input_token_count(
            cumulative_token_usage
        )
        cumulative_qwen_output_tokens = cumulative_token_usage[
            "generator_4b_output_tokens"
        ]
        cumulative_qwen_generation_attempts += evaluated.qwen_generation_attempts
        cumulative_qwen_retry_count += evaluated.qwen_retry_count
        cumulative_qwen_length_retry_count += evaluated.qwen_length_retry_count
        cumulative_qwen_fallback_count += evaluated.qwen_fallback_count
        if (
            generator_token_count(evaluated.token_usage) > 0
            and evaluated.qwen_fallback_count == 0
        ):
            successful_qwen_generations += 1
        cumulative_evaluation_utility += evaluated.utility
        cumulative_qwen_tokens = (
            cumulative_qwen_input_tokens + cumulative_qwen_output_tokens
        )

        trace.append(
            {
                "iteration": iteration,
                "selection_depth": selection_depth,
                "expanded": expanded,
                "terminal_path": list(terminal.path),
                "performance": evaluated.performance,
                "normalized_execution_cost": evaluated.normalized_execution_cost,
                "utility": evaluated.utility,
                "wtb_mode": evaluated.mode,
                "wtb_reuse_depth": evaluated.reuse_depth,
                "estimated_search_tokens": evaluated.estimated_search_tokens,
                "qwen_input_tokens": evaluated.qwen_input_tokens,
                "qwen_output_tokens": evaluated.qwen_output_tokens,
                "token_usage": normalize_token_usage(evaluated.token_usage),
                **normalize_token_usage(evaluated.token_usage),
                "qwen_generation_attempts": evaluated.qwen_generation_attempts,
                "qwen_tokens": (
                    evaluated.qwen_input_tokens + evaluated.qwen_output_tokens
                ),
                "qwen_retry_count": evaluated.qwen_retry_count,
                "qwen_length_retry_count": evaluated.qwen_length_retry_count,
                "qwen_fallback_count": evaluated.qwen_fallback_count,
                "cumulative_qwen_tokens": cumulative_qwen_tokens,
                "cumulative_token_usage": normalize_token_usage(
                    cumulative_token_usage
                ),
                "cumulative_qwen_generation_attempts": (
                    cumulative_qwen_generation_attempts
                ),
                "cumulative_qwen_retry_count": cumulative_qwen_retry_count,
                "cumulative_qwen_length_retry_count": (
                    cumulative_qwen_length_retry_count
                ),
                "cumulative_qwen_fallback_count": cumulative_qwen_fallback_count,
                "cumulative_evaluation_utility": cumulative_evaluation_utility,
                "best_utility_so_far": best[0],
                "best_performance_so_far": best[2],
                "best_normalized_execution_cost_so_far": best[3],
                "tree_decisions": tree_decisions,
                "rollout_decisions": rollout_decisions,
            }
        )

    if set(root.children) != set(root_actions):
        raise RuntimeError("search did not cover every legal root action")
    if best is None:
        raise RuntimeError("search did not evaluate a terminal workflow")
    return SearchResult(
        actions=root_actions,
        q_values=tuple(root.children[action].q for action in root_actions),
        q_performance=tuple(
            root.children[action].q_performance for action in root_actions
        ),
        q_normalized_execution_costs=tuple(
            root.children[action].q_execution_cost for action in root_actions
        ),
        visit_counts=tuple(root.children[action].visits for action in root_actions),
        search_budget=search_budget,
        search_cost_mode=search_cost_mode,
        token_beta=token_beta,
        initial_estimated_search_tokens=initial_token_estimates,
        best_terminal_path=best[1],
        best_terminal_performance=best[2],
        best_terminal_normalized_execution_cost=best[3],
        best_terminal_utility=best[0],
        token_usage=normalize_token_usage(cumulative_token_usage),
        qwen_input_tokens=cumulative_qwen_input_tokens,
        qwen_output_tokens=cumulative_qwen_output_tokens,
        qwen_tokens=(
            cumulative_qwen_input_tokens + cumulative_qwen_output_tokens
        ),
        qwen_generation_attempts=cumulative_qwen_generation_attempts,
        qwen_retry_count=cumulative_qwen_retry_count,
        qwen_length_retry_count=cumulative_qwen_length_retry_count,
        qwen_fallback_count=cumulative_qwen_fallback_count,
        successful_qwen_generations=successful_qwen_generations,
        cumulative_evaluation_utility=cumulative_evaluation_utility,
        mean_evaluation_utility=cumulative_evaluation_utility / search_budget,
        trace=tuple(trace),
        cache=cache.snapshot(root_state.qid),
    )


__all__ = [
    "DEFAULT_SEARCH_COST_MODE",
    "SEARCH_COST_MODES",
    "SEARCH_METHODS",
    "SearchResult",
    "run_search",
]
