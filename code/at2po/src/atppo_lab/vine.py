"""Workflow-adapted VinePPO: actor continuations, MC state values, TD credit."""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any

from uct_engine import add_token_usage, empty_token_usage, generator_token_count
from uct_engine.tokens import qwen_input_token_count, subtract_token_usage

VINE_SETTINGS = {
    "vine_schema": "actor_mc_td0_wtb_v1",
    "vine_gamma": 1.0,
    "vine_budget_unit": "auxiliary_terminal_samples_per_visited_nonterminal_state",
}
COUNTERS = (
    "qwen_generation_attempts",
    "qwen_retry_count",
    "qwen_length_retry_count",
    "qwen_fallback_count",
)


def _sample(probabilities: list[float], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if threshold <= cumulative:
            return index
    return len(probabilities) - 1


def _advance(
    cache: Any, state: Any, action: str
) -> tuple[Any, dict[str, int], dict[str, int], bool]:
    child, hit = cache.advance(state, action)
    usage = (
        empty_token_usage()
        if hit
        else subtract_token_usage(child.token_usage, state.token_usage)
    )
    counters = {
        key: 0 if hit else int(getattr(child, key)) - int(getattr(state, key))
        for key in COUNTERS
    }
    if any(value < 0 for value in counters.values()):
        raise RuntimeError("WTB prefix counters moved backwards")
    return child, usage, counters, hit


def collect_vine_trajectory(
    state: Any,
    policy: Any,
    *,
    rng: random.Random,
    search_seed: int,
    budget: int,
    cache: Any,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    if budget <= 0:
        raise ValueError("VinePPO requires a positive auxiliary sample budget")
    records: list[dict[str, Any]] = []
    # One frozen policy per collection. Cache probabilities, never random draws;
    # duplicates remain independent samples with replacement. Cleared per QID.
    score_cache: dict[tuple[str, tuple[str, ...]], tuple[list[float], list[float]]] = {}

    def distribution(current: Any) -> tuple[list[float], list[float], int]:
        key = (current.prompt(), tuple(current.available_actions()))
        before = policy.actor_scoring_tokens
        if key not in score_cache:
            score_cache[key] = policy.distribution(key[0], list(key[1]))
        probabilities, logs = score_cache[key]
        return probabilities, logs, policy.actor_scoring_tokens - before

    while not state.is_terminal():
        actions = list(state.available_actions())
        probabilities, logs, actor_tokens = distribution(state)
        chosen = _sample(probabilities, rng)
        auxiliary_rng = random.Random(
            f"{search_seed}:{state.qid}:{len(state.path)}:vine"
        )
        trace = []
        usage = empty_token_usage()
        counters = dict.fromkeys(COUNTERS, 0)
        modes: Counter[str] = Counter()
        decision_count = 0
        for sample_index in range(budget):
            terminal = state
            sample_usage = empty_token_usage()
            sample_counters = dict.fromkeys(COUNTERS, 0)
            decisions = []
            hits = 0
            edges = 0
            reused_prefix = len(state.path)
            saw_miss = False
            while not terminal.is_terminal():
                labels = list(terminal.available_actions())
                probs, _, scoring_tokens = distribution(terminal)
                label = labels[_sample(probs, auxiliary_rng)]
                decisions.append(
                    {
                        "path": list(terminal.path),
                        "labels": labels,
                        "probabilities": probs,
                        "chosen_label": label,
                    }
                )
                terminal, edge_usage, edge_counters, hit = _advance(
                    cache, terminal, label
                )
                sample_usage = add_token_usage(
                    sample_usage, edge_usage, {"actor_scoring_tokens": scoring_tokens}
                )
                for key in COUNTERS:
                    sample_counters[key] += edge_counters[key]
                hits += int(hit)
                edges += 1
                if hit and not saw_miss:
                    reused_prefix += 1
                saw_miss = saw_miss or not hit
            outcome = terminal.outcome()
            if not math.isfinite(outcome.utility):
                raise ValueError("non-finite auxiliary utility")
            mode = (
                "full_hit"
                if hits == edges
                else "partial_resume"
                if reused_prefix
                else "full_miss"
            )
            modes[mode] += 1
            decision_count += edges
            usage = add_token_usage(usage, sample_usage)
            for key in COUNTERS:
                counters[key] += sample_counters[key]
            trace.append(
                {
                    "sample": sample_index,
                    "terminal_path": list(terminal.path),
                    "utility": outcome.utility,
                    "performance": outcome.performance,
                    "normalized_execution_cost": outcome.normalized_execution_cost,
                    "wtb_mode": mode,
                    "wtb_reuse_depth": reused_prefix,
                    "token_usage": sample_usage,
                    **sample_counters,
                    "decisions": decisions,
                }
            )
        total_utility = math.fsum(row["utility"] for row in trace)
        input_tokens = qwen_input_token_count(usage)
        output_tokens = usage["generator_4b_output_tokens"]
        record = {
            "qid": state.qid,
            "step": len(state.path),
            "stage": state.stage.value,
            "state_path": list(state.path),
            "prompt": state.prompt(),
            "labels": actions,
            "chosen_label": actions[chosen],
            "old_logprob": logs[chosen],
            "probabilities": probabilities,
            "actor_scoring_tokens": actor_tokens,
            "vine_value": total_utility / budget,
            "vine_samples": budget,
            "vine_trace": trace,
            "search_token_metrics": {
                "logical_evaluations": budget,
                "token_usage": usage,
                "qwen_input_tokens": input_tokens,
                "qwen_output_tokens": output_tokens,
                "qwen_tokens": input_tokens + output_tokens,
                **counters,
                "successful_qwen_generations": sum(
                    int(
                        generator_token_count(row["token_usage"]) > 0
                        and row["qwen_fallback_count"] == 0
                    )
                    for row in trace
                ),
                "cumulative_evaluation_utility": total_utility,
                "mean_evaluation_utility": total_utility / budget,
                "final_best_utility": max(row["utility"] for row in trace),
                "wtb_modes": dict(modes),
                "rollout_decisions": decision_count,
            },
        }
        # MC may never sample the actor's chosen arm. Execute that real edge
        # separately and account for its new work outside the auxiliary budget.
        state, main_usage, main_counters, main_hit = _advance(
            cache, state, actions[chosen]
        )
        record["actor_execution"] = {
            "token_usage": main_usage,
            **main_counters,
            "cache_hit": main_hit,
        }
        records.append(record)
    outcome = state.outcome()
    for i, record in enumerate(records):
        reward = outcome.utility if i == len(records) - 1 else 0.0
        successor = records[i + 1]["vine_value"] if i + 1 < len(records) else 0.0
        record.update(
            reward=reward,
            next_vine_value=successor,
            advantage=reward + successor - record["vine_value"],
            terminated=i == len(records) - 1,
        )
    return state, outcome, records
