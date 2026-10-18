"""Balanced root-action MC evaluation with frozen actor continuations."""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any

from uct_engine import add_token_usage, empty_token_usage, generator_token_count
from uct_engine.tokens import qwen_input_token_count

from .vine import COUNTERS, _advance, _sample

CF_ACTOR_SETTINGS = {
    "cf_actor_schema": "balanced_root_actor_mc_centered_wtb_v1",
    "cf_actor_budget_unit": "terminal_samples_per_visited_state_including_cache_hits",
    "cf_actor_root_allocation": "unvisited_seeded_then_least_visited_legal_order",
    "cf_actor_continuation": "frozen_pi_old_on_materialized_states",
    "cf_actor_credit": "q_chosen_minus_pi_old_weighted_q",
    "cf_actor_training_samples": "actor_main_trajectory_only",
}


def collect_cf_actor_trajectory(
    state: Any,
    policy: Any,
    *,
    rng: random.Random,
    search_seed: int,
    budget: int,
    cache: Any,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    """Collect a main actor path; auxiliary trials supply its action-value credit.

    The caller freezes the actor during collection. Each auxiliary trial forces
    one balanced root action, then samples the actor at every subsequent state.
    All observations are materialized through the same WTB edge interface as
    VinePPO. No tree-selection or cost-softmax rule operates below the root.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("CF-Actor requires a positive integer auxiliary budget")
    records: list[dict[str, Any]] = []
    # Memoize only frozen probabilities, never random draws or sample identities.
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
        if not actions or budget < len(actions):
            raise ValueError("CF-Actor budget must cover every legal root action")
        probabilities, logs, actor_tokens = distribution(state)
        chosen = _sample(probabilities, rng)
        seed_key = f"{search_seed}:{state.qid}:{len(state.path)}:cf-actor"
        root_rng = random.Random(seed_key + ":root")
        suffix_rng = random.Random(seed_key + ":continuation")
        counts = [0] * len(actions)
        per_action: list[list[dict[str, Any]]] = [[] for _ in actions]
        trace: list[dict[str, Any]] = []
        usage = empty_token_usage()
        counters = dict.fromkeys(COUNTERS, 0)
        modes: Counter[str] = Counter()

        for sample_index in range(budget):
            minimum = min(counts)
            candidates = [i for i, count in enumerate(counts) if count == minimum]
            arm = (
                candidates[root_rng.randrange(len(candidates))]
                if minimum == 0
                else candidates[0]
            )
            counts[arm] += 1
            terminal = state
            sample_usage = empty_token_usage()
            sample_counters = dict.fromkeys(COUNTERS, 0)
            decisions: list[dict[str, Any]] = []
            reused_prefix = len(state.path)
            saw_miss = False
            edges = hits = 0

            while not terminal.is_terminal():
                labels = list(terminal.available_actions())
                if edges == 0:
                    label, scoring_tokens = actions[arm], 0
                else:
                    probs, _, scoring_tokens = distribution(terminal)
                    label = labels[_sample(probs, suffix_rng)]
                    decisions.append(
                        {
                            "path": list(terminal.path),
                            "labels": labels,
                            "probabilities": probs,
                            "chosen_label": label,
                            "policy": "frozen_actor",
                            "materialized": True,
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
                edges += 1
                hits += int(hit)
                if hit and not saw_miss:
                    reused_prefix += 1
                saw_miss = saw_miss or not hit

            outcome = terminal.outcome()
            if not all(
                math.isfinite(value)
                for value in (
                    outcome.utility,
                    outcome.performance,
                    outcome.normalized_execution_cost,
                )
            ):
                raise ValueError("non-finite CF-Actor terminal outcome")
            mode = (
                "full_hit"
                if hits == edges
                else "partial_resume"
                if reused_prefix
                else "full_miss"
            )
            sample = {
                "sample": sample_index,
                "root_action_index": arm,
                "root_action": actions[arm],
                "root_allocation": "balanced",
                "terminal_path": list(terminal.path),
                "utility": outcome.utility,
                "performance": outcome.performance,
                "normalized_execution_cost": outcome.normalized_execution_cost,
                "wtb_mode": mode,
                "wtb_reuse_depth": reused_prefix,
                "token_usage": sample_usage,
                **sample_counters,
                "rollout_decisions": decisions,
            }
            per_action[arm].append(sample)
            trace.append(sample)
            modes[mode] += 1
            usage = add_token_usage(usage, sample_usage)
            for key in COUNTERS:
                counters[key] += sample_counters[key]

        def action_means(key: str, groups: list[list[dict[str, Any]]]) -> list[float]:
            return [
                math.fsum(sample[key] for sample in arm) / len(arm) for arm in groups
            ]

        q_values = action_means("utility", per_action)
        baseline = math.fsum(
            p * q for p, q in zip(probabilities, q_values, strict=True)
        )
        best = max(trace, key=lambda sample: sample["utility"])
        total_utility = math.fsum(sample["utility"] for sample in trace)
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
            "cf_actor_schema": CF_ACTOR_SETTINGS["cf_actor_schema"],
            "continuation_policy": "frozen_actor",
            "q_utility": q_values,
            "q_performance": action_means("performance", per_action),
            "q_normalized_execution_cost": action_means(
                "normalized_execution_cost", per_action
            ),
            "visit_counts": counts,
            "search_budget": budget,
            "q_raw_chosen": q_values[chosen],
            "v_tree_baseline": baseline,
            "advantage": q_values[chosen] - baseline,
            "best_searched_workflow": {
                "path": best["terminal_path"],
                "utility": best["utility"],
                "performance": best["performance"],
                "normalized_execution_cost": best["normalized_execution_cost"],
            },
            "search_trace": trace,
            "search_token_metrics": {
                "logical_evaluations": budget,
                "token_usage": usage,
                "qwen_input_tokens": input_tokens,
                "qwen_output_tokens": output_tokens,
                "qwen_tokens": input_tokens + output_tokens,
                **counters,
                "successful_qwen_generations": sum(
                    int(
                        generator_token_count(sample["token_usage"]) > 0
                        and sample["qwen_fallback_count"] == 0
                    )
                    for sample in trace
                ),
                "cumulative_evaluation_utility": total_utility,
                "mean_evaluation_utility": total_utility / budget,
                "final_best_utility": best["utility"],
                "wtb_modes": dict(modes),
                "rollout_decisions": sum(
                    len(sample["rollout_decisions"]) for sample in trace
                ),
            },
        }
        # Only this already independently sampled main action enters PPO.
        state, main_usage, main_counters, main_hit = _advance(
            cache, state, actions[chosen]
        )
        record["actor_execution"] = {
            "token_usage": main_usage,
            **main_counters,
            "cache_hit": main_hit,
        }
        records.append(record)
    return state, state.outcome(), records
