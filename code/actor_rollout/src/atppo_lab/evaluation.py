"""Compute endpoint objectives from utility and realized online Qwen tokens."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Iterable, Iterator
from itertools import chain
from pathlib import Path
from typing import Any

from uct_engine import (
    SEARCH_QWEN_TOKEN_FIELDS,
    TOKEN_FIELDS,
    add_token_usage,
    empty_token_usage,
    legacy_token_usage,
    normalize_token_usage,
    online_search_raw_token_count,
    search_qwen_token_count,
)

DEFAULT_METHODS = ("agentuct_atppo", "uct_atppo", "uniform_atppo", "ppo")
DEFAULT_DEPLOYMENT_HORIZON = 100_000


def endj_static(
    *,
    endpoint_utility_sum: float,
    search_qwen_tokens: int,
    searches: int,
    beta: float,
) -> dict[str, Any]:
    """Compare MCTS methods at their fixed logical search budget."""

    if searches <= 0:
        return {
            "searches": 0,
            "mean_endpoint_utility": None,
            "search_qwen_tokens": search_qwen_tokens,
            "mean_search_qwen_tokens_per_search": None,
            "beta": beta,
            "value": None,
        }
    mean_utility = endpoint_utility_sum / searches
    mean_tokens = search_qwen_tokens / searches
    return {
        "searches": searches,
        "mean_endpoint_utility": mean_utility,
        "search_qwen_tokens": search_qwen_tokens,
        "mean_search_qwen_tokens_per_search": mean_tokens,
        "beta": beta,
        "value": mean_utility - beta * mean_tokens,
    }


def endj_test_search(
    *,
    test_mean_utility: float | None,
    train_search_qwen_tokens: int,
    deployment_horizon: int,
    beta: float,
) -> dict[str, Any]:
    """Compute U_test - beta * C_train-search / N."""

    if deployment_horizon <= 0:
        raise ValueError("deployment_horizon must be positive")
    amortized = train_search_qwen_tokens / deployment_horizon
    return {
        "test_mean_utility": test_mean_utility,
        "train_search_qwen_tokens": train_search_qwen_tokens,
        "deployment_horizon": deployment_horizon,
        "amortized_search_qwen_tokens_per_deployment": amortized,
        "beta": beta,
        "value": (
            None if test_mean_utility is None else test_mean_utility - beta * amortized
        ),
    }


def _generator_counts(value: dict[str, Any]) -> tuple[int, int, bool]:
    if "generator_4b_input_tokens" in value:
        return (
            int(value["generator_4b_input_tokens"]),
            int(value.get("generator_4b_output_tokens", 0)),
            True,
        )
    return (
        int(value.get("qwen_input_tokens", 0)),
        int(value.get("qwen_output_tokens", 0)),
        False,
    )


def _token_usage(value: dict[str, Any]) -> tuple[dict[str, int], bool]:
    if "token_usage" in value:
        return normalize_token_usage(value["token_usage"]), True
    if any(field in value for field in TOKEN_FIELDS):
        return normalize_token_usage(value), True
    inputs, outputs, _ = _generator_counts(value)
    return (
        legacy_token_usage(
            qwen_input_tokens=inputs,
            qwen_output_tokens=outputs,
        ),
        False,
    )


def observed_utility_token_diagnostics(
    search_traces: Iterable[Iterable[dict[str, Any]]], *, beta: float
) -> dict[str, Any]:
    """Summarize internal pulls without treating them as delivered endpoints.

    The AUC uses a right-endpoint step convention: the incumbent observed after
    a pull is weighted by the realized online-Qwen tokens spent on that pull.
    This avoids inventing a utility at token budget zero. A zero-token search has
    no token-axis area and therefore has an undefined AUC.
    """

    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and non-negative")
    logical_evaluations = 0
    qwen_tokens = 0
    utility_sum = 0.0
    area = 0.0
    auc_sum = 0.0
    auc_searches = 0
    searches = 0
    zero_cost_searches = 0
    for trace in search_traces:
        searches += 1
        trace = iter(trace)
        try:
            first_row = next(trace)
        except StopIteration as exc:
            raise ValueError(
                "search trace must contain at least one evaluation"
            ) from exc
        rows = chain((first_row,), trace)
        search_tokens = 0
        search_area = 0.0
        best_utility = float("-inf")
        for row in rows:
            utility = float(row["utility"])
            if not math.isfinite(utility):
                raise ValueError("search utility must be finite")
            usage, _ = _token_usage(row)
            pull_tokens = search_qwen_token_count(usage)
            search_tokens += pull_tokens
            qwen_tokens += pull_tokens
            logical_evaluations += 1
            utility_sum += utility
            best_utility = max(best_utility, utility)
            stored_best = row.get("best_utility_so_far")
            if stored_best is not None and not math.isclose(
                float(stored_best), best_utility, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError("stored best_utility_so_far is inconsistent")
            stored_cumulative = row.get("cumulative_qwen_tokens")
            if (
                stored_cumulative is not None
                and int(stored_cumulative) != search_tokens
            ):
                raise ValueError("stored cumulative_qwen_tokens is inconsistent")
            search_area += best_utility * pull_tokens
        area += search_area
        if search_tokens:
            auc_sum += search_area / search_tokens
            auc_searches += 1
        else:
            zero_cost_searches += 1
    return {
        "searches": searches,
        "logical_evaluations": logical_evaluations,
        "qwen_tokens": qwen_tokens,
        "utility_sum": utility_sum,
        "mean_utility": (
            utility_sum / logical_evaluations if logical_evaluations else None
        ),
        "mean_rollout_utility": (
            utility_sum / logical_evaluations if logical_evaluations else None
        ),
        "cumulative_rollout_net_utility": utility_sum - beta * qwen_tokens,
        "mean_rollout_net_utility": (
            (utility_sum - beta * qwen_tokens) / logical_evaluations
            if logical_evaluations
            else None
        ),
        "observed_utility_token_area": area,
        "observed_utility_token_auc": area / qwen_tokens if qwen_tokens else None,
        "mean_search_observed_utility_token_auc": (
            auc_sum / auc_searches if auc_searches else None
        ),
        "observed_utility_token_auc_searches": auc_searches,
        "zero_qwen_token_searches": zero_cost_searches,
        "auc_convention": "right_endpoint_incumbent_weighted_by_pull_qwen_tokens",
    }


def _trajectory_search_traces(
    cell_dir: Path,
) -> Iterator[list[dict[str, Any]]]:
    path = cell_dir / "trajectories.jsonl"
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            trajectory = json.loads(line)
            for step in trajectory["steps"]:
                trace = step.get("search_trace", ())
                if trace:
                    yield list(trace)


def _cell_ledger(
    output_dir: Path,
    *,
    method: str,
    seed: int,
    deployment_horizon: int,
) -> dict[str, Any]:
    cell_dir = output_dir / f"{method}-seed-{seed}"
    result_path = cell_dir / "run_result.json"
    if not result_path.exists():
        result_path = cell_dir / "metrics.json"
    if not result_path.exists():
        raise FileNotFoundError(f"completed result is missing: {cell_dir}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "complete":
        raise ValueError(f"cell is not complete: {cell_dir}")
    beta = float(result["token_beta"])

    searches = 0
    evaluations = 0
    endpoint_utility = 0.0
    evaluation_utility = 0.0
    generator_input_tokens = 0
    generator_output_tokens = 0
    actor_scoring_tokens = 0
    detailed_sources: list[bool] = []
    train_token_usage = empty_token_usage()
    successful_generations = 0
    generation_attempts = 0
    retry_count = 0
    length_retry_count = 0
    fallback_count = 0
    tree_decisions = 0
    tree_cost_active_decisions = 0
    tree_cost_changed_choices = 0
    rollout_decisions = 0
    rollout_cost_active_decisions = 0
    modes: Counter[str] = Counter()

    summaries = [
        iteration.get("search_tokens", {}) for iteration in result.get("iterations", ())
    ]
    summary_ledger = bool(summaries) and all(
        summary.get("cumulative_evaluation_utility") is not None
        and summary.get("wtb_modes") is not None
        for summary in summaries
    )
    if summary_ledger:
        searches = sum(int(summary["searches"]) for summary in summaries)
        evaluations = sum(int(summary["logical_evaluations"]) for summary in summaries)
        endpoint_utility = sum(
            float(summary["cumulative_endpoint_utility"]) for summary in summaries
        )
        evaluation_utility = sum(
            float(summary["cumulative_evaluation_utility"]) for summary in summaries
        )
        for summary in summaries:
            inputs, outputs, detailed = _generator_counts(summary)
            generator_input_tokens += inputs
            generator_output_tokens += outputs
            detailed_sources.append(detailed)
            actor_scoring_tokens += int(summary.get("actor_scoring_tokens", 0))
            usage, usage_detailed = _token_usage(summary)
            train_token_usage = add_token_usage(train_token_usage, usage)
            detailed_sources.append(usage_detailed)
            modes.update(summary["wtb_modes"])
        successful_generations = sum(
            int(summary.get("successful_qwen_generations", 0)) for summary in summaries
        )
        generation_attempts = sum(
            int(summary.get("qwen_generation_attempts", 0)) for summary in summaries
        )
        retry_count = sum(
            int(summary.get("qwen_retry_count", 0)) for summary in summaries
        )
        length_retry_count = sum(
            int(summary.get("qwen_length_retry_count", 0)) for summary in summaries
        )
        fallback_count = sum(
            int(summary.get("qwen_fallback_count", 0)) for summary in summaries
        )
        tree_decisions = sum(
            int(summary.get("tree_decisions", 0)) for summary in summaries
        )
        tree_cost_active_decisions = sum(
            int(summary.get("tree_cost_active_decisions", 0)) for summary in summaries
        )
        tree_cost_changed_choices = sum(
            int(summary.get("tree_cost_changed_choices", 0)) for summary in summaries
        )
        rollout_decisions = sum(
            int(summary.get("rollout_decisions", 0)) for summary in summaries
        )
        rollout_cost_active_decisions = sum(
            int(summary.get("rollout_cost_active_decisions", 0))
            for summary in summaries
        )
        ledger_source = "iteration_summaries"
    else:
        trajectory_path = cell_dir / "trajectories.jsonl"
        if method != "ppo" and not trajectory_path.exists():
            raise FileNotFoundError(f"search trajectory is missing: {trajectory_path}")
        ledger_source = "trajectories"
        if trajectory_path.exists():
            with trajectory_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    trajectory = json.loads(line)
                    trajectory_actor_tokens = sum(
                        int(step.get("actor_scoring_tokens", 0))
                        for step in trajectory["steps"]
                    )
                    actor_scoring_tokens += trajectory_actor_tokens
                    train_token_usage = add_token_usage(
                        train_token_usage,
                        {"actor_scoring_tokens": trajectory_actor_tokens},
                    )
                    for step in trajectory["steps"]:
                        trace = step.get("search_trace", ())
                        if not trace:
                            continue
                        searches += 1
                        endpoint_utility += float(
                            step["search_token_metrics"]["final_best_utility"]
                        )
                        for row in trace:
                            evaluations += 1
                            evaluation_utility += float(row["utility"])
                            inputs, outputs, detailed = _generator_counts(row)
                            generator_input_tokens += inputs
                            generator_output_tokens += outputs
                            detailed_sources.append(detailed)
                            usage, usage_detailed = _token_usage(row)
                            train_token_usage = add_token_usage(
                                train_token_usage, usage
                            )
                            detailed_sources.append(usage_detailed)
                            generation_attempts += int(
                                row.get("qwen_generation_attempts", 0)
                            )
                            retry_count += int(row.get("qwen_retry_count", 0))
                            length_retry_count += int(
                                row.get("qwen_length_retry_count", 0)
                            )
                            fallback_count += int(row.get("qwen_fallback_count", 0))
                            successful_generations += int(
                                inputs + outputs > 0
                                and int(row.get("qwen_fallback_count", 0)) == 0
                            )
                            tree_rows = row.get("tree_decisions", ())
                            rollout_rows = row.get("rollout_decisions", ())
                            tree_decisions += len(tree_rows)
                            tree_cost_active_decisions += sum(
                                int(bool(decision.get("cost_active", False)))
                                for decision in tree_rows
                            )
                            tree_cost_changed_choices += sum(
                                int(bool(decision.get("cost_changed_choice", False)))
                                for decision in tree_rows
                            )
                            rollout_decisions += len(rollout_rows)
                            rollout_cost_active_decisions += sum(
                                int(bool(decision.get("cost_active", False)))
                                for decision in rollout_rows
                            )
                            modes[str(row["wtb_mode"])] += 1

    optimizer_actor_scoring_tokens = sum(
        int((iteration.get("optimizer") or {}).get("actor_scoring_tokens", 0))
        for iteration in result.get("iterations", ())
    )
    train_token_usage = add_token_usage(
        train_token_usage,
        {"actor_scoring_tokens": optimizer_actor_scoring_tokens},
    )
    test = result.get("test") or {}
    test_inputs, test_outputs, test_detailed = _generator_counts(test)
    test_token_usage, test_usage_detailed = _token_usage(test)
    test_mean_utility = (
        None if test.get("mean_utility") is None else float(test["mean_utility"])
    )
    generator_tokens = generator_input_tokens + generator_output_tokens
    search_qwen_tokens = search_qwen_token_count(train_token_usage)
    trajectory_diagnostics = observed_utility_token_diagnostics(
        _trajectory_search_traces(cell_dir), beta=beta
    )
    if trajectory_diagnostics["searches"]:
        rollout_diagnostics = trajectory_diagnostics
        summary_checks = {
            "summary_search_count_matches": (
                int(rollout_diagnostics["searches"]) == searches
            ),
            "summary_evaluation_count_matches": (
                int(rollout_diagnostics["logical_evaluations"]) == evaluations
            ),
            "summary_qwen_tokens_match": (
                int(rollout_diagnostics["qwen_tokens"]) == search_qwen_tokens
            ),
            "summary_utility_sum_matches": math.isclose(
                float(rollout_diagnostics["utility_sum"]),
                evaluation_utility,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ),
        }
        failed_checks = [name for name, passed in summary_checks.items() if not passed]
        if failed_checks:
            raise ValueError(
                f"trajectory and summary ledgers disagree for {cell_dir}: "
                + ", ".join(failed_checks)
            )
        rollout_diagnostics.update(
            source="trajectories",
            **summary_checks,
            not_a_deployment_objective=True,
        )
    else:
        rollout_diagnostics = {
            "source": "summary_without_observed_curve",
            "searches": searches,
            "logical_evaluations": evaluations,
            "qwen_tokens": search_qwen_tokens,
            "utility_sum": evaluation_utility if evaluations else None,
            "mean_utility": (evaluation_utility / evaluations if evaluations else None),
            "mean_rollout_utility": (
                evaluation_utility / evaluations if evaluations else None
            ),
            "cumulative_rollout_net_utility": (
                evaluation_utility - beta * search_qwen_tokens if evaluations else None
            ),
            "mean_rollout_net_utility": (
                (evaluation_utility - beta * search_qwen_tokens) / evaluations
                if evaluations
                else None
            ),
            "observed_utility_token_area": None,
            "observed_utility_token_auc": None,
            "mean_search_observed_utility_token_auc": None,
            "observed_utility_token_auc_searches": 0,
            "zero_qwen_token_searches": 0,
            "auc_convention": (
                "unavailable_without_per-evaluation_observed_trajectory"
            ),
            "not_a_deployment_objective": True,
        }
    trace_scope = (
        "componentized_qwen_online"
        if detailed_sources
        and all(detailed_sources)
        and test_detailed
        and test_usage_detailed
        else "legacy_qwen_input_collapsed"
    )
    static_metric = endj_static(
        endpoint_utility_sum=endpoint_utility,
        search_qwen_tokens=search_qwen_tokens,
        searches=searches,
        beta=beta,
    )
    static_metric["cost_scope"] = trace_scope
    deploy_metric = endj_test_search(
        test_mean_utility=test_mean_utility,
        train_search_qwen_tokens=search_qwen_tokens,
        deployment_horizon=deployment_horizon,
        beta=beta,
    )
    deploy_metric["cost_scope"] = trace_scope
    return {
        "method": method,
        "seed": seed,
        "search_cost_mode": result.get("search_cost_mode"),
        "token_beta": beta,
        "ledger_source": ledger_source,
        "qwen_cost_trace_scope": trace_scope,
        "generator_4b_input_tokens": generator_input_tokens,
        "generator_4b_output_tokens": generator_output_tokens,
        "train_token_usage": train_token_usage,
        "train_online_search_raw_tokens": online_search_raw_token_count(
            train_token_usage
        ),
        "test_token_usage": test_token_usage,
        "train_search_generator_tokens": generator_tokens,
        "train_search_qwen_tokens": search_qwen_tokens,
        "actor_scoring_tokens": actor_scoring_tokens,
        "optimizer_actor_scoring_tokens": optimizer_actor_scoring_tokens,
        "test_generator_4b_input_tokens": test_inputs,
        "test_generator_4b_output_tokens": test_outputs,
        "internal_rollout_diagnostics": rollout_diagnostics,
        "endj_static": static_metric,
        "endj_test_search": deploy_metric,
        "successful_qwen_generations": successful_generations,
        "qwen_generation_attempts": generation_attempts,
        "qwen_retry_count": retry_count,
        "qwen_length_retry_count": length_retry_count,
        "qwen_fallback_count": fallback_count,
        "tree_decisions": tree_decisions,
        "tree_cost_active_decisions": tree_cost_active_decisions,
        "tree_cost_changed_choices": tree_cost_changed_choices,
        "rollout_decisions": rollout_decisions,
        "rollout_cost_active_decisions": rollout_cost_active_decisions,
        "wtb_modes": dict(sorted(modes.items())),
        "test_f1": test.get("mean_f1"),
        "test_utility": test_mean_utility,
    }


def _aggregate_method(method: str, cells: list[dict[str, Any]]) -> dict[str, Any]:
    beta = float(cells[0]["token_beta"])
    horizon = int(cells[0]["endj_test_search"]["deployment_horizon"])
    searches = sum(int(cell["endj_static"]["searches"]) for cell in cells)
    endpoint_utility_sum = sum(
        float(cell["endj_static"]["mean_endpoint_utility"])
        * int(cell["endj_static"]["searches"])
        for cell in cells
        if cell["endj_static"]["mean_endpoint_utility"] is not None
    )
    generator_input_tokens = sum(
        int(cell["generator_4b_input_tokens"]) for cell in cells
    )
    generator_output_tokens = sum(
        int(cell["generator_4b_output_tokens"]) for cell in cells
    )
    generator_tokens = generator_input_tokens + generator_output_tokens
    train_token_usage = add_token_usage(*(cell["train_token_usage"] for cell in cells))
    search_qwen_tokens = search_qwen_token_count(train_token_usage)
    test_token_usage = add_token_usage(*(cell["test_token_usage"] for cell in cells))
    test_utilities = [
        float(cell["test_utility"])
        for cell in cells
        if cell["test_utility"] is not None
    ]
    mean_test_utility = (
        sum(test_utilities) / len(test_utilities) if test_utilities else None
    )
    rollout_rows = [cell["internal_rollout_diagnostics"] for cell in cells]
    rollout_evaluations = sum(int(row["logical_evaluations"]) for row in rollout_rows)
    rollout_utility_sum = sum(
        float(row["utility_sum"])
        for row in rollout_rows
        if row["utility_sum"] is not None
    )
    area_rows = [
        row for row in rollout_rows if row["observed_utility_token_area"] is not None
    ]
    auc_searches = sum(
        int(row["observed_utility_token_auc_searches"]) for row in area_rows
    )
    auc_sum = sum(
        float(row["mean_search_observed_utility_token_auc"])
        * int(row["observed_utility_token_auc_searches"])
        for row in area_rows
        if row["mean_search_observed_utility_token_auc"] is not None
    )
    observed_area = sum(float(row["observed_utility_token_area"]) for row in area_rows)
    observed_auc_tokens = sum(int(row["qwen_tokens"]) for row in area_rows)
    observed_curve_complete = len(area_rows) == len(rollout_rows)
    aggregate_horizon = horizon * len(cells)
    return {
        "method": method,
        "seed": "all",
        "seed_count": len(cells),
        "qwen_cost_trace_scopes": sorted(
            {str(cell["qwen_cost_trace_scope"]) for cell in cells}
        ),
        "generator_4b_input_tokens": generator_input_tokens,
        "generator_4b_output_tokens": generator_output_tokens,
        "train_search_generator_tokens": generator_tokens,
        "train_search_qwen_tokens": search_qwen_tokens,
        "train_token_usage": train_token_usage,
        "train_online_search_raw_tokens": online_search_raw_token_count(
            train_token_usage
        ),
        "test_token_usage": test_token_usage,
        "actor_scoring_tokens": sum(
            int(cell["actor_scoring_tokens"]) for cell in cells
        ),
        "optimizer_actor_scoring_tokens": sum(
            int(cell["optimizer_actor_scoring_tokens"]) for cell in cells
        ),
        "internal_rollout_diagnostics": {
            "logical_evaluations": rollout_evaluations,
            "qwen_tokens": search_qwen_tokens,
            "utility_sum": rollout_utility_sum if rollout_evaluations else None,
            "mean_utility": (
                rollout_utility_sum / rollout_evaluations
                if rollout_evaluations
                else None
            ),
            "mean_rollout_utility": (
                rollout_utility_sum / rollout_evaluations
                if rollout_evaluations
                else None
            ),
            "cumulative_rollout_net_utility": (
                rollout_utility_sum - beta * search_qwen_tokens
                if rollout_evaluations
                else None
            ),
            "mean_rollout_net_utility": (
                (rollout_utility_sum - beta * search_qwen_tokens) / rollout_evaluations
                if rollout_evaluations
                else None
            ),
            "observed_utility_token_area": (
                observed_area if observed_curve_complete else None
            ),
            "observed_utility_token_auc": (
                observed_area / observed_auc_tokens
                if observed_curve_complete and observed_auc_tokens
                else None
            ),
            "mean_search_observed_utility_token_auc": (
                auc_sum / auc_searches
                if observed_curve_complete and auc_searches
                else None
            ),
            "observed_curve_cells": len(area_rows),
            "cell_count": len(rollout_rows),
            "observed_curve_complete": observed_curve_complete,
            "observed_utility_token_auc_searches": auc_searches,
            "zero_qwen_token_searches": sum(
                int(row["zero_qwen_token_searches"]) for row in area_rows
            ),
            "auc_convention": ("right_endpoint_incumbent_weighted_by_pull_qwen_tokens"),
            "not_a_deployment_objective": True,
        },
        "endj_static": {
            **endj_static(
                endpoint_utility_sum=endpoint_utility_sum,
                search_qwen_tokens=search_qwen_tokens,
                searches=searches,
                beta=beta,
            ),
            "cost_scopes": sorted(
                {str(cell["qwen_cost_trace_scope"]) for cell in cells}
            ),
        },
        "endj_test_search": {
            **endj_test_search(
                test_mean_utility=mean_test_utility,
                train_search_qwen_tokens=search_qwen_tokens,
                deployment_horizon=aggregate_horizon,
                beta=beta,
            ),
            "deployment_horizon_per_seed": horizon,
            "cost_scopes": sorted(
                {str(cell["qwen_cost_trace_scope"]) for cell in cells}
            ),
        },
        "test_f1": (
            sum(float(cell["test_f1"]) for cell in cells if cell["test_f1"] is not None)
            / sum(cell["test_f1"] is not None for cell in cells)
            if any(cell["test_f1"] is not None for cell in cells)
            else None
        ),
        "test_utility": mean_test_utility,
    }


def _comparison(
    left: dict[str, Any], right: dict[str, Any], objective: str
) -> dict[str, Any]:
    left_value = left[objective]["value"]
    right_value = right[objective]["value"]
    return {
        "left": left["method"],
        "right": right["method"],
        "objective": objective,
        "left_value": left_value,
        "right_value": right_value,
        "left_minus_right": (
            None
            if left_value is None or right_value is None
            else float(left_value) - float(right_value)
        ),
        "left_minus_right_train_search_generator_tokens": (
            int(left["train_search_generator_tokens"])
            - int(right["train_search_generator_tokens"])
        ),
        "left_minus_right_train_search_qwen_tokens": (
            int(left["train_search_qwen_tokens"])
            - int(right["train_search_qwen_tokens"])
        ),
    }


def evaluate_run(
    output_dir: Path,
    *,
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
    deployment_horizon: int = DEFAULT_DEPLOYMENT_HORIZON,
) -> dict[str, Any]:
    if deployment_horizon <= 0:
        raise ValueError("deployment_horizon must be positive")
    cells = [
        _cell_ledger(
            output_dir,
            method=method,
            seed=seed,
            deployment_horizon=deployment_horizon,
        )
        for method in methods
        for seed in seeds
    ]
    betas = {float(cell["token_beta"]) for cell in cells}
    if len(betas) != 1:
        raise ValueError(f"cells use different token_beta values: {sorted(betas)}")
    beta = betas.pop()
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("stored token_beta must be finite and non-negative")
    aggregates = [
        _aggregate_method(method, [cell for cell in cells if cell["method"] == method])
        for method in methods
    ]
    by_method = {row["method"]: row for row in aggregates}

    search_comparisons = []
    agent = by_method.get("agentuct_atppo")
    if agent is not None:
        for baseline_name in ("uct_atppo", "uniform_atppo"):
            baseline = by_method.get(baseline_name)
            if baseline is not None:
                search_comparisons.append(_comparison(agent, baseline, "endj_static"))
    deployment_comparisons = []
    ppo = by_method.get("ppo")
    if ppo is not None:
        for method in ("agentuct_atppo", "uct_atppo", "uniform_atppo"):
            tree = by_method.get(method)
            if tree is not None:
                deployment_comparisons.append(
                    _comparison(tree, ppo, "endj_test_search")
                )
    index_usage = empty_token_usage()
    index_path = output_dir / "index_build_tokens.json"
    if index_path.exists():
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        index_usage, _ = _token_usage(index_payload)
    return {
        "method_order": list(methods),
        "seeds": list(seeds),
        "token_beta": beta,
        "deployment_horizon_per_seed": deployment_horizon,
        "token_fields": list(TOKEN_FIELDS),
        "shared_index_token_usage": index_usage,
        "endj_static_semantics": (
            "mean best endpoint utility per actor-state search minus token_beta "
            "times mean physical online Qwen tokens per search"
        ),
        "endj_test_search_semantics": (
            "final tree-free mean test utility minus token_beta times total physical "
            "training-search online Qwen tokens divided by N"
        ),
        "observed_utility_token_auc_semantics": (
            "supporting search diagnostic: normalized right-endpoint area of "
            "best utility observed so far against cumulative physical online-Qwen "
            "tokens; it is not a Pareto objective or delivered deployment reward"
        ),
        "cumulative_rollout_net_utility_semantics": (
            "UCB-style appendix diagnostic: sum of internal rollout utilities minus "
            "token_beta times realized online-Qwen tokens; because rollouts are "
            "counterfactual pulls for one decision, this is not an endpoint objective"
        ),
        "search_qwen_token_fields": list(SEARCH_QWEN_TOKEN_FIELDS),
        "cost_exclusions": (
            "E5, BGE-M3, document indexing, actor scoring, PPO update, validation, "
            "and test inference are traced separately but are not in the primary "
            "online-Qwen C_train-search scalar"
        ),
        "heterogeneous_token_warning": (
            "The unweighted Qwen token sum is a frozen search-cost proxy, not a "
            "claim that 4B and 0.6B tokens have equal FLOP or latency cost; E5/BGE "
            "remain outside this scalar"
        ),
        "cells": cells,
        "method_aggregates": aggregates,
        "agentuct_search_comparisons": search_comparisons,
        "deployment_vs_ppo_comparisons": deployment_comparisons,
    }


def write_evaluation(
    output_dir: Path,
    *,
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
    deployment_horizon: int = DEFAULT_DEPLOYMENT_HORIZON,
) -> dict[str, Any]:
    evaluation = evaluate_run(
        output_dir,
        methods=methods,
        seeds=seeds,
        deployment_horizon=deployment_horizon,
    )
    (output_dir / "evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return evaluation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a completed ASCT run")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=(*DEFAULT_METHODS, "vineppo", "at2po", "cf_actor"),
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[11])
    parser.add_argument(
        "--deployment-horizon",
        type=int,
        default=DEFAULT_DEPLOYMENT_HORIZON,
        help="N used to amortize one trained policy's C_train-search",
    )
    args = parser.parse_args(argv)
    evaluation = write_evaluation(
        args.output_dir,
        methods=tuple(args.methods),
        seeds=tuple(args.seeds),
        deployment_horizon=args.deployment_horizon,
    )
    print(json.dumps(evaluation, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
