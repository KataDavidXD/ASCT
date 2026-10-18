"""Online PPO loop shared by the main and tiny RAG experiments."""

from __future__ import annotations

import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from uct_engine import (
    QIDPrefixCache,
    SearchResult,
    add_token_usage,
    generator_token_count,
    legacy_token_usage,
    normalize_token_usage,
    online_search_raw_token_count,
    run_search,
)

from .at2po import AT2POConfig, AT2POUpdateConfig, collect_at2po_tree, update_at2po
from .dataset import HotpotItem
from .policy import PlannerPolicy
from .rag import JSONTemplateGenerator, RAGComponents, RAGWorkflowState, WorkflowOutcome
from .ragspace import RAGSpace
from .vine import VINE_SETTINGS, collect_vine_trajectory

DEFAULT_METHODS = ("agentuct_atppo", "uct_atppo", "uniform_atppo", "ppo")
METHODS = (*DEFAULT_METHODS, "vineppo", "at2po")
TREE_METHODS = frozenset({"uniform_atppo", "uct_atppo", "agentuct_atppo"})
CACHE_METHODS = TREE_METHODS | {"vineppo", "at2po"}


@dataclass(frozen=True)
class CellSpec:
    method: str
    seed: int


def _sample(probabilities: list[float], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if threshold <= cumulative:
            return index
    return len(probabilities) - 1


def _search(
    state: RAGWorkflowState,
    *,
    method: str,
    search_budget: int,
    token_beta: float,
    search_cost_mode: str,
    rng: random.Random,
    cache: QIDPrefixCache,
) -> SearchResult:
    return run_search(
        state,
        method={
            "uniform_atppo": "uniform",
            "uct_atppo": "uct",
            "agentuct_atppo": "agentuct",
        }[method],
        search_budget=search_budget,
        token_beta=token_beta,
        search_cost_mode=search_cost_mode,
        rng=rng,
        cache=cache,
    )


def _collect_trajectory(
    initial_state: RAGWorkflowState,
    policy: PlannerPolicy,
    *,
    method: str,
    rng: random.Random,
    search_budget: int,
    token_beta: float,
    search_cost_mode: str,
    cache: QIDPrefixCache | None,
    search_seed: int,
    greedy: bool = False,
    at2po_config: AT2POConfig = AT2POConfig(),
) -> tuple[RAGWorkflowState, WorkflowOutcome, list[dict[str, Any]]]:
    if method == "at2po":
        if cache is None or greedy:
            raise ValueError("AT2PO collection requires WTB and sampled actor actions")
        return collect_at2po_tree(
            initial_state, policy, rng=rng, cache=cache, config=at2po_config
        )
    if method == "vineppo":
        if cache is None or greedy:
            raise ValueError("VinePPO collection requires WTB and sampled actor actions")
        return collect_vine_trajectory(
            initial_state, policy, rng=rng, search_seed=search_seed,
            budget=search_budget, cache=cache,
        )
    state = initial_state
    item = state.item
    records: list[dict[str, Any]] = []
    while not state.is_terminal():
        actions = list(state.available_actions())
        prompt = state.prompt()
        scoring_start = policy.actor_scoring_tokens
        probabilities, action_logprobabilities = policy.distribution(prompt, actions)
        actor_scoring_tokens = policy.actor_scoring_tokens - scoring_start
        chosen_index = (
            max(range(len(actions)), key=probabilities.__getitem__)
            if greedy
            else _sample(probabilities, rng)
        )
        action = actions[chosen_index]
        record: dict[str, Any] = {
            "qid": item.qid,
            "step": len(state.path),
            "stage": state.stage.value,
            "state_path": list(state.path),
            "prompt": prompt,
            "labels": actions,
            "chosen_label": action,
            "old_logprob": action_logprobabilities[chosen_index],
            "probabilities": probabilities,
            "actor_scoring_tokens": actor_scoring_tokens,
        }
        if method in TREE_METHODS:
            if cache is None:
                raise RuntimeError("tree method requires a QID-scoped WTB cache")
            result = _search(
                state,
                method=method,
                search_budget=search_budget,
                token_beta=token_beta,
                search_cost_mode=search_cost_mode,
                rng=random.Random(f"{search_seed}:{item.qid}:{len(state.path)}"),
                cache=cache,
            )
            if tuple(actions) != result.actions:
                raise RuntimeError(
                    "search action order differs from actor action order"
                )
            baseline = sum(
                probability * utility
                for probability, utility in zip(
                    probabilities, result.q_values, strict=True
                )
            )
            wtb_modes = Counter(str(row["wtb_mode"]) for row in result.trace)
            tree_decisions = [
                decision
                for row in result.trace
                for decision in row["tree_decisions"]
            ]
            rollout_decisions = [
                decision
                for row in result.trace
                for decision in row["rollout_decisions"]
            ]
            record.update(
                q_performance=list(result.q_performance),
                q_normalized_execution_cost=list(result.q_normalized_execution_costs),
                q_utility=list(result.q_values),
                visit_counts=list(result.visit_counts),
                search_budget=result.search_budget,
                search_cost_mode=result.search_cost_mode,
                token_beta=result.token_beta,
                candidate_estimated_search_tokens=list(
                    result.initial_estimated_search_tokens
                ),
                advantage=result.q_values[chosen_index] - baseline,
                best_searched_workflow={
                    "path": list(result.best_terminal_path),
                    "performance": result.best_terminal_performance,
                    "normalized_execution_cost": (
                        result.best_terminal_normalized_execution_cost
                    ),
                    "utility": result.best_terminal_utility,
                },
                search_token_metrics={
                    "logical_evaluations": result.search_budget,
                    "token_usage": normalize_token_usage(result.token_usage),
                    **normalize_token_usage(result.token_usage),
                    "generator_4b_tokens": generator_token_count(result.token_usage),
                    "online_search_raw_tokens": online_search_raw_token_count(
                        result.token_usage
                    ),
                    "qwen_input_tokens": result.qwen_input_tokens,
                    "qwen_output_tokens": result.qwen_output_tokens,
                    "qwen_tokens": result.qwen_tokens,
                    "qwen_generation_attempts": result.qwen_generation_attempts,
                    "qwen_retry_count": result.qwen_retry_count,
                    "qwen_length_retry_count": result.qwen_length_retry_count,
                    "qwen_fallback_count": result.qwen_fallback_count,
                    "successful_qwen_generations": (
                        result.successful_qwen_generations
                    ),
                    "cumulative_evaluation_utility": (
                        result.cumulative_evaluation_utility
                    ),
                    "mean_evaluation_utility": result.mean_evaluation_utility,
                    "wtb_modes": dict(sorted(wtb_modes.items())),
                    "tree_decisions": len(tree_decisions),
                    "tree_cost_active_decisions": sum(
                        int(bool(decision.get("cost_active", False)))
                        for decision in tree_decisions
                    ),
                    "tree_cost_changed_choices": sum(
                        int(bool(decision.get("cost_changed_choice", False)))
                        for decision in tree_decisions
                    ),
                    "rollout_decisions": len(rollout_decisions),
                    "rollout_cost_active_decisions": sum(
                        int(bool(decision["cost_active"]))
                        for decision in rollout_decisions
                    ),
                    "final_best_utility": result.best_terminal_utility,
                },
                search_trace=list(result.trace),
                wtb_cache=result.cache,
            )
        records.append(record)
        if method in TREE_METHODS:
            state = cache.materialized_state(item.qid, (*state.path, action))
        else:
            state = state.child(action)

    outcome = state.outcome()
    if method == "ppo":
        for record in records:
            record["advantage"] = outcome.utility
    return state, outcome, records


def _trajectory(
    item: HotpotItem,
    policy: PlannerPolicy,
    *,
    method: str,
    rng: random.Random,
    search_budget: int,
    token_beta: float,
    search_cost_mode: str,
    workflow_cost_lambda: float,
    execution_cost_reference: float,
    components: RAGComponents,
    search_seed: int = 0,
    greedy: bool = False,
    cache: QIDPrefixCache | None = None,
    at2po_config: AT2POConfig = AT2POConfig(),
) -> tuple[RAGWorkflowState, WorkflowOutcome, list[dict[str, Any]]]:
    initial_state = RAGWorkflowState(
        item=item,
        workflow_cost_lambda=workflow_cost_lambda,
        execution_cost_reference=execution_cost_reference,
        components=components,
    )
    arguments = dict(
        initial_state=initial_state,
        policy=policy,
        method=method,
        rng=rng,
        search_budget=search_budget,
        token_beta=token_beta,
        search_cost_mode=search_cost_mode,
        search_seed=search_seed,
        greedy=greedy,
        at2po_config=at2po_config,
    )
    if method not in CACHE_METHODS:
        return _collect_trajectory(**arguments, cache=None)
    if cache is None:
        raise RuntimeError("tree method requires one cell-scoped WTB cache")
    cache.begin_qid(initial_state)
    return _collect_trajectory(**arguments, cache=cache)




def _collect_item(
    item: HotpotItem,
    policy: PlannerPolicy,
    *,
    spec: CellSpec,
    iteration: int,
    search_budget: int,
    token_beta: float,
    search_cost_mode: str,
    workflow_cost_lambda: float,
    execution_cost_reference: float,
    components: RAGComponents,
    cache: QIDPrefixCache | None,
    at2po_config: AT2POConfig = AT2POConfig(),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Collect one complete QID with randomness independent of worker order."""

    rng = random.Random(f"{spec.seed}:{iteration}:{item.qid}:actor")
    state, outcome, item_records = _trajectory(
        item,
        policy,
        method=spec.method,
        rng=rng,
        search_budget=search_budget,
        token_beta=token_beta,
        search_cost_mode=search_cost_mode,
        workflow_cost_lambda=workflow_cost_lambda,
        execution_cost_reference=execution_cost_reference,
        components=components,
        search_seed=spec.seed + iteration * 1_000_003,
        cache=cache,
        at2po_config=at2po_config,
    )
    state_path = list(state.path)
    query_history = [list(queries) for queries in state.query_history]
    terminal_token_usage = add_token_usage(
        outcome.token_usage,
        {
            "actor_scoring_tokens": sum(
                int(record.get("actor_scoring_tokens", 0)) for record in item_records
            )
        },
    )
    trajectory = {
        "iteration": iteration,
        "qid": item.qid,
        "actor_path": state_path,
        "terminal": {
            "performance": outcome.performance,
            "execution_cost_words": outcome.execution_cost_words,
            "normalized_execution_cost": outcome.normalized_execution_cost,
            "utility": outcome.utility,
            "support_recall": outcome.support_recall,
            "qwen_input_tokens": outcome.qwen_input_tokens,
            "qwen_output_tokens": outcome.qwen_output_tokens,
            "qwen_tokens": outcome.qwen_input_tokens + outcome.qwen_output_tokens,
            "token_usage": terminal_token_usage,
            **terminal_token_usage,
            "qwen_generation_attempts": outcome.qwen_generation_attempts,
            "qwen_retry_count": outcome.qwen_retry_count,
            "qwen_length_retry_count": outcome.qwen_length_retry_count,
            "qwen_fallback_count": outcome.qwen_fallback_count,
            "query_history": query_history,
            "selected_document_indices": list(outcome.selected_document_indices),
            "selected_titles": [
                item.context_titles[index]
                for index in outcome.selected_document_indices
            ],
            "prediction": outcome.prediction,
            "answer": item.answer,
        },
        "steps": item_records,
    }
    if spec.method == "at2po":
        trajectory["at2po_tree"] = item_records[0].pop("at2po_tree")
        trajectory["representative_path"] = trajectory.pop("actor_path")
        trajectory["terminal_role"] = "first_initial_rollout_for_trace_only"
        trajectory["collection_token_usage"] = trajectory["at2po_tree"]["token_usage"]
    return trajectory, [_training_record(record) for record in item_records]


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def _log_offsets(event_path: Path, trajectory_path: Path) -> dict[str, int]:
    return {
        "events": event_path.stat().st_size if event_path.exists() else 0,
        "trajectories": trajectory_path.stat().st_size
        if trajectory_path.exists()
        else 0,
    }


def _truncate_log(path: Path, offset: int) -> None:
    if offset < 0:
        raise ValueError("checkpoint log offset must be non-negative")
    if not path.exists():
        if offset:
            raise RuntimeError(f"checkpoint expects missing trace log: {path}")
        return
    if path.stat().st_size < offset:
        raise RuntimeError(f"trace log is shorter than its checkpoint offset: {path}")
    with path.open("r+b") as handle:
        handle.truncate(offset)


def _training_record(record: dict[str, Any]) -> dict[str, Any]:
    output = {
        key: record[key]
        for key in (
            "prompt",
            "labels",
            "chosen_label",
            "old_logprob",
            "advantage",
            "actor_scoring_tokens",
        )
    }
    if "search_token_metrics" in record:
        output["search_token_metrics"] = record["search_token_metrics"]
    if "vine_value" in record:
        for key in ("qid", "step", "vine_value", "next_vine_value", "vine_samples", "reward", "terminated", "actor_execution"):
            output[key] = record[key]
    if "at2po_sequence" in record:
        for key in ("qid", "at2po_node", "at2po_sequence", "at2po_sequence_length"):
            output[key] = record[key]
    return output


def _summarize_search_tokens(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    searches = [
        record["search_token_metrics"]
        for record in records
        if "search_token_metrics" in record
    ]
    search_usage = add_token_usage(
        *(
            normalize_token_usage(search["token_usage"])
            if "token_usage" in search
            else legacy_token_usage(
                qwen_input_tokens=int(search.get("qwen_input_tokens", 0)),
                qwen_output_tokens=int(search.get("qwen_output_tokens", 0)),
            )
            for search in searches
        )
    )
    token_usage = add_token_usage(
        search_usage,
        {
            "actor_scoring_tokens": sum(
                int(record.get("actor_scoring_tokens", 0)) for record in records
            )
        },
    )
    logical_evaluations = sum(
        int(search["logical_evaluations"]) for search in searches
    )
    qwen_input_tokens = sum(int(search["qwen_input_tokens"]) for search in searches)
    qwen_output_tokens = sum(
        int(search["qwen_output_tokens"]) for search in searches
    )
    qwen_tokens = qwen_input_tokens + qwen_output_tokens
    qwen_generation_attempts = sum(
        int(search.get("qwen_generation_attempts", 0)) for search in searches
    )
    qwen_retry_count = sum(int(search["qwen_retry_count"]) for search in searches)
    qwen_length_retry_count = sum(
        int(search["qwen_length_retry_count"]) for search in searches
    )
    qwen_fallback_count = sum(
        int(search.get("qwen_fallback_count", 0)) for search in searches
    )
    generation_ledgers = [
        search for search in searches if "successful_qwen_generations" in search
    ]
    successful_qwen_generations = sum(
        int(search["successful_qwen_generations"])
        for search in generation_ledgers
    )
    utility_ledgers = [
        search for search in searches if "cumulative_evaluation_utility" in search
    ]
    cumulative_evaluation_utility = sum(
        float(search["cumulative_evaluation_utility"]) for search in utility_ledgers
    )
    cumulative_endpoint_utility = sum(
        float(search["final_best_utility"]) for search in searches
    )
    mode_ledgers = [search for search in searches if "wtb_modes" in search]
    wtb_modes: Counter[str] = Counter()
    for search in mode_ledgers:
        wtb_modes.update(
            {str(mode): int(count) for mode, count in search["wtb_modes"].items()}
        )
    return {
        "searches": len(searches),
        **({
            "vine_actor_execution_token_usage": add_token_usage(*(
                record["actor_execution"]["token_usage"] for record in records
            )),
            "vine_auxiliary_actor_scoring_tokens": search_usage["actor_scoring_tokens"],
        } if records and "vine_value" in records[0] else {}),
        "logical_evaluations": logical_evaluations,
        "token_usage": token_usage,
        **token_usage,
        "generator_4b_tokens": generator_token_count(search_usage),
        "online_search_raw_tokens": online_search_raw_token_count(search_usage),
        "qwen_input_tokens": qwen_input_tokens,
        "qwen_output_tokens": qwen_output_tokens,
        "qwen_tokens": qwen_tokens,
        "qwen_generation_attempts": qwen_generation_attempts,
        "qwen_retry_count": qwen_retry_count,
        "qwen_length_retry_count": qwen_length_retry_count,
        "qwen_fallback_count": qwen_fallback_count,
        "tree_decisions": sum(
            int(search.get("tree_decisions", 0)) for search in searches
        ),
        "tree_cost_active_decisions": sum(
            int(search.get("tree_cost_active_decisions", 0))
            for search in searches
        ),
        "tree_cost_changed_choices": sum(
            int(search.get("tree_cost_changed_choices", 0))
            for search in searches
        ),
        "rollout_decisions": sum(
            int(search.get("rollout_decisions", 0)) for search in searches
        ),
        "rollout_cost_active_decisions": sum(
            int(search.get("rollout_cost_active_decisions", 0))
            for search in searches
        ),
        "successful_qwen_generations": (
            successful_qwen_generations
            if len(generation_ledgers) == len(searches)
            else None
        ),
        "utility_ledger_searches": len(utility_ledgers),
        "cumulative_evaluation_utility": (
            cumulative_evaluation_utility
            if len(utility_ledgers) == len(searches)
            else None
        ),
        "mean_evaluation_utility": (
            cumulative_evaluation_utility / logical_evaluations
            if len(utility_ledgers) == len(searches) and logical_evaluations
            else None
        ),
        "cumulative_endpoint_utility": cumulative_endpoint_utility,
        "wtb_modes": (
            dict(sorted(wtb_modes.items()))
            if len(mode_ledgers) == len(searches)
            else None
        ),
        "mean_qwen_tokens_per_logical_evaluation": (
            qwen_tokens / logical_evaluations if logical_evaluations else 0.0
        ),
        "mean_final_best_utility": (
            sum(float(search["final_best_utility"]) for search in searches)
            / len(searches)
            if searches
            else 0.0
        ),
    }


def _save_checkpoint(
    path: Path,
    *,
    spec: CellSpec,
    policy: PlannerPolicy,
    iteration: int,
    completed_qids: list[str],
    records: list[dict[str, Any]],
    iteration_results: list[dict[str, Any]],
    algorithm_settings: dict[str, float | int | str],
    log_offsets: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    policy.torch.save(
        {
            "method": spec.method,
            "seed": spec.seed,
            "iteration": iteration,
            "completed_qids": completed_qids,
            "records": records,
            "iteration_results": iteration_results,
            "algorithm_settings": algorithm_settings,
            "log_offsets": log_offsets,
            "adapter": policy.adapter_state(),
            "optimizer": policy.optimizer_state(),
        },
        temporary,
    )
    os.replace(temporary, path)


def _update(
    policy: PlannerPolicy,
    records: list[dict[str, Any]],
    *,
    epochs: int,
    batch_size: int,
    method: str | None = None,
    at2po_config: AT2POConfig = AT2POConfig(),
    at2po_update_config: AT2POUpdateConfig = AT2POUpdateConfig(),
) -> dict[str, float | int]:
    if method == "at2po":
        return update_at2po(
            policy, records, epochs=epochs, batch_size=batch_size, config=at2po_config,
            execution=at2po_update_config,
        )
    torch = policy.torch
    scoring_start = policy.actor_scoring_tokens
    if policy.device.type == "cuda":
        torch.cuda.empty_cache()
    policy.model.train()
    advantages = [float(record["advantage"]) for record in records]
    mean_advantage = sum(advantages) / len(advantages) if advantages else 0.0
    variance = (
        sum((value - mean_advantage) ** 2 for value in advantages) / len(advantages)
        if advantages
        else 0.0
    )
    advantage_std = math.sqrt(variance)
    normalized = [
        (value - mean_advantage) / advantage_std if advantage_std > 1e-8 else value
        for value in advantages
    ]
    losses: list[float] = []
    gradient_norms: list[float] = []
    optimizer_steps = 0
    indexed = list(zip(records, normalized, strict=True))
    # Keep the effective PPO batch unchanged while limiting the number of
    # full Qwen scoring graphs resident on one GPU at a time.
    scoring_microbatch_size = 1
    for _ in range(epochs):
        for start in range(0, len(indexed), batch_size):
            active = [
                row
                for row in indexed[start : start + batch_size]
                if not math.isclose(row[1], 0.0, abs_tol=1e-12)
            ]
            if not active:
                continue
            if policy.optimizer is None:
                raise RuntimeError("PPO update requires the learner policy")
            policy.optimizer.zero_grad(set_to_none=True)
            batch_loss_values: list[float] = []
            for micro_start in range(0, len(active), scoring_microbatch_size):
                microbatch = active[
                    micro_start : micro_start + scoring_microbatch_size
                ]
                label_groups = [
                    list(record["labels"]) for record, _ in microbatch
                ]
                logprob_groups = policy.batch_action_logprobs(
                    [str(record["prompt"]) for record, _ in microbatch],
                    label_groups,
                    grad=True,
                )
                micro_losses = []
                for (record, advantage), labels, logits in zip(
                    microbatch,
                    label_groups,
                    logprob_groups,
                    strict=True,
                ):
                    chosen = labels.index(str(record["chosen_label"]))
                    new_logprob = torch.log_softmax(logits, dim=0)[chosen]
                    ratio = torch.exp(new_logprob - float(record["old_logprob"]))
                    weighted = ratio * advantage
                    clipped = torch.clamp(ratio, 0.8, 1.2) * advantage
                    micro_losses.append(-torch.minimum(weighted, clipped))
                torch.stack(micro_losses).sum().div(len(active)).backward()
                batch_loss_values.extend(
                    float(loss.detach().cpu()) for loss in micro_losses
                )
            gradient_norms.append(
                float(torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0))
            )
            policy.optimizer.step()
            optimizer_steps += 1
            losses.append(sum(batch_loss_values) / len(batch_loss_values))
    return {
        "mean_loss": sum(losses) / len(losses) if losses else 0.0,
        "mean_gradient_norm": sum(gradient_norms) / len(gradient_norms)
        if gradient_norms
        else 0.0,
        "max_gradient_norm": max(gradient_norms, default=0.0),
        "optimizer_steps": float(optimizer_steps),
        "samples": float(len(records)),
        "advantage_mean": mean_advantage,
        "advantage_std": advantage_std,
        "actor_scoring_tokens": policy.actor_scoring_tokens - scoring_start,
    }


def _evaluation_rows(
    policy: PlannerPolicy,
    items: list[HotpotItem],
    *,
    rng: random.Random,
    search_budget: int,
    token_beta: float,
    search_cost_mode: str,
    workflow_cost_lambda: float,
    execution_cost_reference: float,
    components: RAGComponents,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        scoring_start = policy.actor_scoring_tokens
        state, outcome, _ = _trajectory(
            item,
            policy,
            method="ppo",
            rng=rng,
            search_budget=search_budget,
            token_beta=token_beta,
            search_cost_mode=search_cost_mode,
            workflow_cost_lambda=workflow_cost_lambda,
            execution_cost_reference=execution_cost_reference,
            components=components,
            greedy=True,
        )
        token_usage = add_token_usage(
            outcome.token_usage,
            {
                "actor_scoring_tokens": (
                    policy.actor_scoring_tokens - scoring_start
                )
            },
        )
        rows.append(
            {
                "qid": item.qid,
                "workflow_path": list(state.path),
                "query_history": [list(queries) for queries in state.query_history],
                "selected_titles": [
                    item.context_titles[index]
                    for index in outcome.selected_document_indices
                ],
                "prediction": outcome.prediction,
                "answer": item.answer,
                "f1": outcome.performance,
                "support_recall": outcome.support_recall,
                "execution_cost_words": outcome.execution_cost_words,
                "normalized_execution_cost": outcome.normalized_execution_cost,
                "utility": outcome.utility,
                "qwen_input_tokens": outcome.qwen_input_tokens,
                "qwen_output_tokens": outcome.qwen_output_tokens,
                "qwen_tokens": outcome.qwen_input_tokens + outcome.qwen_output_tokens,
                "token_usage": token_usage,
                **token_usage,
                "qwen_generation_attempts": outcome.qwen_generation_attempts,
                "qwen_retry_count": outcome.qwen_retry_count,
                "qwen_length_retry_count": outcome.qwen_length_retry_count,
                "qwen_fallback_count": outcome.qwen_fallback_count,
            }
        )
    return rows


def _summarize_evaluation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("evaluation requires at least one row")
    token_usage = add_token_usage(
        *(
            normalize_token_usage(row["token_usage"])
            if "token_usage" in row
            else legacy_token_usage(
                qwen_input_tokens=int(row.get("qwen_input_tokens", 0)),
                qwen_output_tokens=int(row.get("qwen_output_tokens", 0)),
            )
            for row in rows
        )
    )
    return {
        "mean_f1": sum(row["f1"] for row in rows) / len(rows),
        "mean_support_recall": sum(row["support_recall"] for row in rows) / len(rows),
        "mean_execution_cost_words": sum(row["execution_cost_words"] for row in rows)
        / len(rows),
        "mean_utility": sum(row["utility"] for row in rows) / len(rows),
        "token_usage": token_usage,
        **token_usage,
        "qwen_input_tokens": sum(row["qwen_input_tokens"] for row in rows),
        "qwen_output_tokens": sum(row["qwen_output_tokens"] for row in rows),
        "qwen_tokens": sum(row["qwen_tokens"] for row in rows),
        "qwen_generation_attempts": sum(
            row["qwen_generation_attempts"] for row in rows
        ),
        "qwen_retry_count": sum(row["qwen_retry_count"] for row in rows),
        "qwen_length_retry_count": sum(
            row["qwen_length_retry_count"] for row in rows
        ),
        "qwen_fallback_count": sum(
            int(row.get("qwen_fallback_count", 0)) for row in rows
        ),
        "rows": rows,
    }


def _evaluate(
    policy: PlannerPolicy,
    items: list[HotpotItem],
    *,
    rng: random.Random,
    search_budget: int,
    token_beta: float,
    search_cost_mode: str,
    workflow_cost_lambda: float,
    execution_cost_reference: float,
    components: RAGComponents,
) -> dict[str, Any]:
    return _summarize_evaluation(
        _evaluation_rows(
            policy,
            items,
            rng=rng,
            search_budget=search_budget,
            token_beta=token_beta,
            search_cost_mode=search_cost_mode,
            workflow_cost_lambda=workflow_cost_lambda,
            execution_cost_reference=execution_cost_reference,
            components=components,
        )
    )


def _run_cell(
    spec: CellSpec,
    *,
    model_path: Path,
    rag_model_root: Path,
    rag_index_root: Path,
    output_dir: Path,
    train_items: list[HotpotItem],
    valid_items: list[HotpotItem],
    test_items: list[HotpotItem],
    device: str,
    resume: bool,
    max_new_train_items: int | None,
    search_budget: int,
    token_beta: float,
    search_cost_mode: str,
    workflow_cost_lambda: float,
    execution_cost_reference: float,
    iterations: int,
    epochs: int,
    ppo_batch_size: int,
    checkpoint_interval_qids: int,
    cache: QIDPrefixCache | None,
    at2po_config: AT2POConfig = AT2POConfig(),
    at2po_update_config: AT2POUpdateConfig = AT2POUpdateConfig(),
) -> dict[str, Any]:
    if spec.method not in METHODS:
        raise ValueError(f"unknown method: {spec.method}")
    if iterations <= 0 or ppo_batch_size <= 0 or checkpoint_interval_qids <= 0:
        raise ValueError(
            "iterations, ppo_batch_size, and checkpoint_interval_qids must be positive"
        )
    if (spec.method in CACHE_METHODS) != (cache is not None):
        raise ValueError("tree methods require exactly one cell-scoped WTB cache")
    algorithm_settings: dict[str, float | int | str] = {
        "search_budget": search_budget,
        "token_beta": token_beta,
        "search_cost_mode": search_cost_mode,
        "workflow_cost_lambda": workflow_cost_lambda,
        "execution_cost_reference": execution_cost_reference,
        "iterations": iterations,
        "epochs": epochs,
        "ppo_batch_size": ppo_batch_size,
        **(VINE_SETTINGS if spec.method == "vineppo" else {}),
        **(at2po_config.settings() if spec.method == "at2po" else {}),
    }
    cell_dir = output_dir / f"{spec.method}-seed-{spec.seed}"
    checkpoint_path = cell_dir / "checkpoint.pt"
    event_path = cell_dir / "events.jsonl"
    trajectory_path = cell_dir / "trajectories.jsonl"
    policy = PlannerPolicy(
        model_path, device, seed=spec.seed,
        **({"learning_rate": at2po_config.learning_rate} if spec.method == "at2po" else {}),
    )
    components = RAGComponents(
        generator=JSONTemplateGenerator(policy),
        ragspace=RAGSpace(
            rag_model_root,
            device,
            index_root=rag_index_root,
        ),
    )
    iteration = 0
    completed: list[str] = []
    records: list[dict[str, Any]] = []
    iteration_results: list[dict[str, Any]] = []
    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"resume checkpoint does not exist: {checkpoint_path}"
            )
        payload = policy.torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if payload["method"] != spec.method or int(payload["seed"]) != spec.seed:
            raise ValueError("checkpoint method/seed does not match the requested cell")
        saved_settings = payload.get("algorithm_settings")
        if not isinstance(saved_settings, dict) or any(
            saved_settings.get(name) != value
            for name, value in algorithm_settings.items()
        ):
            raise ValueError(
                "resume requires the same algorithm settings: "
                f"checkpoint={saved_settings}, "
                f"requested={algorithm_settings}"
            )
        offsets = payload.get("log_offsets")
        if not isinstance(offsets, dict):
            raise ValueError("checkpoint does not contain resumable trace-log offsets")
        _truncate_log(event_path, int(offsets["events"]))
        _truncate_log(trajectory_path, int(offsets["trajectories"]))
        policy.load_adapter_state(payload["adapter"])
        policy.load_optimizer_state(payload["optimizer"])
        iteration = int(payload["iteration"])
        completed = list(payload["completed_qids"])
        records = list(payload["records"])
        iteration_results = list(payload["iteration_results"])
        _append_jsonl(
            event_path,
            {
                "event": "resumed",
                "iteration": iteration,
                "completed_qids": len(completed),
            },
        )
    elif checkpoint_path.exists():
        raise FileExistsError(
            f"fresh run refuses existing checkpoint: {checkpoint_path}"
        )

    collected_this_call = 0

    def save_checkpoint() -> None:
        _save_checkpoint(
            checkpoint_path,
            spec=spec,
            policy=policy,
            iteration=iteration,
            completed_qids=completed,
            records=records,
            iteration_results=iteration_results,
            algorithm_settings=algorithm_settings,
            log_offsets=_log_offsets(event_path, trajectory_path),
        )

    if not resume:
        save_checkpoint()

    while iteration < iterations:
        policy.model.eval()
        completed_set = set(completed)
        for item in train_items:
            if item.qid in completed_set:
                continue
            if (
                max_new_train_items is not None
                and collected_this_call >= max_new_train_items
            ):
                save_checkpoint()
                return {
                    "method": spec.method,
                    "seed": spec.seed,
                    "status": "paused",
                    "iteration": iteration,
                    "completed": len(completed),
                }
            trajectory, training_records = _collect_item(
                item,
                policy,
                spec=spec,
                iteration=iteration,
                search_budget=search_budget,
                token_beta=token_beta,
                search_cost_mode=search_cost_mode,
                workflow_cost_lambda=workflow_cost_lambda,
                execution_cost_reference=execution_cost_reference,
                components=components,
                cache=cache,
                at2po_config=at2po_config,
            )
            records.extend(training_records)
            completed.append(item.qid)
            completed_set.add(item.qid)
            collected_this_call += 1
            _append_jsonl(
                trajectory_path,
                trajectory,
            )
            _append_jsonl(
                event_path,
                {
                    "event": "qid_collected",
                    "iteration": iteration,
                    "qid": item.qid,
                    "completed": len(completed),
                    "utility": trajectory["terminal"]["utility"],
                },
            )
            if (
                len(completed) < len(train_items)
                and len(completed) % checkpoint_interval_qids == 0
            ):
                save_checkpoint()

        components.ragspace.offload_models()
        telemetry = _update(
            policy, records, epochs=epochs, batch_size=ppo_batch_size,
            method=spec.method, at2po_config=at2po_config,
            at2po_update_config=at2po_update_config,
        )
        components.ragspace.restore_models()
        if policy.device.type == "cuda":
            policy.torch.cuda.empty_cache()
        policy.model.eval()
        valid = _evaluate(
            policy,
            valid_items,
            rng=random.Random(f"{spec.seed}:{iteration}:valid"),
            search_budget=search_budget,
            token_beta=token_beta,
            search_cost_mode=search_cost_mode,
            workflow_cost_lambda=workflow_cost_lambda,
            execution_cost_reference=execution_cost_reference,
            components=components,
        )
        test = (
            _evaluate(
                policy,
                test_items,
                rng=random.Random(f"{spec.seed}:{iteration}:test"),
                search_budget=search_budget,
                token_beta=token_beta,
                search_cost_mode=search_cost_mode,
                workflow_cost_lambda=workflow_cost_lambda,
                execution_cost_reference=execution_cost_reference,
                components=components,
            )
            if iteration == iterations - 1
            else None
        )
        iteration_result = {
            "iteration": iteration,
            "train_qids": len(completed),
            "optimizer": telemetry,
            "search_tokens": _summarize_search_tokens(records),
            "valid": valid,
            "test": test,
        }
        iteration_results.append(iteration_result)
        completed_iteration = iteration
        iteration += 1
        completed = []
        records = []
        save_checkpoint()
        _append_jsonl(
            event_path,
            {
                "event": "iteration_completed",
                "iteration": completed_iteration,
                "valid_f1": valid["mean_f1"],
                "test_f1": None if test is None else test["mean_f1"],
            },
        )

    final_iteration = iteration_results[-1]
    result = {
        "method": spec.method,
        **({"at2po_settings": at2po_config.settings()} if spec.method == "at2po" else {}),
        "seed": spec.seed,
        "status": "complete",
        "completed_iterations": iterations,
        "search_cost_mode": search_cost_mode,
        "token_beta": token_beta,
        "workflow_cost_lambda": workflow_cost_lambda,
        "execution_cost_reference": execution_cost_reference,
        "iterations": iteration_results,
        "valid": final_iteration["valid"],
        "test": final_iteration["test"],
    }
    (cell_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _append_jsonl(event_path, {"event": "complete", "iterations": iterations})
    return result


def run_cell(
    spec: CellSpec,
    *,
    model_path: Path,
    rag_model_root: Path,
    rag_index_root: Path,
    output_dir: Path,
    train_items: list[HotpotItem],
    valid_items: list[HotpotItem],
    test_items: list[HotpotItem],
    device: str,
    resume: bool,
    max_new_train_items: int | None,
    search_budget: int,
    token_beta: float,
    search_cost_mode: str,
    workflow_cost_lambda: float,
    execution_cost_reference: float,
    iterations: int,
    epochs: int,
    ppo_batch_size: int,
    checkpoint_interval_qids: int,
    at2po_config: AT2POConfig = AT2POConfig(),
    at2po_update_config: AT2POUpdateConfig = AT2POUpdateConfig(),
) -> dict[str, Any]:
    arguments = dict(
        model_path=model_path,
        rag_model_root=rag_model_root,
        rag_index_root=rag_index_root,
        output_dir=output_dir,
        train_items=train_items,
        valid_items=valid_items,
        test_items=test_items,
        device=device,
        resume=resume,
        max_new_train_items=max_new_train_items,
        search_budget=search_budget,
        token_beta=token_beta,
        search_cost_mode=search_cost_mode,
        workflow_cost_lambda=workflow_cost_lambda,
        execution_cost_reference=execution_cost_reference,
        iterations=iterations,
        epochs=epochs,
        ppo_batch_size=ppo_batch_size,
        checkpoint_interval_qids=checkpoint_interval_qids,
        at2po_config=at2po_config,
        at2po_update_config=at2po_update_config,
    )
    if spec.method not in CACHE_METHODS:
        return _run_cell(spec, **arguments, cache=None)
    with TemporaryDirectory(
        prefix=f"atppo-wtb-{spec.method}-{spec.seed}-"
    ) as directory:
        cache = QIDPrefixCache(directory)
        try:
            return _run_cell(spec, **arguments, cache=cache)
        finally:
            cache.close()


__all__ = [
    "DEFAULT_METHODS",
    "METHODS",
    "TREE_METHODS",
    "CellSpec",
    "run_cell",
]
