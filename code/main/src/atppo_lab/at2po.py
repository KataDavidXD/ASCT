"""AT²PO's public-code tree credit, adapted to categorical workflow actions.

Reference: zzfoutofspace/ATPO@3cafdff6e3ef2c515d234fa9ae42728062707e09.
Each sampled occurrence is a distinct tree node even when WTB reuses its work.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from uct_engine import add_token_usage, empty_token_usage, generator_token_count
from uct_engine.tokens import qwen_input_token_count

from .vine import COUNTERS, _advance, _sample

UPSTREAM_COMMIT = "3cafdff6e3ef2c515d234fa9ae42728062707e09"


@dataclass(frozen=True)
class AT2POUpdateConfig:
    """Execution knobs; kept out of the algorithm/resume settings."""

    node_microbatch_size: int = 2
    backend: str = "parallel"

    def __post_init__(self) -> None:
        if type(self.node_microbatch_size) is not int or self.node_microbatch_size < 1:
            raise ValueError("AT2PO node microbatch size must be a positive integer")
        if self.backend not in {"parallel", "rank0"}:
            raise ValueError("AT2PO update backend must be parallel or rank0")


@dataclass(frozen=True)
class AT2POConfig:
    leaf_budget: int = 22
    initial_rollouts: int | None = None
    expansion_rounds: int = 2
    clip_low: float = 0.003
    clip_high: float = 0.004
    learning_rate: float = 1e-6

    def __post_init__(self) -> None:
        for name in ("leaf_budget", "expansion_rounds"):
            value = getattr(self, name)
            if type(value) is not int or value < (2 if name == "leaf_budget" else 0):
                raise ValueError(f"invalid AT2PO {name}: {value}")
        if self.initial_rollouts is not None and (
            type(self.initial_rollouts) is not int
            or not 1 <= self.initial_rollouts <= self.leaf_budget
        ):
            raise ValueError("AT2PO initial_rollouts must be in [1, leaf_budget]")
        if self.expansion_rounds == 0 and self.initial != self.leaf_budget:
            raise ValueError("zero expansion rounds requires all leaves initially")
        if not (0 < self.clip_low < 1 and 0 < self.clip_high < 1):
            raise ValueError("AT2PO clip bounds must be in (0, 1)")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("AT2PO learning rate must be finite and positive")

    @property
    def initial(self) -> int:
        # Official 10 initial + 2 * 6 expansion at B=22. For a new leaf budget,
        # keep the initial fraction and distribute remaining draws over rounds.
        return (
            self.initial_rollouts
            if self.initial_rollouts is not None
            else max(1, (self.leaf_budget * 10 + 11) // 22)
        )

    @property
    def beams(self) -> tuple[int, ...]:
        if not self.expansion_rounds:
            return ()
        quotient, remainder = divmod(
            self.leaf_budget - self.initial, self.expansion_rounds
        )
        return tuple(
            quotient + int(i < remainder) for i in range(self.expansion_rounds)
        )

    def settings(self) -> dict[str, int | float | str]:
        return {
            "at2po_schema": "categorical_tree_credit_wtb_v1",
            "at2po_upstream_commit": UPSTREAM_COMMIT,
            "at2po_budget_unit": "terminal_samples_per_qid_tree_including_cache_hits",
            "at2po_leaf_budget": self.leaf_budget,
            "at2po_initial_rollouts": self.initial,
            "at2po_expansion_rounds": self.expansion_rounds,
            "at2po_expansion_beams": ",".join(map(str, self.beams)),
            "at2po_expansion_penalty": 0.05,
            "at2po_entropy": "sampled_action_negative_logprob",
            "at2po_leaf_norm": "tree_population_std_plus_1e-6",
            "at2po_node_value": "child_softmax",
            "at2po_node_advantage": "node_value",
            "at2po_loss": "leaf_mean_action_mean_no_global_whitening",
            "at2po_clip_low": self.clip_low,
            "at2po_clip_high": self.clip_high,
            "at2po_learning_rate": self.learning_rate,
            "at2po_batch_unit": "leaf_trajectories",
        }


@dataclass
class _Node:
    index: int
    parent: int | None
    state: Any  # state immediately BEFORE the sampled action
    record: dict[str, Any] | None = None
    terminal: Any = None
    entropy: float = 0.0
    children: list[int] = field(default_factory=list)
    value: float = 0.0


def _breadth_first(nodes: list[_Node], root: int = 0) -> list[int]:
    result = [root]
    for index in result:
        result.extend(nodes[index].children)
    return result


def _expansion_nodes(nodes: list[_Node], count: int) -> list[tuple[int, float]]:
    scored = []
    for index in _breadth_first(nodes):
        node = nodes[index]
        if node.terminal is not None:
            continue
        parent = nodes[node.parent] if node.parent is not None else node
        score = node.entropy - 0.05 * max(0, len(parent.children) - 1)
        scored.append((index, score))
    scored.sort(key=lambda pair: pair[1], reverse=True)  # upstream stable BFS tie order
    if count and not scored:
        raise ValueError("AT2PO has no expansion candidates")
    return [scored[i % len(scored)] for i in range(count)]


def _backup(nodes: list[_Node]) -> tuple[float, float]:
    leaves = [node for node in nodes if node.terminal is not None]
    rewards = [float(node.terminal.outcome().utility) for node in leaves]
    if not rewards or not all(math.isfinite(r) for r in rewards):
        raise ValueError("AT2PO requires finite terminal utilities")
    mean = math.fsum(rewards) / len(rewards)
    std = math.sqrt(math.fsum((r - mean) ** 2 for r in rewards) / len(rewards))
    for node, reward in zip(leaves, rewards, strict=True):
        node.value = (reward - mean) / (std + 1e-6)
    for index in reversed(_breadth_first(nodes)):
        node = nodes[index]
        if not node.children:
            continue
        maximum = max(nodes[c].entropy for c in node.children)
        weights = [math.exp(nodes[c].entropy - maximum) for c in node.children]
        node.value = math.fsum(
            w * nodes[c].value for w, c in zip(weights, node.children, strict=True)
        ) / math.fsum(weights)
    return mean, std


def collect_at2po_tree(
    state: Any,
    policy: Any,
    *,
    rng: random.Random,
    cache: Any,
    config: AT2POConfig,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    if state.is_terminal():
        raise ValueError("AT2PO requires a nonterminal QID root")
    nodes = [_Node(0, None, state)]
    samples: list[dict[str, Any]] = []
    score_cache: dict[tuple[str, tuple[str, ...]], tuple[list[float], list[float]]] = {}
    usage = empty_token_usage()
    counters = dict.fromkeys(COUNTERS, 0)
    modes: Counter[str] = Counter()

    def rollout(
        start: Any, parent: int, selected: int, phase: int, score: float
    ) -> None:
        nonlocal usage
        current = start
        sample_usage = empty_token_usage()
        sample_counters = dict.fromkeys(COUNTERS, 0)
        hits = edges = 0
        reused = len(start.path)
        missed = False
        new_nodes = []
        while not current.is_terminal():
            labels = list(current.available_actions())
            prompt = current.prompt()
            key = (prompt, tuple(labels))
            before = policy.actor_scoring_tokens
            if key not in score_cache:
                score_cache[key] = policy.distribution(prompt, labels)
            probabilities, logs = score_cache[key]
            scoring = policy.actor_scoring_tokens - before
            chosen = _sample(probabilities, rng)
            logprob = float(logs[chosen])
            if not math.isfinite(logprob):
                raise ValueError("non-finite AT2PO action log probability")
            record = {
                "qid": current.qid,
                "step": len(current.path),
                "stage": current.stage.value,
                "state_path": list(current.path),
                "prompt": prompt,
                "labels": labels,
                "probabilities": probabilities,
                "chosen_label": labels[chosen],
                "old_logprob": logprob,
                # Real scoring is recorded once in the QID tree ledger below.
                "actor_scoring_tokens": 0,
            }
            node = _Node(len(nodes), parent, current, record, entropy=-logprob)
            nodes[parent].children.append(node.index)
            nodes.append(node)
            new_nodes.append(node.index)
            current, edge_usage, edge_counters, hit = _advance(
                cache, current, labels[chosen]
            )
            sample_usage = add_token_usage(
                sample_usage, edge_usage, {"actor_scoring_tokens": scoring}
            )
            for name in COUNTERS:
                sample_counters[name] += edge_counters[name]
            hits += int(hit)
            edges += 1
            if hit and not missed:
                reused += 1
            missed = missed or not hit
            if current.is_terminal():
                node.terminal = current
            parent = node.index
        outcome = current.outcome()
        mode = (
            "full_hit" if hits == edges else "partial_resume" if reused else "full_miss"
        )
        modes[mode] += 1
        usage = add_token_usage(usage, sample_usage)
        for name in COUNTERS:
            counters[name] += sample_counters[name]
        samples.append(
            {
                "sample": len(samples),
                "phase": phase,
                "selected_node": selected,
                "selection_score": score,
                "fork_state_path": list(start.path),
                "new_node_ids": new_nodes,
                "leaf_node": parent,
                "terminal_path": list(current.path),
                "utility": outcome.utility,
                "performance": outcome.performance,
                "prediction": getattr(outcome, "prediction", None),
                "answer": getattr(current.item, "answer", None),
                "execution_cost_words": getattr(outcome, "execution_cost_words", None),
                "support_recall": getattr(outcome, "support_recall", None),
                "normalized_execution_cost": outcome.normalized_execution_cost,
                "token_usage": sample_usage,
                **sample_counters,
                "wtb_mode": mode,
                "wtb_reuse_depth": reused,
            }
        )

    for _ in range(config.initial):
        rollout(state, 0, 0, -1, 0.0)
    for phase, beam in enumerate(config.beams):
        # Selection is a fixed snapshot for the whole expansion round.
        selected = _expansion_nodes(nodes, beam)
        for index, score in selected:
            node = nodes[index]
            rollout(
                node.state,
                node.parent if node.parent is not None else 0,
                index,
                phase,
                score,
            )
    if len(samples) != config.leaf_budget:
        raise RuntimeError("AT2PO terminal sample count differs from requested budget")
    mean, std = _backup(nodes)
    records: list[dict[str, Any]] = []
    leaf_ids = [i for i in _breadth_first(nodes) if nodes[i].terminal is not None]
    for leaf_id in leaf_ids:
        path = []
        index = leaf_id
        while index:
            path.append(index)
            index = nodes[index].parent
        path.reverse()
        for index in path:
            record = dict(nodes[index].record)
            record.update(
                advantage=nodes[index].value,
                at2po_node=index,
                at2po_sequence=leaf_id,
                at2po_sequence_length=len(path),
            )
            records.append(record)
    input_tokens = qwen_input_token_count(usage)
    output_tokens = usage["generator_4b_output_tokens"]
    total_utility = math.fsum(s["utility"] for s in samples)
    records[0]["search_token_metrics"] = {
        "logical_evaluations": len(samples),
        "token_usage": usage,
        "qwen_input_tokens": input_tokens,
        "qwen_output_tokens": output_tokens,
        "qwen_tokens": input_tokens + output_tokens,
        **counters,
        "successful_qwen_generations": sum(
            int(
                generator_token_count(s["token_usage"]) > 0
                and not s["qwen_fallback_count"]
            )
            for s in samples
        ),
        "cumulative_evaluation_utility": total_utility,
        "mean_evaluation_utility": mean,
        "final_best_utility": max(s["utility"] for s in samples),
        "wtb_modes": dict(modes),
        "rollout_decisions": len(nodes) - 1,
    }
    records[0]["at2po_tree"] = {
        **config.settings(),
        "samples": samples,
        "leaf_ids": leaf_ids,
        "leaf_utility_mean": mean,
        "leaf_utility_population_std": std,
        "training_occurrences": len(records),
        "unique_sampled_nodes": len(nodes) - 1,
        "nodes": [
            {
                "node": n.index,
                "parent": n.parent,
                "children": n.children,
                "state_path": list(n.state.path),
                "entropy": n.entropy,
                "value": n.value,
                "leaf": n.terminal is not None,
                "action": n.record["chosen_label"] if n.record else None,
            }
            for n in nodes
        ],
        "token_usage": usage,
    }
    # The common collection interface carries a representative terminal. Every
    # sampled leaf participates in training; no best-leaf actor path is selected.
    first = nodes[samples[0]["leaf_node"]].terminal
    return first, first.outcome(), records


def categorical_turn_loss(
    torch: Any, logprob: Any, old_logprob: float, advantage: float, config: AT2POConfig
) -> Any:
    # Specialization of upstream gspo_turn to one categorical action per turn.
    log_ratio = logprob - logprob.detach() + (logprob - old_logprob).detach()
    ratio = torch.exp(torch.clamp(log_ratio, max=10.0))
    return -torch.minimum(
        ratio * advantage,
        torch.clamp(ratio, 1 - config.clip_low, 1 + config.clip_high) * advantage,
    )


def update_at2po(
    policy: Any,
    records: list[dict[str, Any]],
    *,
    epochs: int,
    batch_size: int,
    config: AT2POConfig,
    execution: AT2POUpdateConfig = AT2POUpdateConfig(),
    dist: Any = None,
    progress: Any = None,
) -> dict[str, float | int]:
    from .at2po_update import update

    return update(
        policy, records, epochs=epochs, batch_size=batch_size, config=config,
        execution=execution, dist=dist, progress=progress,
    )
