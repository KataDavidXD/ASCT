from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
from uct_engine import QIDPrefixCache, add_token_usage

from atppo_lab.experiment import (
    CACHE_METHODS,
    DEFAULT_METHODS,
    TREE_METHODS,
    _summarize_search_tokens,
    _training_record,
)
from atppo_lab.vine import collect_vine_trajectory


@dataclass(frozen=True)
class State:
    qid: str = "q"
    path: tuple[str, ...] = ()

    @property
    def item(self):
        return SimpleNamespace(qid=self.qid)

    @property
    def stage(self):
        return SimpleNamespace(value=str(len(self.path)))

    @property
    def token_usage(self):
        return add_token_usage(
            {
                "generator_4b_input_tokens": 3 * len(self.path),
                "generator_4b_output_tokens": len(self.path),
            }
        )

    @property
    def qwen_generation_attempts(self):
        return len(self.path)

    @property
    def qwen_retry_count(self):
        return len(self.path) // 2

    @property
    def qwen_length_retry_count(self):
        return 0

    @property
    def qwen_fallback_count(self):
        return 0

    def is_terminal(self):
        return len(self.path) == 7

    def available_actions(self):
        return (
            ("query.identity", "query.decomposition"),
            ("retriever.bm25",),
            ("selection.relevance",),
            ("retrieve.k=3",),
            ("control.stop",),
            ("reranker.identity",),
            ("top_m=2", "top_m=4"),
            (),
        )[len(self.path)]

    def prompt(self):
        return "observed:" + "/".join(self.path)

    def preview_child(self, action):
        raise AssertionError("actor must see real observations")

    def child(self, action):
        assert action in self.available_actions()
        return replace(self, path=(*self.path, action))

    def outcome(self):
        assert self.is_terminal()
        utility = 0.2 + 0.4 * (self.path[-1] == "top_m=4")
        return SimpleNamespace(
            utility=utility,
            performance=utility + 0.1,
            normalized_execution_cost=1.0,
            token_usage=self.token_usage,
            qwen_generation_attempts=self.qwen_generation_attempts,
            qwen_retry_count=self.qwen_retry_count,
            qwen_length_retry_count=self.qwen_length_retry_count,
            qwen_fallback_count=self.qwen_fallback_count,
        )

    def snapshot(self):
        return {"qid": self.qid, "path": list(self.path)}

    def restore(self, data):
        return State(data["qid"], tuple(data["path"]))

    def path_costs(self):
        return (4.0,) * len(self.path)


class Policy:
    def __init__(self):
        self.actor_scoring_tokens = 0

    def distribution(self, prompt, actions):
        assert prompt.startswith("observed:")
        self.actor_scoring_tokens += 5
        p = [1 / len(actions)] * len(actions)
        return p, [math.log(x) for x in p]


class Cache:
    def __init__(self):
        self.entries = {}
        self.new_edges = 0

    def advance(self, state, action):
        key = (state.qid, (*state.path, action))
        hit = key in self.entries
        if not hit:
            self.entries[key] = state.child(action)
            self.new_edges += 1
        return self.entries[key], hit


def collect(cache=None, budget=12):
    policy = Policy()
    cache = cache or Cache()
    state, outcome, records = collect_vine_trajectory(
        State(),
        policy,
        rng=random.Random(11),
        search_seed=11,
        budget=budget,
        cache=cache,
    )
    return state, outcome, records, policy, cache


def test_budget_credit_and_observed_continuations():
    state, outcome, records, _, _ = collect()
    assert len(records) == len(state.path) == 7
    assert sum(len(row["vine_trace"]) for row in records) == 7 * 12
    for i, row in enumerate(records):
        assert row["vine_value"] == pytest.approx(
            sum(x["utility"] for x in row["vine_trace"]) / 12
        )
        successor = records[i + 1]["vine_value"] if i < 6 else 0
        reward = outcome.utility if i == 6 else 0
        assert row["advantage"] == pytest.approx(reward + successor - row["vine_value"])
        assert row["next_vine_value"] == successor
        assert all(
            x["decisions"][0]["path"] == row["state_path"] for x in row["vine_trace"]
        )
    assert sum(row["advantage"] for row in records) == pytest.approx(
        outcome.utility - records[0]["vine_value"]
    )
    assert sum(row["reward"] for row in records) == outcome.utility


def test_actor_rng_independent_of_auxiliary_budget():
    a = collect(budget=1)
    b = collect(budget=12)
    assert a[0].path == b[0].path
    assert [r["old_logprob"] for r in a[2]] == [r["old_logprob"] for r in b[2]]
    assert [r["vine_trace"] for r in b[2]] == [r["vine_trace"] for r in collect()[2]]


def test_token_ledger_counts_new_edges_and_aux_actor_once():
    _, _, records, policy, cache = collect()
    compact = [_training_record(row) for row in records]
    summary = _summarize_search_tokens(compact)
    main = summary["vine_actor_execution_token_usage"]
    assert (
        summary["qwen_tokens"]
        + main["generator_4b_input_tokens"]
        + main["generator_4b_output_tokens"]
        == cache.new_edges * 4
    )
    assert summary["actor_scoring_tokens"] == policy.actor_scoring_tokens
    assert summary["logical_evaluations"] == 84
    hits = [
        sample
        for row in records
        for sample in row["vine_trace"]
        if sample["wtb_mode"] == "full_hit"
    ]
    assert hits and all(
        sample["token_usage"]["generator_4b_input_tokens"] == 0 for sample in hits
    )
    assert any(
        sample["qwen_retry_count"] > 0
        for row in records
        for sample in row["vine_trace"]
    )
    assert len(compact) == 7 and all("vine_trace" not in row for row in compact)
    assert compact[0]["vine_value"] == records[0]["vine_value"]


def test_duplicate_terminal_samples_keep_multiplicity():
    records = collect(budget=24)[2]
    last = records[-1]
    assert len({tuple(x["terminal_path"]) for x in last["vine_trace"]}) <= 2
    assert last["vine_samples"] == 24
    assert last["vine_value"] == pytest.approx(
        sum(x["utility"] for x in last["vine_trace"]) / 24
    )


def test_cli_opt_in_and_original_defaults():
    from atppo_lab.main import _parser
    from atppo_lab.tiny_debug_parallel_resume import _parser as tiny_parser

    args = [
        "--model-path",
        "m",
        "--dataset-root",
        "d",
        "--output-dir",
        "o",
        "--devices",
        "cuda:0",
    ]
    assert tuple(_parser().parse_args(args).methods) == DEFAULT_METHODS
    for parser in (_parser(), tiny_parser()):
        assert parser.parse_args([*args, "--methods", "vineppo"]).methods == ["vineppo"]
    assert "vineppo" in CACHE_METHODS and "vineppo" not in TREE_METHODS
    with pytest.raises(ValueError):
        collect(budget=0)


def test_native_wtb_prefix_fork_hit_and_legacy_evaluate(tmp_path):
    pytest.importorskip("wtb")
    cache = QIDPrefixCache(tmp_path)
    try:
        root = State()
        cache.begin_qid(root)
        state, _, records, _, _ = collect(cache=cache, budget=2)
        restored = cache.materialized_state("q", state.path)
        assert restored.path == state.path
        assert (
            cache.evaluate(qid="q", path=state.path, path_costs=state.path_costs()).mode
            == "full_hit"
        )
        assert len(records) == 7
        cache.begin_qid(State("another"))
        with pytest.raises(ValueError):
            cache.materialized_state("q", state.path)
        # Original full-path evaluation and a subsequent actor prefix fork.
        cache.begin_qid(root)
        cache.evaluate(qid="q", path=state.path, path_costs=state.path_costs())
        parent = cache.materialized_state("q", state.path[:-1])
        other = "top_m=4" if state.path[-1] == "top_m=2" else "top_m=2"
        child, _ = cache.advance(parent, other)
        assert child.path == (*parent.path, other)
    finally:
        cache.close()
