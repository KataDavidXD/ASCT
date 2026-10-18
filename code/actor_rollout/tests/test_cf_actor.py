"""CF-Actor's controlled allocation, credit, randomness and physical ledger."""

import math
import random
from dataclasses import dataclass

import pytest
from uct_engine import QIDPrefixCache, add_token_usage

from atppo_lab.cf_actor import CF_ACTOR_SETTINGS, collect_cf_actor_trajectory
from atppo_lab.experiment import (
    CACHE_METHODS,
    DEFAULT_METHODS,
    TREE_METHODS,
    _collect_trajectory,
    _summarize_search_tokens,
    _training_record,
)
from tests.test_vine import Cache, Policy, State


def collect(*, cache=None, budget=12, policy=None, root=None, search_seed=11):
    policy = policy or Policy()
    cache = cache or Cache()
    state, outcome, records = collect_cf_actor_trajectory(
        root or State(),
        policy,
        rng=random.Random(11),
        search_seed=search_seed,
        budget=budget,
        cache=cache,
    )
    return state, outcome, records, policy, cache


@pytest.mark.parametrize("budget", [2, 12, 13])
def test_balanced_root_coverage_q_means_and_centered_main_credit(budget):
    state, _, records, _, _ = collect(budget=budget)
    assert len(records) == len(state.path) == 7
    for row in records:
        assert sum(row["visit_counts"]) == len(row["search_trace"]) == budget
        assert min(row["visit_counts"]) >= 1
        assert max(row["visit_counts"]) - min(row["visit_counts"]) <= 1
        for index, action in enumerate(row["labels"]):
            samples = [s for s in row["search_trace"] if s["root_action"] == action]
            assert len(samples) == row["visit_counts"][index]
            assert row["q_utility"][index] == pytest.approx(
                sum(s["utility"] for s in samples) / len(samples)
            )
            for sample in samples:
                assert sample["terminal_path"][row["step"]] == action
                assert sample["root_action_index"] == index
                for decision in sample["rollout_decisions"]:
                    assert len(decision["path"]) > row["step"]
                    assert (
                        decision["policy"] == "frozen_actor"
                        and decision["materialized"]
                    )
        baseline = sum(p * q for p, q in zip(row["probabilities"], row["q_utility"]))
        assert row["v_tree_baseline"] == pytest.approx(baseline)
        assert row["advantage"] == pytest.approx(row["q_raw_chosen"] - baseline)
        assert sum(
            p * (q - baseline) for p, q in zip(row["probabilities"], row["q_utility"])
        ) == pytest.approx(0)
        assert row["chosen_label"] == state.path[row["step"]]


class LastActionActor(Policy):
    def distribution(self, prompt, actions):
        assert prompt.startswith("observed:")
        self.actor_scoring_tokens += 5
        return [0.0] * (len(actions) - 1) + [1.0], [-1e9] * (len(actions) - 1) + [0.0]


def test_actor_controls_every_suffix_even_when_root_forces_zero_probability_arm():
    state, _, records, _, _ = collect(policy=LastActionActor())
    assert state.path[0] == "query.decomposition" and state.path[-1] == "top_m=4"
    first = records[0]
    assert first["visit_counts"] == [6, 6]
    assert any(s["root_action"] == "query.identity" for s in first["search_trace"])
    for row in records:
        for sample in row["search_trace"]:
            for decision in sample["rollout_decisions"]:
                assert decision["chosen_label"] == decision["labels"][-1]
                assert decision["probabilities"][-1] == 1
    last = records[-1]
    assert last["q_utility"] == pytest.approx([0.2, 0.6])
    assert last["v_tree_baseline"] == pytest.approx(0.6)
    assert last["v_tree_baseline"] != pytest.approx(sum(last["q_utility"]) / 2)
    assert last["advantage"] == pytest.approx(0)


def test_main_rng_is_independent_of_auxiliary_budget_and_seed():
    a, b, c = collect(budget=2), collect(budget=24), collect(search_seed=37)
    assert a[0].path == b[0].path == c[0].path
    assert [r["old_logprob"] for r in a[2]] == [r["old_logprob"] for r in b[2]]
    assert [r["search_trace"] for r in b[2]] == [
        r["search_trace"] for r in collect(budget=24)[2]
    ]
    assert [r["search_trace"] for r in b[2]] != [
        r["search_trace"] for r in collect(budget=24, search_seed=37)[2]
    ]


def test_component_tokens_retries_full_hits_and_main_only_training_records():
    _, _, records, policy, cache = collect()
    compact = [_training_record(row) for row in records]
    summary = _summarize_search_tokens(compact)
    assert summary["logical_evaluations"] == 84
    assert len(compact) == 7 and all("search_trace" not in r for r in compact)
    assert summary["qwen_tokens"] == cache.new_edges * 4
    assert summary["actor_scoring_tokens"] == policy.actor_scoring_tokens
    assert summary["cf_actor_auxiliary_actor_scoring_tokens"] > 0
    assert sum(summary["cf_actor_actor_execution_token_usage"].values()) == 0
    assert all(r["actor_execution"]["cache_hit"] for r in records)
    samples = [sample for row in records for sample in row["search_trace"]]
    assert (
        summary["qwen_retry_count"] == sum(s["qwen_retry_count"] for s in samples) > 0
    )
    assert any(s["wtb_mode"] == "partial_resume" for s in samples)
    hits = [s for s in samples if s["wtb_mode"] == "full_hit"]
    assert hits and all(
        s["token_usage"]["generator_4b_input_tokens"] == 0 for s in hits
    )
    for row in records:
        assert row["search_token_metrics"]["token_usage"] == add_token_usage(
            *(s["token_usage"] for s in row["search_trace"])
        )


def test_warm_cache_preserves_logical_sample_multiplicity_and_credit():
    cache = Cache()
    cold, warm = collect(cache=cache, budget=24), collect(cache=cache, budget=24)
    assert [r["q_utility"] for r in cold[2]] == [r["q_utility"] for r in warm[2]]
    summary = _summarize_search_tokens([_training_record(r) for r in warm[2]])
    assert summary["logical_evaluations"] == 168 and summary["qwen_tokens"] == 0
    assert summary["wtb_modes"] == {"full_hit": 168}
    assert len(warm[2][-1]["search_trace"]) == 24
    assert len({tuple(s["terminal_path"]) for s in warm[2][-1]["search_trace"]}) == 2


def test_native_wtb_materialization_and_collection_routing(tmp_path):
    pytest.importorskip("wtb")
    cache = QIDPrefixCache(tmp_path)
    try:
        cache.begin_qid(State())
        state, _, records = _collect_trajectory(
            State(),
            Policy(),
            method="cf_actor",
            rng=random.Random(11),
            search_budget=12,
            token_beta=1e-4,
            search_cost_mode="agentuct-Bellman",
            search_seed=11,
            cache=cache,
        )
        assert cache.materialized_state("q", state.path).path == state.path
        assert len(records) == 7
        with pytest.raises(ValueError, match="sampled actor"):
            _collect_trajectory(
                State(),
                Policy(),
                method="cf_actor",
                rng=random.Random(11),
                search_budget=12,
                token_beta=1e-4,
                search_cost_mode="agentuct-Bellman",
                search_seed=11,
                cache=cache,
                greedy=True,
            )
    finally:
        cache.close()


def test_cli_is_opt_in_and_existing_defaults_are_preserved():
    from atppo_lab.main import _parser

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
    assert _parser().parse_args([*args, "--methods", "cf_actor"]).methods == [
        "cf_actor"
    ]
    assert "cf_actor" in CACHE_METHODS and "cf_actor" not in TREE_METHODS
    assert (
        CF_ACTOR_SETTINGS["cf_actor_training_samples"] == "actor_main_trajectory_only"
    )


@pytest.mark.parametrize("budget", [0, -1, 1, 1.5, True])
def test_budget_validation(budget):
    with pytest.raises(ValueError):
        collect(budget=budget)


def test_nonfinite_outcome_is_rejected():
    @dataclass(frozen=True)
    class BadState(State):
        def outcome(self):
            outcome = super().outcome()
            outcome.utility = math.nan
            return outcome

    with pytest.raises(ValueError, match="non-finite"):
        collect(root=BadState())


def test_distributed_checkpoint_records_the_same_cf_actor_contract():
    from types import SimpleNamespace

    from atppo_lab.at2po import AT2POConfig
    from atppo_lab.distributed_cell import _algorithm_settings
    from atppo_lab.experiment import CellSpec

    config = SimpleNamespace(
        spec=CellSpec("cf_actor", 11),
        search_budget=12,
        token_beta=1e-4,
        search_cost_mode="agentuct-Bellman",
        workflow_cost_lambda=0.1,
        execution_cost_reference=4096,
        iterations=3,
        epochs=2,
        ppo_batch_size=8,
        world_size=2,
        at2po_config=AT2POConfig(),
    )
    settings = _algorithm_settings(config)
    assert all(settings[key] == value for key, value in CF_ACTOR_SETTINGS.items())
    assert "at2po_schema" not in settings and "vine_schema" not in settings
