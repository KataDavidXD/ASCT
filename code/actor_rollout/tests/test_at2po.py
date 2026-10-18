from __future__ import annotations

import json
import math
import random
from types import SimpleNamespace

import pytest

from atppo_lab.at2po import (
    AT2POConfig,
    _backup,
    _expansion_nodes,
    _Node,
    categorical_turn_loss,
    collect_at2po_tree,
    update_at2po,
)
from atppo_lab.at2po_budget import estimate_budget
from atppo_lab.experiment import (
    CACHE_METHODS,
    DEFAULT_METHODS,
    TREE_METHODS,
    _summarize_search_tokens,
    _training_record,
)
from tests.test_vine import Cache, Policy, State


def collect(budget=22, cache=None, state=None, policy=None):
    policy = policy or Policy()
    cache = cache or Cache()
    result = collect_at2po_tree(
        state or State(),
        policy,
        rng=random.Random(11),
        cache=cache,
        config=AT2POConfig(leaf_budget=budget),
    )
    return *result, policy, cache


def test_official_and_scaled_schedules():
    config = AT2POConfig()
    assert config.initial == 10 and config.beams == (6, 6)
    assert AT2POConfig(leaf_budget=106).initial == 48
    assert AT2POConfig(leaf_budget=106).beams == (29, 29)
    for budget in range(2, 130):
        config = AT2POConfig(leaf_budget=budget)
        assert config.initial + sum(config.beams) == budget
    for kwargs in (
        {"leaf_budget": 1},
        {"leaf_budget": 2.5},
        {"initial_rollouts": 23},
        {"expansion_rounds": -1},
        {"expansion_rounds": 0},
        {"clip_low": float("nan")},
        {"learning_rate": 0},
    ):
        with pytest.raises(ValueError):
            AT2POConfig(**kwargs)


def test_leaf_budget_multiplicity_trace_and_tokens():
    _, _, records, policy, cache = collect()
    tree = records[0]["at2po_tree"]
    assert len(tree["samples"]) == len(tree["leaf_ids"]) == 22
    assert len({tuple(s["terminal_path"]) for s in tree["samples"]}) <= 4
    assert len({r["at2po_sequence"] for r in records}) == 22
    assert len(records) == 22 * 7
    assert (
        sum(
            r.get("search_token_metrics", {}).get("logical_evaluations", 0)
            for r in records
        )
        == 22
    )
    summary = _summarize_search_tokens([_training_record(r) for r in records])
    assert summary["qwen_tokens"] == cache.new_edges * 4
    assert summary["actor_scoring_tokens"] == policy.actor_scoring_tokens
    assert summary["qwen_retry_count"] > 0
    assert sum(summary["wtb_modes"].values()) == 22
    assert (
        sum(s["token_usage"]["generator_4b_input_tokens"] for s in tree["samples"])
        == cache.new_edges * 3
    )
    assert summary["searches"] == 1  # a QID tree, not one search per training turn
    assert all("at2po_tree" not in _training_record(r) for r in records)
    assert all(
        r["advantage"] == tree["nodes"][r["at2po_node"]]["value"] for r in records
    )
    assert tree == collect()[2][0]["at2po_tree"]


def test_cache_hit_keeps_logical_leaves_but_zero_environment_tokens():
    cache = Cache()
    collect(cache=cache)
    _, _, records, _, _ = collect(cache=cache)
    summary = _summarize_search_tokens(records)
    assert summary["logical_evaluations"] == 22
    assert summary["qwen_tokens"] == 0
    assert summary["wtb_modes"] == {"full_hit": 22}
    assert summary["actor_scoring_tokens"] > 0


def test_softmax_backup_population_leaf_norm_and_constant_reward():
    outcome = lambda reward: SimpleNamespace(
        outcome=lambda: SimpleNamespace(utility=reward)
    )
    nodes = [
        _Node(0, None, None, children=[1, 2]),
        _Node(1, 0, None, terminal=outcome(1.0), entropy=0.0),
        _Node(2, 0, None, terminal=outcome(3.0), entropy=math.log(3.0)),
    ]
    mean, std = _backup(nodes)
    assert mean == 2 and std == 1
    assert nodes[0].value == pytest.approx(0.5 / 1.000001)
    for n in nodes[1:]:
        n.terminal = outcome(4.0)
    _backup(nodes)
    assert all(n.value == 0 for n in nodes)


def test_expansion_snapshot_ties_sibling_penalty_and_candidate_repeat():
    nodes = [
        _Node(0, None, None, children=[1, 2]),
        _Node(1, 0, None, entropy=1.0, children=[3]),
        _Node(2, 0, None, entropy=1.0),
        _Node(3, 1, None, terminal=object(), entropy=100),
    ]
    selected = _expansion_nodes(nodes, 5)
    assert [i for i, _ in selected] == [1, 2, 0, 1, 2]
    assert selected[0][1] == pytest.approx(0.95)
    assert all(i != 3 for i, _ in selected)


def test_forks_regenerate_selected_action_and_shared_prefix_credit():
    records = collect()[2]
    tree = records[0]["at2po_tree"]
    for sample in tree["samples"]:
        selected = tree["nodes"][sample["selected_node"]]
        assert sample["fork_state_path"] == selected["state_path"]
        first = tree["nodes"][sample["new_node_ids"][0]]
        assert first["parent"] == (selected["parent"] or 0)
    occurrences = {}
    for record in records:
        occurrences.setdefault(record["at2po_node"], []).append(record)
    shared = [group for group in occurrences.values() if len(group) > 1]
    assert shared
    assert all(len({r["advantage"] for r in group}) == 1 for group in shared)


@pytest.mark.parametrize("advantage", [-0.3, 0.0, 0.4])
@pytest.mark.parametrize("ratio", [0.5, 0.998, 1.0, 1.002, 1.5])
def test_categorical_turn_ratio_value_and_gradient(advantage, ratio):
    torch = pytest.importorskip("torch")
    x = torch.tensor(math.log(ratio) - 1, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    config = AT2POConfig()
    actual = categorical_turn_loss(torch, x, -1.0, advantage, config)
    # Literal one-token specialization of upstream Eq.5 and clipped objective.
    r = torch.exp(y - (-1.0))
    expected = torch.maximum(-advantage * r, -advantage * torch.clamp(r, 0.997, 1.004))
    actual.backward()
    expected.backward()
    assert actual.item() == pytest.approx(expected.item())
    assert x.grad.item() == pytest.approx(y.grad.item())


def test_update_leaf_weighting_shared_prefix_folding_and_no_whitening():
    torch = pytest.importorskip("torch")

    class Learner:
        def __init__(self):
            self.torch = torch
            self.model = torch.nn.Linear(1, 2, bias=False, dtype=torch.float64)
            self.model.weight.data.zero_()
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
            self.actor_scoring_tokens = 0

        def batch_action_logprobs(self, prompts, labels, grad):
            self.actor_scoring_tokens += len(prompts)
            return [self.model(torch.ones(1, dtype=torch.float64)) for _ in prompts]

    policy = Learner()
    baseline = Learner()
    records = []
    for seq, node_ids in ((10, [1, 2]), (11, [1, 3, 4])):
        for node in node_ids:
            records.append(
                {
                    "qid": "q",
                    "at2po_sequence": seq,
                    "at2po_sequence_length": len(node_ids),
                    "at2po_node": node,
                    "prompt": "p",
                    "labels": ["a", "b"],
                    "chosen_label": "a",
                    "old_logprob": -math.log(2),
                    "advantage": 0.0 if node == 2 else 0.1 * node,
                }
            )
    loss = 0
    for record in records:
        scores = baseline.model(torch.ones(1, dtype=torch.float64))
        logprob = torch.log_softmax(scores, 0)[0]
        loss = loss + categorical_turn_loss(
            torch, logprob, record["old_logprob"], record["advantage"], AT2POConfig()
        ) / (2 * record["at2po_sequence_length"])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(baseline.model.parameters(), 1)
    baseline.optimizer.step()
    telemetry = update_at2po(
        policy, records, epochs=1, batch_size=2, config=AT2POConfig()
    )
    torch.testing.assert_close(policy.model.weight, baseline.model.weight)
    assert telemetry["leaf_sequences"] == 2 and telemetry["samples"] == 5
    assert telemetry["actor_scoring_tokens"] == 3  # shared node once; zero skipped
    assert telemetry["optimizer_steps"] == 1
    assert telemetry["mean_loss"] == pytest.approx(loss.item())
    with pytest.raises(ValueError, match="complete leaf"):
        update_at2po(policy, records[:-1], epochs=1, batch_size=2, config=AT2POConfig())


def test_cli_defaults_and_checkpoint_settings():
    from atppo_lab.distributed_cell import _algorithm_settings
    from atppo_lab.main import _parser

    args = _parser().parse_args(
        [
            "--model-path",
            "m",
            "--dataset-root",
            "d",
            "--output-dir",
            "o",
            "--devices",
            "cuda:0",
        ]
    )
    assert tuple(args.methods) == DEFAULT_METHODS
    assert args.at2po_leaf_budget == 22 and args.search_budget == 12
    assert "at2po" in CACHE_METHODS and "at2po" not in TREE_METHODS
    cfg = SimpleNamespace(
        search_budget=12,
        token_beta=1e-4,
        search_cost_mode="x",
        workflow_cost_lambda=0.1,
        execution_cost_reference=4096,
        iterations=3,
        epochs=2,
        ppo_batch_size=8,
        world_size=2,
        spec=SimpleNamespace(method="at2po"),
        at2po_config=AT2POConfig(),
    )
    assert _algorithm_settings(cfg)["at2po_leaf_budget"] == 22
    old = _algorithm_settings(cfg)
    cfg.at2po_config = AT2POConfig(leaf_budget=106)
    assert old != _algorithm_settings(cfg)


def test_budget_estimate_uses_train_only_and_rejects_duplicates(tmp_path):
    path = tmp_path / "metrics.json"
    data = {
        "method": "uct_atppo",
        "seed": 11,
        "status": "complete",
        "completed_iterations": 1,
        "test": {"mean_utility": -999},
        "iterations": [
            {
                "iteration": 0,
                "train_qids": ["a", "b"],
                "search_tokens": {"logical_evaluations": 210},
            }
        ],
    }
    path.write_text(json.dumps(data))
    result = estimate_budget([path])
    assert result["recommended_leaf_budget"] == 105
    assert result["reference_mean_evaluations_per_qid"] == 105
    data["test"]["mean_utility"] = 999
    path.write_text(json.dumps(data))
    assert estimate_budget([path])["recommended_leaf_budget"] == 105
    with pytest.raises(ValueError, match="duplicate"):
        estimate_budget([path, path])


def test_native_wtb_cache_for_at2po(tmp_path):
    pytest.importorskip("wtb")
    from uct_engine import QIDPrefixCache

    cache = QIDPrefixCache(tmp_path)
    try:
        cache.begin_qid(State())
        _, _, records, _, _ = collect(cache=cache, budget=6)
        assert len(records[0]["at2po_tree"]["samples"]) == 6
        assert _summarize_search_tokens(records)["logical_evaluations"] == 6
    finally:
        cache.close()


def test_manifest_preserves_qid_order_and_detects_overlap(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from atppo_lab.dataset import load_manifest_splits

    root = tmp_path / "fullwiki"
    root.mkdir()
    rows = [
        {
            "id": str(i),
            "question": "q",
            "answer": "a",
            "context": {"title": [], "sentences": []},
            "supporting_facts": {"title": []},
        }
        for i in range(4)
    ]
    pq.write_table(pa.Table.from_pylist(rows), root / "train-000.parquet")
    manifest = tmp_path / "splits.json"
    manifest.write_text(
        json.dumps({"train": ["2", "0"], "valid": ["3"], "test": ["1"]})
    )
    result = load_manifest_splits(tmp_path, manifest)
    assert [i.qid for i in result["train"]] == ["2", "0"]
    assert result["valid"][0].context_titles == ()  # source kept intact
    manifest.write_text(
        json.dumps({"train": ["2", "0"], "valid": ["3"], "test": ["2"]})
    )
    with pytest.raises(ValueError, match="repeat"):
        load_manifest_splits(tmp_path, manifest)
