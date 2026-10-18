"""Real-model CF-Actor gate: balanced B=12, actor suffixes, PPO and resume."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path

from uct_engine import add_token_usage

from .cf_actor import CF_ACTOR_SETTINGS
from .main import RunConfig, run_experiment


def audit_traces(rows: list[dict], result: dict, budget: int) -> dict:
    modes: Counter[str] = Counter()
    logical = main_steps = qwen_tokens = auxiliary_scoring = 0
    fields = (
        "generator_4b_input_tokens",
        "generator_4b_output_tokens",
        "query_embedding_06b_input_tokens",
        "reranker_06b_input_tokens",
    )
    for row in rows:
        assert len(row["steps"]) == len(row["actor_path"])
        for step in row["steps"]:
            main_steps += 1
            assert step["chosen_label"] == row["actor_path"][step["step"]]
            assert len(step["search_trace"]) == sum(step["visit_counts"]) == budget
            assert min(step["visit_counts"]) >= 1
            assert max(step["visit_counts"]) - min(step["visit_counts"]) <= 1
            baseline = math.fsum(
                p * q
                for p, q in zip(step["probabilities"], step["q_utility"], strict=True)
            )
            chosen = step["labels"].index(step["chosen_label"])
            assert math.isclose(
                step["advantage"], step["q_utility"][chosen] - baseline, abs_tol=1e-12
            )
            for arm, action in enumerate(step["labels"]):
                samples = [
                    s for s in step["search_trace"] if s["root_action_index"] == arm
                ]
                assert len(samples) == step["visit_counts"][arm]
                assert all(s["root_action"] == action for s in samples)
                assert math.isclose(
                    step["q_utility"][arm],
                    math.fsum(s["utility"] for s in samples) / len(samples),
                    abs_tol=1e-12,
                )
            usage = add_token_usage(*(s["token_usage"] for s in step["search_trace"]))
            assert usage == step["search_token_metrics"]["token_usage"]
            qwen_tokens += sum(usage[key] for key in fields)
            auxiliary_scoring += usage["actor_scoring_tokens"]
            for sample in step["search_trace"]:
                logical += 1
                modes[sample["wtb_mode"]] += 1
                if sample["wtb_mode"] == "full_hit":
                    assert all(sample["token_usage"][key] == 0 for key in fields)
                for decision in sample["rollout_decisions"]:
                    assert (
                        decision["policy"] == "frozen_actor"
                        and decision["materialized"]
                    )
                    assert len(decision["path"]) > step["step"]
                    assert len(decision["labels"]) == len(decision["probabilities"])
                    assert math.isclose(
                        sum(decision["probabilities"]), 1.0, abs_tol=1e-6
                    )
                    assert (
                        decision["probabilities"][
                            decision["labels"].index(decision["chosen_label"])
                        ]
                        > 0
                    )
    iterations = result["iterations"]
    assert (
        sum(i["search_tokens"]["logical_evaluations"] for i in iterations)
        == logical
        == budget * main_steps
    )
    assert sum(i["search_tokens"]["qwen_tokens"] for i in iterations) == qwen_tokens
    assert (
        sum(
            i["search_tokens"]["cf_actor_auxiliary_actor_scoring_tokens"]
            for i in iterations
        )
        == auxiliary_scoring
    )
    assert sum(i["optimizer"]["samples"] for i in iterations) == main_steps
    assert all(i["optimizer"]["optimizer_steps"] > 0 for i in iterations)
    return {
        "main_training_actions": main_steps,
        "logical_auxiliary_evaluations": logical,
        "wtb_modes": dict(modes),
        "search_qwen_tokens": qwen_tokens,
        "auxiliary_actor_scoring_tokens": auxiliary_scoring,
        "optimizer_steps": [i["optimizer"]["optimizer_steps"] for i in iterations],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--rag-model-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("tiny gate requires a fresh output directory")
    config = RunConfig(
        model_path=args.model_path,
        rag_model_root=args.rag_model_root,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        train_size=2,
        valid_size=1,
        test_size=1,
        methods=("cf_actor",),
        seeds=(11,),
        devices=(args.device,),
        search_budget=12,
        iterations=2,
        epochs=2,
        checkpoint_interval_qids=1,
    )
    paused = run_experiment(replace(config, max_new_train_items=3))[0]
    assert paused["status"] == "paused" and paused["iteration"] == 1
    import torch

    cell = args.output_dir / "cf_actor-seed-11"
    before = torch.load(cell / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert len(before["iteration_results"]) == len(before["completed_qids"]) == 1
    assert all(
        before["algorithm_settings"][key] == value
        for key, value in CF_ACTOR_SETTINGS.items()
    )
    assert before["optimizer"]["state"]
    assert any(
        bool(v.abs().max() > 0) for k, v in before["adapter"].items() if "lora_B" in k
    )
    result = run_experiment(replace(config, resume=True))[0]
    assert result["status"] == "complete" and result["completed_iterations"] == 2
    rows = [
        json.loads(line)
        for line in (cell / "trajectories.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == len({(r["iteration"], r["qid"]) for r in rows}) == 4
    restored = next(
        r
        for r in rows
        if r["iteration"] == 1 and r["qid"] == before["completed_qids"][0]
    )
    assert [s["advantage"] for s in restored["steps"]] == [
        s["advantage"] for s in before["records"]
    ]
    assert len(result["valid"]["rows"]) == len(result["test"]["rows"]) == 1
    assert all("search_trace" not in row for row in result["test"]["rows"])
    receipt = {
        "status": "passed",
        "method": "cf_actor",
        "real_model_run": True,
        "train_valid_test": [2, 1, 1],
        "iterations": 2,
        "seed": 11,
        "budget": 12,
        "device": args.device,
        "resume_after_first_update": True,
        "cf_actor_settings": CF_ACTOR_SETTINGS,
        "torch": torch.__version__,
        **audit_traces(rows, result, config.search_budget),
    }
    (args.output_dir / "tiny_cf_actor_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
