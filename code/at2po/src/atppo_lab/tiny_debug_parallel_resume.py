"""10/2/2 real-model interruption-and-resume gate for the main launcher."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from uct_engine import DEFAULT_SEARCH_COST_MODE, SEARCH_COST_MODES

from .experiment import DEFAULT_METHODS
from .main import RunConfig, run_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the real 10/2/2 parallel resume gate"
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--rag-model-root", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=(*DEFAULT_METHODS, "vineppo"),
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument("--search-budget", type=int, default=12)
    parser.add_argument("--beta-tok", type=float, default=1e-4)
    parser.add_argument(
        "--search-cost-mode",
        choices=SEARCH_COST_MODES,
        default=DEFAULT_SEARCH_COST_MODE,
    )
    parser.add_argument("--workflow-cost-lambda", type=float, default=0.1)
    parser.add_argument("--execution-cost-reference", type=float, default=4096.0)
    return parser


def _cell_is_complete(cell_dir: Path) -> bool:
    if (cell_dir / "metrics.json").exists():
        return True
    result_path = cell_dir / "run_result.json"
    return result_path.exists() and json.loads(
        result_path.read_text(encoding="utf-8")
    ).get("status") == "complete"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if len(args.devices) != 2 or len(set(args.devices)) != 2:
        raise ValueError(
            "tiny_debug_parallel_resume requires exactly two distinct CUDA devices"
        )
    config = RunConfig(
        model_path=args.model_path,
        rag_model_root=args.rag_model_root,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        train_size=10,
        valid_size=2,
        test_size=2,
        methods=tuple(args.methods),
        seeds=(11,),
        devices=tuple(args.devices),
        search_budget=args.search_budget,
        token_beta=args.beta_tok,
        search_cost_mode=args.search_cost_mode,
        workflow_cost_lambda=args.workflow_cost_lambda,
        execution_cost_reference=args.execution_cost_reference,
    )
    missing_methods = tuple(
        method
        for method in args.methods
        if not _cell_is_complete(args.output_dir / f"{method}-seed-11")
        and not (args.output_dir / f"{method}-seed-11" / "checkpoint.pt").exists()
    )
    if missing_methods:
        paused = run_experiment(
            replace(
                config,
                methods=missing_methods,
                resume=False,
                max_new_train_items=5,
            )
        )
        if any(result.get("status") != "paused" for result in paused):
            raise RuntimeError("debug interruption phase did not pause every cell")
        for result in paused:
            if (
                result.get("world_size") != 2
                or result.get("optimizer_update_count") != 0
            ):
                raise RuntimeError(
                    "debug interruption must pause two ranks before PPO update"
                )
            if sorted(result.get("rank_completed", [])) != [2, 3]:
                raise RuntimeError(
                    "first five debug QIDs were not split 3/2 across two ranks"
                )
    completed = run_experiment(replace(config, resume=True))
    if any(result.get("status") != "complete" for result in completed):
        raise RuntimeError("debug resume phase did not complete every cell")
    if [result.get("method") for result in completed] != list(args.methods):
        raise RuntimeError("debug methods did not run in the requested order")
    for method in args.methods:
        cell_dir = args.output_dir / f"{method}-seed-11"
        event_path = cell_dir / "events.jsonl"
        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
        ]
        if not any(event.get("event") == "resumed" for event in events):
            raise RuntimeError(f"{method} did not exercise checkpoint resume")
        updates = [
            event for event in events if event.get("event") == "ppo_update_committed"
        ]
        if len(updates) != config.iterations:
            raise RuntimeError(
                f"{method} did not commit exactly one PPO update per iteration"
            )
        for iteration in range(config.iterations):
            rank_indices = [set(), set()]
            for path in sorted(
                (
                    cell_dir
                    / "shards"
                    / f"iteration-{iteration:03d}"
                    / "train"
                ).glob("item-*.json")
            ):
                payload = json.loads(path.read_text(encoding="utf-8"))
                rank = int(payload["rank"])
                index = int(payload["dataset_index"])
                if rank != index % 2:
                    raise RuntimeError(
                        f"{method} iteration {iteration} item {index} "
                        "was written by the wrong rank"
                    )
                if int(payload["actor_version"]) != iteration:
                    raise RuntimeError(
                        f"{method} iteration {iteration} used the wrong Actor version"
                    )
                rank_indices[rank].add(index)
            if rank_indices[0] & rank_indices[1] or rank_indices[0] | rank_indices[
                1
            ] != set(range(10)):
                raise RuntimeError(
                    f"{method} iteration {iteration} QID shards overlap or have gaps"
                )
            splits = (
                ("valid", "test")
                if iteration == config.iterations - 1
                else ("valid",)
            )
            for split in splits:
                evaluation_indices = [set(), set()]
                for rank in range(2):
                    path = (
                        cell_dir
                        / "shards"
                        / f"iteration-{iteration:03d}"
                        / split
                        / f"rank-{rank:03d}.json"
                    )
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    for row in payload["rows"]:
                        index = int(row["dataset_index"])
                        if index % 2 != rank:
                            raise RuntimeError(
                                f"{method} iteration {iteration} {split} item "
                                f"{index} was written by the wrong rank"
                            )
                        evaluation_indices[rank].add(index)
                if (
                    evaluation_indices[0] & evaluation_indices[1]
                    or evaluation_indices[0] | evaluation_indices[1]
                    != set(range(2))
                ):
                    raise RuntimeError(
                        f"{method} iteration {iteration} {split} shards "
                        "overlap or have gaps"
                    )
    print(
        json.dumps(
            {
                "status": "debug_passed",
                "methods": list(args.methods),
                "devices": list(args.devices),
                "train": 10,
                "valid": 2,
                "test": 2,
                "iterations": config.iterations,
                "search_cost_mode": config.search_cost_mode,
                "token_beta": config.token_beta,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
