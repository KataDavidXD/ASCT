"""Main launcher for the ASCT paper experiment."""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from uct_engine import (
    DEFAULT_SEARCH_COST_MODE,
    SEARCH_COST_MODES,
    add_token_usage,
    empty_token_usage,
    normalize_token_usage,
)

from .at2po import AT2POConfig, AT2POUpdateConfig
from .dataset import HotpotItem, load_experiment_splits, load_manifest_splits
from .distributed_cell import run_distributed_cell
from .experiment import DEFAULT_METHODS, METHODS, CellSpec, run_cell
from .ragspace import RAGSpace


@dataclass(frozen=True)
class RunConfig:
    model_path: Path
    rag_model_root: Path | None
    dataset_root: Path
    output_dir: Path
    train_size: int = 2000
    valid_size: int = 500
    test_size: int = 500
    split_seed: int = 2027
    methods: tuple[str, ...] = DEFAULT_METHODS
    seeds: tuple[int, ...] = (11, 23, 37)
    devices: tuple[str, ...] = ("cuda:0",)
    resume: bool = False
    max_new_train_items: int | None = None
    search_budget: int = 12
    token_beta: float = 1e-4
    search_cost_mode: str = DEFAULT_SEARCH_COST_MODE
    workflow_cost_lambda: float = 0.1
    execution_cost_reference: float = 4096.0
    iterations: int = 3
    epochs: int = 2
    ppo_batch_size: int = 8
    checkpoint_interval_qids: int = 100
    deployment_horizon: int = 100_000
    at2po_config: AT2POConfig = AT2POConfig()
    at2po_update_config: AT2POUpdateConfig = AT2POUpdateConfig()
    split_manifest: Path | None = None


def _prepare_index_worker(
    model_root: Path,
    index_root: Path,
    items: list[HotpotItem],
    device: str,
    result_queue: Any | None = None,
) -> dict[str, int]:
    usage = RAGSpace(model_root, device, index_root=index_root).prepare(items)
    if result_queue is not None:
        result_queue.put(usage)
    return usage


def _prepare_rag_index(
    config: RunConfig, splits: dict[str, list[HotpotItem]]
) -> dict[str, int]:
    items = [*splits["train"], *splits["valid"], *splits["test"]]
    index_root = config.dataset_root / "rag-index"
    missing = [item for item in items if not (index_root / f"{item.qid}.pt").exists()]
    if not missing:
        return empty_token_usage()
    model_root = (
        config.rag_model_root
        if config.rag_model_root is not None
        else config.model_path.parent / "paper-components"
    )
    if len(config.devices) == 1:
        return _prepare_index_worker(
            model_root, index_root, missing, config.devices[0]
        )
    context = get_context("spawn")
    result_queue = context.Queue()
    assignments = [
        (device, missing[rank :: len(config.devices)])
        for rank, device in enumerate(config.devices)
        if missing[rank :: len(config.devices)]
    ]
    processes = [
        context.Process(
            target=_prepare_index_worker,
            args=(model_root, index_root, items, device, result_queue),
        )
        for device, items in assignments
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failed = [process.exitcode for process in processes if process.exitcode != 0]
    if failed:
        raise RuntimeError(f"RAG cold indexing failed in workers: {failed}")
    return add_token_usage(*(result_queue.get() for _ in processes))


def _run_spec(
    spec: CellSpec,
    config: RunConfig,
    splits: dict[str, list[HotpotItem]],
) -> dict[str, Any]:
    checkpoint_path = (
        config.output_dir / f"{spec.method}-seed-{spec.seed}" / "checkpoint.pt"
    )
    arguments = dict(
        spec=spec,
        model_path=config.model_path,
        rag_model_root=(
            config.rag_model_root
            if config.rag_model_root is not None
            else config.model_path.parent / "paper-components"
        ),
        rag_index_root=config.dataset_root / "rag-index",
        output_dir=config.output_dir,
        train_items=splits["train"],
        valid_items=splits["valid"],
        test_items=splits["test"],
        resume=config.resume and checkpoint_path.exists(),
        max_new_train_items=config.max_new_train_items,
        search_budget=config.search_budget,
        token_beta=config.token_beta,
        search_cost_mode=config.search_cost_mode,
        workflow_cost_lambda=config.workflow_cost_lambda,
        execution_cost_reference=config.execution_cost_reference,
        iterations=config.iterations,
        epochs=config.epochs,
        ppo_batch_size=config.ppo_batch_size,
        checkpoint_interval_qids=config.checkpoint_interval_qids,
        at2po_config=config.at2po_config,
        at2po_update_config=config.at2po_update_config,
    )
    if len(config.devices) == 1:
        return run_cell(
            **arguments,
            device=config.devices[0],
        )
    return run_distributed_cell(
        **arguments,
        devices=config.devices,
    )


def _release_cuda(*, model_was_local: bool) -> None:
    gc.collect()
    if not model_was_local:
        return
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def run_experiment(config: RunConfig) -> list[dict[str, Any]]:
    if not config.devices:
        raise ValueError("at least one CUDA device is required")
    if len(set(config.devices)) != len(config.devices):
        raise ValueError("CUDA devices must be distinct")
    if len(set(config.methods) - set(METHODS)) > 0:
        raise ValueError(f"methods must be chosen from {METHODS}")
    if not math.isfinite(config.token_beta) or config.token_beta < 0:
        raise ValueError("token_beta must be finite and non-negative")
    if config.deployment_horizon <= 0:
        raise ValueError("deployment_horizon must be positive")
    if config.search_cost_mode not in SEARCH_COST_MODES:
        raise ValueError(f"search_cost_mode must be one of {SEARCH_COST_MODES}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.split_manifest is not None:
        splits = load_manifest_splits(config.dataset_root, config.split_manifest)
    else:
        splits = load_experiment_splits(
            config.dataset_root,
            train_size=config.train_size,
            valid_size=config.valid_size,
            test_size=config.test_size,
            seed=config.split_seed,
        )
    if tuple(len(splits[role]) for role in ("train", "valid", "test")) != (
        config.train_size, config.valid_size, config.test_size
    ):
        raise ValueError("split manifest sizes differ from requested train/valid/test sizes")
    split_ids = {role: [item.qid for item in items] for role, items in splits.items()}
    split_path = config.output_dir / "splits.json"
    if split_path.exists():
        existing = json.loads(split_path.read_text(encoding="utf-8"))
        if existing != split_ids:
            raise ValueError(
                "output directory contains different train/valid/test QIDs"
            )
    else:
        split_path.write_text(
            json.dumps(split_ids, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    index_usage = _prepare_rag_index(config, splits)
    index_trace_path = config.output_dir / "index_build_tokens.json"
    previous_index_usage = empty_token_usage()
    if index_trace_path.exists():
        previous_payload = json.loads(index_trace_path.read_text(encoding="utf-8"))
        previous_index_usage = normalize_token_usage(
            previous_payload.get("token_usage", previous_payload)
        )
    cumulative_index_usage = add_token_usage(previous_index_usage, index_usage)
    index_trace_path.write_text(
        json.dumps(
            {
                "token_usage": cumulative_index_usage,
                **cumulative_index_usage,
                "charged_to_train_search": False,
                "semantics": (
                    "document-side input tokens encoded by E5, Qwen3-Embedding-0.6B, "
                    "and BGE-M3 while physically building missing shared indexes"
                ),
            },
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    specs = [
        CellSpec(method, seed) for method in config.methods for seed in config.seeds
    ]
    results = []
    for spec in specs:
        results.append(_run_spec(spec, config, splits))
        _release_cuda(model_was_local=len(config.devices) == 1)
    if results and all(result.get("status") == "complete" for result in results):
        from .evaluation import write_evaluation

        write_evaluation(
            config.output_dir,
            methods=config.methods,
            seeds=config.seeds,
            deployment_horizon=config.deployment_horizon,
        )
    print(json.dumps(results, ensure_ascii=False, allow_nan=False, indent=2))
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ASCT paper experiment")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--rag-model-root",
        type=Path,
        help="directory containing E5, BGE-M3, Qwen3 embedding and reranker",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--valid-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--split-seed", type=int, default=2027)
    parser.add_argument(
        "--split-manifest", type=Path,
        help="exact ordered {train,valid,test} QIDs; overrides split-seed sampling",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=(*DEFAULT_METHODS, "vineppo", "at2po"),
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37])
    parser.add_argument("--devices", nargs="+", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-train-items", type=int)
    parser.add_argument("--search-budget", type=int, default=12)
    parser.add_argument(
        "--at2po-leaf-budget", type=int, default=22,
        help="terminal samples per QID tree, independent of --search-budget",
    )
    parser.add_argument(
        "--at2po-initial-rollouts", type=int,
        help="default: rounded 10/22 fraction of leaf budget",
    )
    parser.add_argument("--at2po-expansion-rounds", type=int, default=2)
    parser.add_argument("--at2po-clip-low", type=float, default=0.003)
    parser.add_argument("--at2po-clip-high", type=float, default=0.004)
    parser.add_argument("--at2po-learning-rate", type=float, default=1e-6)
    parser.add_argument(
        "--at2po-node-microbatch-size", type=int, default=2,
        help="AT2PO nodes per scoring/backward call per device; leaves/batch stay unchanged",
    )
    parser.add_argument(
        "--at2po-update-backend", choices=("parallel", "rank0"), default="parallel",
        help="AT2PO only: split each global leaf minibatch across the collection devices",
    )
    parser.add_argument("--beta-tok", type=float, default=1e-4)
    parser.add_argument(
        "--search-cost-mode",
        choices=SEARCH_COST_MODES,
        default=DEFAULT_SEARCH_COST_MODE,
    )
    parser.add_argument("--workflow-cost-lambda", type=float, default=0.1)
    parser.add_argument("--execution-cost-reference", type=float, default=4096.0)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--ppo-batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-interval-qids", type=int, default=100)
    parser.add_argument(
        "--deployment-horizon",
        type=int,
        default=100_000,
        help="N for EndJ_test_search = U_test - beta * C_train-search / N",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_experiment(
        RunConfig(
            model_path=args.model_path,
            rag_model_root=args.rag_model_root,
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            train_size=args.train_size,
            valid_size=args.valid_size,
            test_size=args.test_size,
            split_seed=args.split_seed,
            split_manifest=args.split_manifest,
            methods=tuple(args.methods),
            seeds=tuple(args.seeds),
            devices=tuple(args.devices),
            resume=args.resume,
            max_new_train_items=args.max_new_train_items,
            search_budget=args.search_budget,
            token_beta=args.beta_tok,
            search_cost_mode=args.search_cost_mode,
            workflow_cost_lambda=args.workflow_cost_lambda,
            execution_cost_reference=args.execution_cost_reference,
            iterations=args.iterations,
            epochs=args.epochs,
            ppo_batch_size=args.ppo_batch_size,
            checkpoint_interval_qids=args.checkpoint_interval_qids,
            deployment_horizon=args.deployment_horizon,
            at2po_config=AT2POConfig(
                leaf_budget=args.at2po_leaf_budget,
                initial_rollouts=args.at2po_initial_rollouts,
                expansion_rounds=args.at2po_expansion_rounds,
                clip_low=args.at2po_clip_low,
                clip_high=args.at2po_clip_high,
                learning_rate=args.at2po_learning_rate,
            ),
            at2po_update_config=AT2POUpdateConfig(
                node_microbatch_size=args.at2po_node_microbatch_size,
                backend=args.at2po_update_backend,
            ),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
