"""One online PPO cell collected across several CUDA devices."""

from __future__ import annotations

import json
import os
import random
import socket
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing import get_context
from multiprocessing.connection import wait
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from uct_engine import QIDPrefixCache

from .at2po import AT2POConfig, AT2POUpdateConfig, update_at2po
from .dataset import HotpotItem
from .experiment import (
    CACHE_METHODS,
    METHODS,
    CellSpec,
    _collect_item,
    _evaluation_rows,
    _summarize_evaluation,
    _summarize_search_tokens,
    _update,
)
from .policy import PlannerPolicy
from .rag import JSONTemplateGenerator, RAGComponents
from .ragspace import RAGSpace
from .vine import VINE_SETTINGS

_PROCESS_GROUP_TIMEOUT = timedelta(hours=1)
_RANK0_UPDATE_WAIT_SECONDS = 6 * 60 * 60


@dataclass(frozen=True)
class _WorkerConfig:
    spec: CellSpec
    model_path: Path
    rag_model_root: Path
    rag_index_root: Path
    output_dir: Path
    device: str
    rank: int
    world_size: int
    master_port: int
    resume: bool
    max_new_train_items: int | None
    search_budget: int
    token_beta: float
    search_cost_mode: str
    workflow_cost_lambda: float
    execution_cost_reference: float
    iterations: int
    epochs: int
    ppo_batch_size: int
    checkpoint_interval_qids: int
    train_count: int
    train_qids: tuple[str, ...]
    at2po_config: AT2POConfig = AT2POConfig()
    at2po_update_config: AT2POUpdateConfig = AT2POUpdateConfig()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def _train_path(cell_dir: Path, iteration: int, index: int) -> Path:
    return (
        cell_dir
        / "shards"
        / f"iteration-{iteration:03d}"
        / "train"
        / f"item-{index:06d}.json"
    )


def _evaluation_path(cell_dir: Path, iteration: int, split: str, rank: int) -> Path:
    return (
        cell_dir
        / "shards"
        / f"iteration-{iteration:03d}"
        / split
        / f"rank-{rank:03d}.json"
    )


def _algorithm_settings(config: _WorkerConfig) -> dict[str, float | int | str]:
    return {
        "search_budget": config.search_budget,
        "token_beta": config.token_beta,
        "search_cost_mode": config.search_cost_mode,
        "workflow_cost_lambda": config.workflow_cost_lambda,
        "execution_cost_reference": config.execution_cost_reference,
        "iterations": config.iterations,
        "epochs": config.epochs,
        "ppo_batch_size": config.ppo_batch_size,
        "world_size": config.world_size,
        **(VINE_SETTINGS if config.spec.method == "vineppo" else {}),
        **(config.at2po_config.settings() if config.spec.method == "at2po" else {}),
    }


def _checkpoint_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in (
            "iteration",
            "phase",
            "completed_indices",
            "iteration_results",
            "optimizer_update_count",
            "optimizer_telemetry",
            "search_tokens",
        )
    }


def _completed_result(
    config: _WorkerConfig,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    iteration_results = list(metadata["iteration_results"])
    if len(iteration_results) != config.iterations:
        raise RuntimeError("complete checkpoint has incomplete iteration results")
    if int(metadata["optimizer_update_count"]) != config.iterations:
        raise RuntimeError("complete checkpoint has the wrong PPO update count")
    return {
        "method": config.spec.method,
        "seed": config.spec.seed,
        "status": "complete",
        **({"at2po_settings": config.at2po_config.settings()} if config.spec.method == "at2po" else {}),
        "completed_iterations": config.iterations,
        "optimizer_update_count": metadata["optimizer_update_count"],
        "world_size": config.world_size,
        "search_cost_mode": config.search_cost_mode,
        "token_beta": config.token_beta,
        "workflow_cost_lambda": config.workflow_cost_lambda,
        "execution_cost_reference": config.execution_cost_reference,
        "iterations": iteration_results,
        "valid": iteration_results[-1]["valid"],
        "test": iteration_results[-1]["test"],
    }


def _save_checkpoint(
    path: Path,
    *,
    config: _WorkerConfig,
    policy: PlannerPolicy,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    policy.torch.save(
        {
            "method": config.spec.method,
            "seed": config.spec.seed,
            "algorithm_settings": _algorithm_settings(config),
            **metadata,
            "adapter": policy.adapter_state(),
            "optimizer": policy.optimizer_state(),
        },
        temporary,
    )
    os.replace(temporary, path)


def _load_checkpoint(
    path: Path,
    *,
    config: _WorkerConfig,
    policy: PlannerPolicy,
) -> dict[str, Any]:
    payload = policy.torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("method") != config.spec.method
        or int(payload.get("seed", -1)) != config.spec.seed
    ):
        raise ValueError("checkpoint method/seed does not match the requested cell")
    saved_settings = payload.get("algorithm_settings")
    requested_settings = _algorithm_settings(config)
    if not isinstance(saved_settings, dict) or any(
        saved_settings.get(name) != value for name, value in requested_settings.items()
    ):
        raise ValueError(
            "resume requires the same algorithm settings: "
            f"checkpoint={saved_settings}, requested={requested_settings}"
        )
    if payload.get("phase") not in {"collecting", "evaluating", "complete"}:
        raise ValueError("checkpoint does not contain a supported distributed phase")
    policy.load_adapter_state(payload["adapter"])
    policy.load_optimizer_state(payload["optimizer"])
    return payload


def _broadcast_object(value: Any, *, rank: int, device: Any, dist: Any) -> Any:
    values = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0, device=device)
    return values[0]


def _sync_metadata(
    metadata: dict[str, Any] | None,
    *,
    rank: int,
    device: Any,
    dist: Any,
) -> dict[str, Any]:
    runtime = None
    if rank == 0:
        if metadata is None:
            raise RuntimeError("rank 0 has no cell metadata")
        runtime = {
            key: metadata[key]
            for key in (
                "iteration",
                "phase",
                "completed_indices",
                "optimizer_update_count",
                "optimizer_telemetry",
            )
        }
    received = _broadcast_object(runtime, rank=rank, device=device, dist=dist)
    return metadata if rank == 0 else dict(received)


def _broadcast_adapter(policy: PlannerPolicy, *, dist: Any) -> None:
    for parameter in policy.trainable_parameters():
        dist.broadcast(parameter.data, src=0)


def _merge_training(
    cell_dir: Path,
    *,
    iteration: int,
    train_qids: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    seen_qids: set[str] = set()
    for index, expected_qid in enumerate(train_qids):
        path = _train_path(cell_dir, iteration, index)
        if not path.exists():
            raise RuntimeError(f"collection shard is missing: {path}")
        payload = _read_json(path)
        if (
            int(payload.get("dataset_index", -1)) != index
            or payload.get("qid") != expected_qid
        ):
            raise RuntimeError(f"collection shard differs from train item {index}")
        if expected_qid in seen_qids:
            raise RuntimeError(f"duplicate training QID: {expected_qid}")
        seen_qids.add(expected_qid)
        records.extend(payload["records"])
        trajectories.append(payload["trajectory"])
    return records, trajectories


def _write_trajectories(
    cell_dir: Path, train_count: int, through_iteration: int
) -> None:
    destination = cell_dir / "trajectories.jsonl"
    temporary = destination.with_name(destination.name + ".tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        for iteration in range(through_iteration + 1):
            for index in range(train_count):
                payload = _read_json(_train_path(cell_dir, iteration, index))
                handle.write(
                    json.dumps(
                        payload["trajectory"],
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
    os.replace(temporary, destination)


def _merge_evaluation(
    cell_dir: Path,
    *,
    iteration: int,
    split: str,
    world_size: int,
) -> dict[str, Any]:
    indexed_rows: list[tuple[int, dict[str, Any]]] = []
    for rank in range(world_size):
        path = _evaluation_path(cell_dir, iteration, split, rank)
        if not path.exists():
            raise RuntimeError(f"evaluation shard is missing: {path}")
        for entry in _read_json(path)["rows"]:
            indexed_rows.append((int(entry["dataset_index"]), dict(entry["row"])))
    indexed_rows.sort(key=lambda pair: pair[0])
    indices = [index for index, _ in indexed_rows]
    if indices != list(range(len(indices))):
        raise RuntimeError(f"{split} evaluation shards overlap or have gaps")
    return _summarize_evaluation([row for _, row in indexed_rows])


def _collect_local_batch(
    *,
    config: _WorkerConfig,
    policy: PlannerPolicy,
    components: RAGComponents,
    cache: QIDPrefixCache | None,
    local_train: dict[int, HotpotItem],
    iteration: int,
    indices: list[int],
) -> None:
    policy.model.eval()
    for index in indices:
        if index % config.world_size != config.rank:
            continue
        item = local_train.get(index)
        if item is None:
            raise RuntimeError(f"rank {config.rank} does not own train item {index}")
        destination = _train_path(
            config.output_dir / f"{config.spec.method}-seed-{config.spec.seed}",
            iteration,
            index,
        )
        if destination.exists():
            continue
        trajectory, records = _collect_item(
            item,
            policy,
            spec=config.spec,
            iteration=iteration,
            search_budget=config.search_budget,
            token_beta=config.token_beta,
            search_cost_mode=config.search_cost_mode,
            workflow_cost_lambda=config.workflow_cost_lambda,
            execution_cost_reference=config.execution_cost_reference,
            components=components,
            cache=cache,
            at2po_config=config.at2po_config,
        )
        _write_json(
            destination,
            {
                "rank": config.rank,
                "dataset_index": index,
                "qid": item.qid,
                "actor_version": iteration,
                "trajectory": trajectory,
                "records": records,
            },
        )


def _evaluate_local(
    *,
    config: _WorkerConfig,
    policy: PlannerPolicy,
    components: RAGComponents,
    iteration: int,
    split: str,
    indexed_items: list[tuple[int, HotpotItem]],
) -> None:
    cell_dir = config.output_dir / f"{config.spec.method}-seed-{config.spec.seed}"
    destination = _evaluation_path(cell_dir, iteration, split, config.rank)
    if destination.exists():
        return
    items = [item for _, item in indexed_items]
    rows = (
        _evaluation_rows(
            policy,
            items,
            rng=random.Random(f"{config.spec.seed}:{iteration}:{split}:{config.rank}"),
            search_budget=config.search_budget,
            token_beta=config.token_beta,
            search_cost_mode=config.search_cost_mode,
            workflow_cost_lambda=config.workflow_cost_lambda,
            execution_cost_reference=config.execution_cost_reference,
            components=components,
        )
        if items
        else []
    )
    _write_json(
        destination,
        {
            "rank": config.rank,
            "rows": [
                {"dataset_index": index, "row": row}
                for (index, _), row in zip(indexed_items, rows, strict=True)
            ],
        },
    )


def _rank_main(
    config: _WorkerConfig,
    local_train_items: list[tuple[int, HotpotItem]],
    local_valid_items: list[tuple[int, HotpotItem]],
    local_test_items: list[tuple[int, HotpotItem]],
    rank0_update_done: Any,
) -> None:
    import torch
    import torch.distributed as dist

    device = torch.device(config.device)
    if device.type != "cuda":
        raise ValueError("distributed cells require CUDA devices")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{config.master_port}",
        rank=config.rank,
        world_size=config.world_size,
        timeout=_PROCESS_GROUP_TIMEOUT,
    )
    cell_dir = config.output_dir / f"{config.spec.method}-seed-{config.spec.seed}"
    checkpoint_path = cell_dir / "checkpoint.pt"
    event_path = cell_dir / "events.jsonl"
    result_path = cell_dir / "run_result.json"
    policy: PlannerPolicy | None = None
    cache: QIDPrefixCache | None = None
    temporary_directory: TemporaryDirectory[str] | None = None
    try:
        policy = PlannerPolicy(
            config.model_path,
            config.device,
            seed=config.spec.seed,
            learner=config.rank == 0,
            **({"learning_rate": config.at2po_config.learning_rate}
               if config.spec.method == "at2po" else {}),
        )
        components = RAGComponents(
            generator=JSONTemplateGenerator(policy),
            ragspace=RAGSpace(
                config.rag_model_root,
                config.device,
                index_root=config.rag_index_root,
            ),
        )
        metadata: dict[str, Any] | None = None
        if config.rank == 0:
            if config.resume:
                if not checkpoint_path.exists():
                    raise FileNotFoundError(
                        f"resume checkpoint does not exist: {checkpoint_path}"
                    )
                payload = _load_checkpoint(
                    checkpoint_path, config=config, policy=policy
                )
                metadata = _checkpoint_metadata(payload)
                _append_event(
                    event_path,
                    {
                        "event": "resumed",
                        "iteration": metadata["iteration"],
                        "phase": metadata["phase"],
                        "completed_qids": len(metadata["completed_indices"]),
                    },
                )
            else:
                if checkpoint_path.exists():
                    raise FileExistsError(
                        f"fresh run refuses existing checkpoint: {checkpoint_path}"
                    )
                if cell_dir.exists() and any(cell_dir.iterdir()):
                    raise FileExistsError(
                        f"fresh run refuses non-empty cell directory: {cell_dir}"
                    )
                metadata = {
                    "iteration": 0,
                    "phase": "collecting",
                    "completed_indices": [],
                    "iteration_results": [],
                    "optimizer_update_count": 0,
                    "optimizer_telemetry": None,
                    "search_tokens": None,
                }
                _save_checkpoint(
                    checkpoint_path, config=config, policy=policy, metadata=metadata
                )
        metadata = _sync_metadata(metadata, rank=config.rank, device=device, dist=dist)
        _broadcast_adapter(policy, dist=dist)
        if metadata["phase"] == "complete":
            if config.rank == 0:
                metrics_path = cell_dir / "metrics.json"
                result = _completed_result(config, metadata)
                _write_json(metrics_path, result)
                _write_json(result_path, result)
            return

        if config.spec.method in CACHE_METHODS:
            temporary_directory = TemporaryDirectory(
                prefix=(
                    f"atppo-wtb-{config.spec.method}-{config.spec.seed}-"
                    f"rank-{config.rank}-"
                )
            )
            cache = QIDPrefixCache(temporary_directory.name)
        local_train = dict(local_train_items)
        collected_this_call = 0

        while int(metadata["iteration"]) < config.iterations:
            iteration = int(metadata["iteration"])
            if metadata["phase"] == "collecting":
                while True:
                    command: dict[str, Any] | None = None
                    if config.rank == 0:
                        completed = {
                            int(value) for value in metadata["completed_indices"]
                        }
                        pending = [
                            index
                            for index in range(config.train_count)
                            if index not in completed
                        ]
                        remaining = (
                            None
                            if config.max_new_train_items is None
                            else config.max_new_train_items - collected_this_call
                        )
                        if not pending:
                            rank0_update_done.clear()
                            command = {"kind": "update"}
                        elif remaining is not None and remaining <= 0:
                            rank_completed = [0] * config.world_size
                            for index in completed:
                                rank_completed[index % config.world_size] += 1
                            _write_json(
                                result_path,
                                {
                                    "method": config.spec.method,
                                    "seed": config.spec.seed,
                                    "status": "paused",
                                    "iteration": iteration,
                                    "completed": len(completed),
                                    "rank_completed": rank_completed,
                                    "optimizer_update_count": metadata[
                                        "optimizer_update_count"
                                    ],
                                    "world_size": config.world_size,
                                },
                            )
                            _append_event(
                                event_path,
                                {
                                    "event": "paused",
                                    "iteration": iteration,
                                    "completed_qids": len(completed),
                                },
                            )
                            command = {"kind": "pause"}
                        else:
                            count = min(config.checkpoint_interval_qids, len(pending))
                            if remaining is not None:
                                count = min(count, remaining)
                            command = {"kind": "collect", "indices": pending[:count]}
                    command = _broadcast_object(
                        command, rank=config.rank, device=device, dist=dist
                    )
                    if command["kind"] == "pause":
                        return
                    if command["kind"] == "update":
                        break
                    indices = [int(value) for value in command["indices"]]
                    _collect_local_batch(
                        config=config,
                        policy=policy,
                        components=components,
                        cache=cache,
                        local_train=local_train,
                        iteration=iteration,
                        indices=indices,
                    )
                    dist.barrier()
                    if config.rank == 0:
                        for index in indices:
                            path = _train_path(cell_dir, iteration, index)
                            if not path.exists():
                                raise RuntimeError(
                                    f"rank did not commit train item {index}"
                                )
                            payload = _read_json(path)
                            if payload.get("qid") != config.train_qids[index]:
                                raise RuntimeError(
                                    f"train item {index} has the wrong QID"
                                )
                        completed = {
                            int(value) for value in metadata["completed_indices"]
                        }
                        completed.update(indices)
                        metadata["completed_indices"] = sorted(completed)
                        collected_this_call += len(indices)
                        _save_checkpoint(
                            checkpoint_path,
                            config=config,
                            policy=policy,
                            metadata=metadata,
                        )
                        _append_event(
                            event_path,
                            {
                                "event": "qid_batch_committed",
                                "iteration": iteration,
                                "indices": indices,
                                "completed_qids": len(completed),
                            },
                        )
                    metadata = _sync_metadata(
                        metadata,
                        rank=config.rank,
                        device=device,
                        dist=dist,
                    )

                parallel_update = (
                    config.spec.method == "at2po"
                    and config.at2po_update_config.backend == "parallel"
                )
                records = []
                if config.rank == 0:
                    records, collected_trajectories = _merge_training(
                        cell_dir,
                        iteration=iteration,
                        train_qids=config.train_qids,
                    )
                    del collected_trajectories
                    metadata["search_tokens"] = _summarize_search_tokens(records)
                    _write_trajectories(cell_dir, config.train_count, iteration)
                if parallel_update or config.rank == 0:
                    torch.manual_seed(config.spec.seed + iteration * 1_000_003 + 71)
                    torch.cuda.manual_seed(
                        config.spec.seed + iteration * 1_000_003 + 71
                    )
                    components.ragspace.offload_models()
                    try:
                        if parallel_update:
                            telemetry = update_at2po(
                                policy, records, epochs=config.epochs,
                                batch_size=config.ppo_batch_size,
                                config=config.at2po_config,
                                execution=config.at2po_update_config, dist=dist,
                                progress=lambda values: _append_event(event_path, {
                                    "event": "at2po_update_progress",
                                    "iteration": iteration, **values,
                                }),
                            )
                        else:
                            telemetry = _update(
                                policy, records, epochs=config.epochs,
                                batch_size=config.ppo_batch_size, method=config.spec.method,
                                at2po_config=config.at2po_config,
                                at2po_update_config=config.at2po_update_config,
                            )
                    finally:
                        components.ragspace.restore_models()
                    del records
                    if policy.device.type == "cuda":
                        policy.torch.cuda.empty_cache()
                if config.rank == 0:
                    metadata["phase"] = "evaluating"
                    metadata["optimizer_update_count"] = (
                        int(metadata["optimizer_update_count"]) + 1
                    )
                    metadata["optimizer_telemetry"] = telemetry
                    _save_checkpoint(
                        checkpoint_path,
                        config=config,
                        policy=policy,
                        metadata=metadata,
                    )
                    _append_event(
                        event_path,
                        {
                            "event": "ppo_update_committed",
                            "iteration": iteration,
                            "optimizer_update_count": metadata[
                                "optimizer_update_count"
                            ],
                        },
                    )
                    rank0_update_done.set()
                elif not parallel_update:
                    if not rank0_update_done.wait(timeout=_RANK0_UPDATE_WAIT_SECONDS):
                        raise TimeoutError("rank 0 PPO update did not finish in time")
                metadata = _sync_metadata(
                    metadata,
                    rank=config.rank,
                    device=device,
                    dist=dist,
                )
                _broadcast_adapter(policy, dist=dist)

            if metadata["phase"] != "evaluating":
                raise RuntimeError(
                    f"unsupported distributed phase: {metadata['phase']}"
                )
            policy.model.eval()
            _evaluate_local(
                config=config,
                policy=policy,
                components=components,
                iteration=iteration,
                split="valid",
                indexed_items=local_valid_items,
            )
            final_iteration = iteration == config.iterations - 1
            if final_iteration:
                _evaluate_local(
                    config=config,
                    policy=policy,
                    components=components,
                    iteration=iteration,
                    split="test",
                    indexed_items=local_test_items,
                )
            dist.barrier()
            if config.rank == 0:
                valid = _merge_evaluation(
                    cell_dir,
                    iteration=iteration,
                    split="valid",
                    world_size=config.world_size,
                )
                test = (
                    _merge_evaluation(
                        cell_dir,
                        iteration=iteration,
                        split="test",
                        world_size=config.world_size,
                    )
                    if final_iteration
                    else None
                )
                iteration_result = {
                    "iteration": iteration,
                    "train_qids": config.train_count,
                    "optimizer": metadata["optimizer_telemetry"],
                    "search_tokens": metadata["search_tokens"],
                    "valid": valid,
                    "test": test,
                }
                metadata["iteration_results"] = [
                    *metadata["iteration_results"],
                    iteration_result,
                ]
                metadata["iteration"] = iteration + 1
                metadata["completed_indices"] = []
                metadata["optimizer_telemetry"] = None
                metadata["search_tokens"] = None
                metadata["phase"] = (
                    "complete"
                    if int(metadata["iteration"]) >= config.iterations
                    else "collecting"
                )
                result: dict[str, Any] | None = None
                if metadata["phase"] == "complete":
                    result = _completed_result(config, metadata)
                    _write_json(cell_dir / "metrics.json", result)
                    _write_json(result_path, result)
                _save_checkpoint(
                    checkpoint_path,
                    config=config,
                    policy=policy,
                    metadata=metadata,
                )
                _append_event(
                    event_path,
                    {
                        "event": "iteration_completed",
                        "iteration": iteration,
                        "valid_f1": valid["mean_f1"],
                        "test_f1": None if test is None else test["mean_f1"],
                    },
                )
                if metadata["phase"] == "complete":
                    _append_event(
                        event_path,
                        {"event": "complete", "iterations": config.iterations},
                    )
            metadata = _sync_metadata(
                metadata,
                rank=config.rank,
                device=device,
                dist=dist,
            )
        dist.barrier()
    finally:
        if cache is not None:
            cache.close()
        if temporary_directory is not None:
            temporary_directory.cleanup()
        if dist.is_initialized():
            dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _split_items(
    items: list[HotpotItem],
    world_size: int,
) -> list[list[tuple[int, HotpotItem]]]:
    shards: list[list[tuple[int, HotpotItem]]] = [[] for _ in range(world_size)]
    for index, item in enumerate(items):
        shards[index % world_size].append((index, item))
    return shards


def run_distributed_cell(
    spec: CellSpec,
    *,
    model_path: Path,
    rag_model_root: Path,
    rag_index_root: Path,
    output_dir: Path,
    train_items: list[HotpotItem],
    valid_items: list[HotpotItem],
    test_items: list[HotpotItem],
    devices: tuple[str, ...],
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
    """Run one method/seed with fixed disjoint QID shards on every device."""

    if spec.method not in METHODS:
        raise ValueError(f"unknown method: {spec.method}")
    if len(devices) < 2 or len(set(devices)) != len(devices):
        raise ValueError("distributed cell requires at least two distinct CUDA devices")
    if iterations <= 0 or ppo_batch_size <= 0 or checkpoint_interval_qids <= 0:
        raise ValueError(
            "iterations, ppo_batch_size, and checkpoint interval must be positive"
        )
    world_size = len(devices)
    train_shards = _split_items(train_items, world_size)
    valid_shards = _split_items(valid_items, world_size)
    test_shards = _split_items(test_items, world_size)
    master_port = _free_port()
    cell_dir = output_dir / f"{spec.method}-seed-{spec.seed}"
    result_path = cell_dir / "run_result.json"
    if result_path.exists():
        result_path.unlink()

    context = get_context("spawn")
    rank0_update_done = context.Event()
    processes = []
    for rank, device in enumerate(devices):
        config = _WorkerConfig(
            spec=spec,
            model_path=model_path,
            rag_model_root=rag_model_root,
            rag_index_root=rag_index_root,
            output_dir=output_dir,
            device=device,
            rank=rank,
            world_size=world_size,
            master_port=master_port,
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
            train_count=len(train_items),
            train_qids=tuple(item.qid for item in train_items),
            at2po_config=at2po_config,
            at2po_update_config=at2po_update_config,
        )
        process = context.Process(
            target=_rank_main,
            args=(
                config,
                train_shards[rank],
                valid_shards[rank],
                test_shards[rank],
                rank0_update_done,
            ),
            name=f"atppo-{spec.method}-seed-{spec.seed}-rank-{rank}",
        )
        process.start()
        processes.append(process)

    remaining = set(processes)
    failed: list[tuple[str, int | None]] = []
    while remaining:
        ready = wait([process.sentinel for process in remaining], timeout=1.0)
        for process in list(remaining):
            if process.sentinel not in ready:
                continue
            process.join()
            remaining.remove(process)
            if process.exitcode != 0:
                failed.append((process.name, process.exitcode))
        if failed:
            for process in remaining:
                process.terminate()
            for process in remaining:
                process.join()
            raise RuntimeError(f"distributed cell worker failed: {failed}")
    if not result_path.exists():
        raise RuntimeError("distributed cell finished without a result")
    return dict(_read_json(result_path))


__all__ = ["run_distributed_cell"]
