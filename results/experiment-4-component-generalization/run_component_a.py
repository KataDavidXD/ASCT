"""Run all frozen planners on the four component-A descriptions."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from protocol import CONDITIONS


SEEDS = (11, 23, 37)


def checkpoint(resources: Path, method: str, seed: int) -> tuple[Path, str]:
    if method in {"ppo", "vineppo"}:
        path = (
            resources / "github-trained-planners-20260915" / method
            / f"{method}-seed-{seed}" / "checkpoint.pt"
        )
        return path, method
    return resources / "rl-checkpoints" / f"{method}_atppo-seed-{seed}.pt", f"{method}_atppo"


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pending.replace(path)


def available_devices(gpus: tuple[int, ...]) -> dict[int, str]:
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    devices = {}
    for row in rows:
        index, uuid, used = map(str.strip, row.split(","))
        if int(index) in gpus:
            if int(used) >= 1000:
                raise RuntimeError(f"GPU {index} is already occupied")
            devices[int(index)] = uuid
    if set(devices) != set(gpus):
        raise RuntimeError("requested GPUs are unavailable")
    return devices


def build_jobs(args: argparse.Namespace, qids: list[str], output_name: str) -> list[dict]:
    jobs = []
    rank = 0
    for method in args.rl_methods:
        for seed in SEEDS:
            checkpoint_path, checkpoint_method = checkpoint(args.resources, method, seed)
            jobs.append(
                {
                    "protocol": "component-description-controls-v1",
                    "method": method,
                    "seed": seed,
                    "planner_kind": "rl",
                    "conditions": list(CONDITIONS),
                    "rank": rank,
                    "qids": qids,
                    "dataset": str(args.dataset),
                    "output": str(
                        args.root / output_name / "rl" / method / f"seed-{seed}"
                    ),
                    "model": str(args.resources / "models/Qwen3-4B-Instruct-2507"),
                    "rag_models": str(args.resources / "models/paper-components"),
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_payload_method": checkpoint_method,
                }
            )
            rank += 1
    if args.include_base:
        jobs.append(
            {
                "protocol": "component-description-controls-v1",
                "method": "base_qwen3_4b",
                "seed": 20260915,
                "planner_kind": "base",
                "conditions": list(CONDITIONS),
                "rank": rank,
                "qids": qids,
                "dataset": str(args.dataset),
                "output": str(args.root / output_name / "base_qwen3_4b"),
                "model": str(args.resources / "models/Qwen3-4B-Instruct-2507"),
                "rag_models": str(args.resources / "models/paper-components"),
            }
        )
    return jobs


def execute(args: argparse.Namespace, jobs: list[dict]) -> None:
    devices = available_devices(args.gpus)
    pending: queue.Queue[dict] = queue.Queue()
    for job in jobs:
        pending.put(job)
    errors: list[dict] = []
    stop = threading.Event()

    environment = dict(
        os.environ,
        PYTHONPATH=f"{args.code / 'src'}:{args.code / 'packages/agent-uct'}:{args.tools}",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
    )

    def worker(gpu: int) -> None:
        worker_environment = dict(environment, CUDA_VISIBLE_DEVICES=devices[gpu])
        while not stop.is_set():
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            output = Path(job["output"])
            job_path = output / "jobs" / f"rank-{job['rank']}.json"
            save(job_path, job)
            log_path = output / "jobs" / f"rank-{job['rank']}.log"
            try:
                with log_path.open("a", encoding="utf-8") as stream:
                    subprocess.run(
                        [str(args.python), str(args.tools / "eval_description_controls.py"), "--job", str(job_path)],
                        cwd=args.code,
                        env=worker_environment,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=14400,
                    )
            except Exception as error:
                errors.append({"job": str(job_path), "error": str(error)})
                stop.set()

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        list(pool.map(worker, args.gpus))
    if errors:
        save(args.root / "uct-errors.json", errors)
        raise RuntimeError(errors)


def verify(args: argparse.Namespace, expected_qids: set[str], output_name: str) -> None:
    for method in args.rl_methods:
        for seed in SEEDS:
            for condition in CONDITIONS:
                directory = args.root / output_name / "rl" / method / f"seed-{seed}" / condition / "rows"
                rows = [json.loads(path.read_text(encoding="utf-8")) for path in directory.glob("*.json")]
                if len(rows) != len(expected_qids) or {row["qid"] for row in rows} != expected_qids:
                    raise RuntimeError(f"incomplete rows: method={method} seed={seed} condition={condition}")
                for row in rows:
                    if row["method"] != method or row["condition"] != condition:
                        raise RuntimeError("row identity mismatch")
                    if row["new_search_evaluations"] or row["new_optimizer_updates"]:
                        raise RuntimeError("evaluation mutated training or search state")
    if args.include_base:
        for condition in CONDITIONS:
            directory = args.root / output_name / "base_qwen3_4b" / condition / "rows"
            rows = [json.loads(path.read_text(encoding="utf-8")) for path in directory.glob("*.json")]
            if len(rows) != len(expected_qids) or {row["qid"] for row in rows} != expected_qids:
                raise RuntimeError(f"incomplete base rows: condition={condition}")
            if not all(row["base_adapter_disabled_for_planner"] for row in rows):
                raise RuntimeError("base planner adapter was not disabled")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument(
        "--rl-methods",
        nargs="+",
        choices=("ppo", "vineppo", "agentuct", "uniform", "uct"),
        default=["ppo", "vineppo", "agentuct", "uniform", "uct"],
    )
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--status-name", default="component-a-status.json")
    args = parser.parse_args()
    args.gpus = tuple(args.gpus)

    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    qids = [row["id"] for row in rows]
    if len(qids) != 100 or len(set(qids)) != 100:
        raise RuntimeError("component-A protocol requires 100 unique QIDs")

    gate_jobs = build_jobs(args, qids[:1], "uct-gate")
    execute(args, gate_jobs)
    verify(args, set(qids[:1]), "uct-gate")
    execute(args, build_jobs(args, qids, "runs"))
    verify(args, set(qids), "runs")
    endpoint_count = (
        len(args.rl_methods) * len(SEEDS) + int(args.include_base)
    ) * len(CONDITIONS) * len(qids)
    save(
        args.root / args.status_name,
        {
            "status": "complete",
            "protocol": "component-description-controls-v1",
            "rl_planners": list(args.rl_methods),
            "base_qwen3_4b": args.include_base,
            "seeds": list(SEEDS),
            "conditions": list(CONDITIONS),
            "endpoints": endpoint_count,
            "new_search_evaluations": 0,
            "new_optimizer_updates": 0,
        },
    )


if __name__ == "__main__":
    main()
