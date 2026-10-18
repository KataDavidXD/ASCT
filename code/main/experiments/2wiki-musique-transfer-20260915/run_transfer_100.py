"""Resume-safe multi-GPU evaluator for the 100-item transfer cohorts."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


SEEDS = (11, 23, 37)


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def checkpoint(resources: Path, method: str, seed: int) -> tuple[Path | None, str | None]:
    if method == "base":
        return None, None
    if method in {"ppo", "vineppo"}:
        path = resources / "github-trained-planners-20260915" / method / f"{method}-seed-{seed}" / "checkpoint.pt"
        return path, method
    path = resources / "rl-checkpoints" / f"{method}_atppo-seed-{seed}.pt"
    return path, f"{method}_atppo"


def gpu_devices(indices: tuple[int, ...]) -> dict[int, str]:
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    devices = {}
    for row in rows:
        index, uuid, used = map(str.strip, row.split(","))
        if int(index) in indices:
            if int(used) >= 1000:
                raise RuntimeError(f"GPU {index} is already occupied")
            devices[int(index)] = uuid
    if set(devices) != set(indices):
        raise RuntimeError("requested GPUs are unavailable")
    return devices


def jobs(args: argparse.Namespace, gate: bool) -> list[dict]:
    result = []
    rank = 0
    for dataset_spec in args.datasets:
        dataset_name, source_name = dataset_spec.split("=", 1)
        source = Path(source_name)
        qids = [row["id"] for row in json.loads(source.read_text(encoding="utf-8"))]
        if gate:
            qids = qids[:1]
        for method in args.methods:
            method_seeds = (20260915,) if method == "base" else SEEDS
            for seed in method_seeds:
                model_checkpoint, payload_method = checkpoint(args.resources, method, seed)
                output = args.root / ("gate" if gate else "runs") / dataset_name / method / f"seed-{seed}"
                result.append(
                    {
                        "protocol": args.protocol,
                        "dataset_name": dataset_name,
                        "method": method,
                        "role": "planner" if method in {"base", "ppo", "vineppo"} else "rl",
                        "seed": seed,
                        "rank": rank,
                        "qids": qids,
                        "dataset": str(source),
                        "output": str(output),
                        "model": str(args.resources / "models/Qwen3-4B-Instruct-2507"),
                        "rag_models": str(args.resources / "models/paper-components"),
                        "checkpoint": str(model_checkpoint) if model_checkpoint else None,
                        "checkpoint_payload_method": payload_method,
                    }
                )
                rank += 1
    return result


def execute(args: argparse.Namespace, specs: list[dict]) -> None:
    devices = gpu_devices(args.gpus)
    pending: queue.Queue[dict] = queue.Queue()
    for spec in specs:
        pending.put(spec)
    errors = []
    stop = threading.Event()
    base_environment = dict(
        os.environ,
        PYTHONPATH=f"{args.repo / 'src'}:{args.repo / 'packages/agent-uct'}:{args.tools}",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
    )

    def worker(gpu: int) -> None:
        environment = dict(base_environment, CUDA_VISIBLE_DEVICES=devices[gpu])
        while not stop.is_set():
            try:
                spec = pending.get_nowait()
            except queue.Empty:
                return
            output = Path(spec["output"])
            job_path = output / "jobs" / f"rank-{spec['rank']}.json"
            save(job_path, spec)
            log_path = output / "jobs" / f"rank-{spec['rank']}.log"
            try:
                with log_path.open("a", encoding="utf-8") as stream:
                    subprocess.run(
                        [str(args.python), str(args.tools / "eval_external_planner.py"), "--job", str(job_path)],
                        cwd=args.repo,
                        env=environment,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=14400,
                    )
            except Exception as error:
                errors.append({"job": str(job_path), "error": str(error), "log": str(log_path)})
                stop.set()

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        list(pool.map(worker, args.gpus))
    if errors:
        save(args.root / "errors.json", errors)
        raise RuntimeError(errors)


def verify(args: argparse.Namespace, gate: bool) -> int:
    total = 0
    for spec in jobs(args, gate):
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in (Path(spec["output"]) / "rows").glob("*.json")]
        if len(rows) != len(spec["qids"]) or {row["qid"] for row in rows} != set(spec["qids"]):
            raise RuntimeError(f"incomplete cell: {spec['dataset_name']}/{spec['method']}/{spec['seed']}")
        if any(row["new_search_evaluations"] or row["new_optimizer_updates"] for row in rows):
            raise RuntimeError("evaluation mutated search or training state")
        total += len(rows)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", required=True, choices=("base", "ppo", "vineppo", "agentuct", "uniform", "uct"))
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    args = parser.parse_args()
    args.gpus = tuple(args.gpus)
    args.root.mkdir(parents=True, exist_ok=True)
    execute(args, jobs(args, True))
    gate_rows = verify(args, True)
    execute(args, jobs(args, True))
    execute(args, jobs(args, False))
    formal_rows = verify(args, False)
    save(
        args.root / "status.json",
        {
            "phase": "complete",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "protocol": args.protocol,
            "datasets": args.datasets,
            "methods": args.methods,
            "gate_rows": gate_rows,
            "formal_rows": formal_rows,
            "new_search_evaluations": 0,
            "new_optimizer_updates": 0,
        },
    )


if __name__ == "__main__":
    main()
