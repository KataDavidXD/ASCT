"""Single-GPU, real-model 2/1/1 VinePPO gate with a post-update resume."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

from .main import RunConfig, run_experiment


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
        methods=("vineppo",),
        seeds=(11,),
        devices=(args.device,),
        search_budget=12,
        iterations=2,
        epochs=2,
        checkpoint_interval_qids=1,
    )
    paused = run_experiment(replace(config, max_new_train_items=3))
    assert paused[0]["status"] == "paused" and paused[0]["iteration"] == 1
    import torch

    checkpoint = args.output_dir / "vineppo-seed-11" / "checkpoint.pt"
    before = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert len(before["iteration_results"]) == 1 and len(before["completed_qids"]) == 1
    assert before["algorithm_settings"]["vine_gamma"] == 1.0
    assert any(
        bool(value.abs().max() > 0)
        for name, value in before["adapter"].items()
        if "lora_B" in name
    )
    assert before["optimizer"]["state"]
    frozen_records = before["records"]
    completed = run_experiment(replace(config, resume=True))
    result = completed[0]
    assert result["status"] == "complete" and result["completed_iterations"] == 2
    cell = args.output_dir / "vineppo-seed-11"
    rows = [
        json.loads(line)
        for line in (cell / "trajectories.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 4 and len({(row["iteration"], row["qid"]) for row in rows}) == 4
    restored = next(
        row
        for row in rows
        if row["iteration"] == 1 and row["qid"] == before["completed_qids"][0]
    )
    assert [row["advantage"] for row in restored["steps"]] == [
        row["advantage"] for row in frozen_records
    ]
    logical = 0
    modes: dict[str, int] = {}
    for row in rows:
        for step in row["steps"]:
            assert len(step["vine_trace"]) == step["vine_samples"] == 12
            assert math.isclose(
                step["advantage"],
                step["reward"] + step["next_vine_value"] - step["vine_value"],
                abs_tol=1e-12,
            )
            logical += 12
            for sample in step["vine_trace"]:
                mode = sample["wtb_mode"]
                modes[mode] = modes.get(mode, 0) + 1
                if mode == "full_hit":
                    assert all(
                        sample["token_usage"][key] == 0
                        for key in (
                            "generator_4b_input_tokens",
                            "generator_4b_output_tokens",
                            "query_embedding_06b_input_tokens",
                            "reranker_06b_input_tokens",
                        )
                    )
    assert (
        sum(
            item["search_tokens"]["logical_evaluations"]
            for item in result["iterations"]
        )
        == logical
    )
    assert all(
        item["optimizer"]["optimizer_steps"] > 0 for item in result["iterations"]
    )
    assert len(result["test"]["rows"]) == len(result["valid"]["rows"]) == 1
    assert all("vine_trace" not in row for row in result["test"]["rows"])
    receipt = {
        "status": "passed",
        "method": "vineppo",
        "train_valid_test": [2, 1, 1],
        "iterations": 2,
        "seed": 11,
        "budget": 12,
        "device": args.device,
        "resume_after_first_update": True,
        "logical_auxiliary_evaluations": logical,
        "wtb_modes": modes,
        "torch": torch.__version__,
        "search_qwen_tokens": sum(
            item["search_tokens"]["qwen_tokens"] for item in result["iterations"]
        ),
        "auxiliary_actor_scoring_tokens": sum(
            item["search_tokens"]["vine_auxiliary_actor_scoring_tokens"]
            for item in result["iterations"]
        ),
    }
    (args.output_dir / "tiny_vine_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
