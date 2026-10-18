"""Real-model 2/1/1 AT²PO collection, update, and checkpoint-resume gate."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .at2po import AT2POConfig
from .main import RunConfig, run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--rag-model-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--leaf-budget", type=int, default=22)
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
        methods=("at2po",),
        seeds=(11,),
        devices=tuple(args.devices),
        iterations=2,
        epochs=2,
        checkpoint_interval_qids=1,
        at2po_config=AT2POConfig(leaf_budget=args.leaf_budget),
    )
    # Two QIDs in iteration 0, then one in iteration 1 before interruption.
    paused = run_experiment(replace(config, max_new_train_items=3))[0]
    assert paused["status"] == "paused" and paused["iteration"] == 1
    import torch

    cell = args.output_dir / "at2po-seed-11"
    before = torch.load(cell / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert before["algorithm_settings"]["at2po_leaf_budget"] == args.leaf_budget
    assert before["optimizer"]["state"]
    assert any(
        bool(v.abs().max() > 0) for k, v in before["adapter"].items() if "lora_B" in k
    )
    result = run_experiment(replace(config, resume=True))[0]
    assert result["status"] == "complete" and result["completed_iterations"] == 2
    rows = [
        json.loads(line)
        for line in (cell / "trajectories.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 4 and len({(r["iteration"], r["qid"]) for r in rows}) == 4
    for row in rows:
        tree = row["at2po_tree"]
        assert len(tree["samples"]) == len(tree["leaf_ids"]) == args.leaf_budget
        assert len({r["at2po_sequence"] for r in row["steps"]}) == args.leaf_budget
        assert "actor_path" not in row
        for record in row["steps"]:
            assert record["advantage"] == tree["nodes"][record["at2po_node"]]["value"]
    logical = sum(
        r["search_tokens"]["logical_evaluations"] for r in result["iterations"]
    )
    assert logical == 4 * args.leaf_budget
    assert all(r["optimizer"]["optimizer_steps"] > 0 for r in result["iterations"])
    assert len(result["valid"]["rows"]) == len(result["test"]["rows"]) == 1
    receipt = {
        "status": "passed",
        "method": "at2po",
        "train_valid_test": [2, 1, 1],
        "iterations": 2,
        "seed": 11,
        "devices": args.devices,
        "resume_after_first_update": True,
        "terminal_evaluations": logical,
        "at2po_settings": config.at2po_config.settings(),
        "torch": torch.__version__,
        "real_model_run": True,
    }
    (args.output_dir / "tiny_at2po_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
