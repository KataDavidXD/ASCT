"""Estimate a per-QID AT²PO leaf budget from existing TRAIN ledgers only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .at2po import AT2POConfig


def estimate_budget(paths: list[Path]) -> dict[str, Any]:
    cells, rows, seen = [], [], set()
    for path in paths:
        raw = path.read_bytes()
        data = json.loads(raw)
        key = (data["method"], int(data["seed"]))
        if key in seen:
            raise ValueError(f"duplicate method/seed metrics: {key}")
        seen.add(key)
        if data.get("status") != "complete":
            raise ValueError(f"incomplete cell: {path}")
        iterations = data.get("iterations", [])
        if not iterations or len(iterations) != data.get("completed_iterations"):
            raise ValueError(f"missing completed iterations: {path}")
        total_evaluations = total_qids = 0
        for row in iterations:
            qids = row["train_qids"]
            count = len(qids) if isinstance(qids, list) else int(qids)
            evaluations = row["search_tokens"]["logical_evaluations"]
            if count <= 0 or type(evaluations) is not int or evaluations <= 0:
                raise ValueError(f"invalid training evaluation ledger: {path}")
            total_evaluations += evaluations
            total_qids += count
            rows.append(
                {
                    "method": key[0],
                    "seed": key[1],
                    "iteration": row["iteration"],
                    "qid_visits": count,
                    "terminal_evaluations": evaluations,
                    "evaluations_per_qid": evaluations / count,
                }
            )
        cells.append(
            {
                "source": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "method": key[0],
                "seed": key[1],
                "qid_visits": total_qids,
                "terminal_evaluations": total_evaluations,
                "evaluations_per_qid": total_evaluations / total_qids,
            }
        )
    if not cells:
        raise ValueError("provide at least one completed tree/VinePPO metrics file")
    # Each selected method/seed receives equal weight. Declared before AT2PO
    # training; validation/test rewards are never consulted.
    mean = math.fsum(c["evaluations_per_qid"] for c in cells) / len(cells)
    budget = max(2, math.ceil(mean))
    return {
        "schema": "at2po_training_leaf_budget_v1",
        "cells": cells,
        "iterations": rows,
        "reference_mean_evaluations_per_qid": mean,
        "recommended_leaf_budget": budget,
        "relative_sample_count_overhead": budget / mean - 1,
        "schedule": AT2POConfig(leaf_budget=budget).settings(),
        "counting": "logical terminal samples including repeats and full WTB hits",
        "matching": "approximate terminal-evaluation count; actual token/time/update costs remain measured",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = estimate_budget(args.metrics)
    if args.output.exists():
        raise FileExistsError(f"budget receipt already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "leaf_budget": result["recommended_leaf_budget"],
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
