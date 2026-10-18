"""Audit, aggregate, and report an expanded planner evaluation grid."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path


SEEDS = (11, 23, 37)
LABELS = {
    "base": "Base Qwen3-4B",
    "ppo": "PPO",
    "vineppo": "VinePPO",
    "agentuct": "AgentUCT-RL",
    "uniform": "Uniform-RL",
    "uct": "UCT-RL",
}


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def aggregate(rows: list[dict], *, include_hops: bool = True) -> dict:
    result = {
        "n": len(rows),
        "f1": statistics.mean(row["f1_official"] for row in rows),
        "em": statistics.mean(row["em_official"] for row in rows),
        "utility": statistics.mean(row["utility_official"] for row in rows),
        "workflow_words": statistics.mean(row["execution_cost_words"] for row in rows),
    }
    hops = sorted({row.get("hop") for row in rows if row.get("hop") is not None}) if include_hops else []
    if hops:
        result["by_hop"] = {
            str(hop): aggregate(
                [row for row in rows if row["hop"] == hop], include_hops=False
            )
            for hop in hops
        }
    return result


def mean_sd(values: list[float]) -> dict:
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    from atppo_lab.official_metrics import exact_match_score, f1_score

    cells = []
    row_counts = {}
    for dataset_spec in args.datasets:
        dataset, source_name = dataset_spec.split("=", 1)
        source_rows = json.loads(Path(source_name).read_text(encoding="utf-8"))
        source = {row["id"]: row for row in source_rows}
        for method in args.methods:
            method_seeds = (20260915,) if method == "base" else SEEDS
            for seed in method_seeds:
                directory = args.root / "runs" / dataset / method / f"seed-{seed}" / "rows"
                rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))]
                if len(rows) != len(source) or {row["qid"] for row in rows} != set(source):
                    raise RuntimeError(f"incomplete cell: {dataset}/{method}/{seed}")
                for row in rows:
                    gold = source[row["qid"]]
                    golds = gold.get("answer_aliases", [gold["answer"]])
                    f1 = max(f1_score(row["prediction"], answer)[0] for answer in golds)
                    em = max(float(exact_match_score(row["prediction"], answer)) for answer in golds)
                    utility = f1 - 0.1 * row["execution_cost_words"] / 4096
                    if not math.isclose(row["f1_official"], f1, abs_tol=1e-12):
                        raise RuntimeError("F1 mismatch")
                    if not math.isclose(row["em_official"], em, abs_tol=1e-12):
                        raise RuntimeError("EM mismatch")
                    if not math.isclose(row["utility_official"], utility, abs_tol=1e-12):
                        raise RuntimeError("utility mismatch")
                    if row["path"] != [step["chosen_label"] for step in row["steps"]]:
                        raise RuntimeError("path mismatch")
                    for key, value in row["token_usage"].items():
                        if value != sum(step["token_usage"][key] for step in row["steps"]):
                            raise RuntimeError("token ledger mismatch")
                    if row["new_search_evaluations"] or row["new_optimizer_updates"]:
                        raise RuntimeError("evaluation mutated search or training state")
                    row["hop"] = gold.get("hop")
                cells.append({"dataset": dataset, "method": method, "seed": seed, **aggregate(rows)})
                row_counts[f"{dataset}/{method}/{seed}"] = len(rows)

    summaries = []
    for dataset_spec in args.datasets:
        dataset = dataset_spec.split("=", 1)[0]
        for method in args.methods:
            selected = [cell for cell in cells if cell["dataset"] == dataset and cell["method"] == method]
            if method == "base":
                summaries.append({"dataset": dataset, "method": method, "metrics": {key: selected[0][key] for key in ("f1", "em", "utility", "workflow_words")}, "by_hop": selected[0].get("by_hop")})
                continue
            entry = {
                "dataset": dataset,
                "method": method,
                "metrics": {
                    key: mean_sd([cell[key] for cell in selected])
                    for key in ("f1", "em", "utility", "workflow_words")
                },
            }
            if all("by_hop" in cell for cell in selected):
                entry["by_hop"] = {
                    hop: {
                        key: mean_sd([cell["by_hop"][hop][key] for cell in selected])
                        for key in ("f1", "em", "utility")
                    }
                    for hop in selected[0]["by_hop"]
                }
            summaries.append(entry)

    result = {
        "complete": True,
        "title": args.title,
        "datasets": args.datasets,
        "methods": args.methods,
        "cells": cells,
        "summaries": summaries,
    }
    audit = {
        "status": "passed",
        "formal_endpoints": sum(row_counts.values()),
        "row_counts": row_counts,
        "official_f1_em_utility_recomputed": True,
        "paths_and_token_ledgers_checked": True,
        "new_search_evaluations": 0,
        "new_optimizer_updates": 0,
    }
    save(args.root / "summary.json", result)
    save(args.root / "audit-final.json", audit)

    lines = [f"# {args.title}", "", "| Dataset | Planner | F1 | EM | U | Workflow words |", "|---|---|---:|---:|---:|---:|"]
    for entry in summaries:
        metric = entry["metrics"]
        if entry["method"] == "base":
            values = [f"{metric[key]:.6f}" for key in ("f1", "em", "utility", "workflow_words")]
        else:
            values = [f"{metric[key]['mean']:.6f} ± {metric[key]['sample_sd']:.6f}" for key in ("f1", "em", "utility", "workflow_words")]
        lines.append(f"| {entry['dataset']} | {LABELS[entry['method']]} | " + " | ".join(values) + " |")
    lines += ["", "RL values are mean ± sample SD over seeds 11/23/37. Base is one deterministic run.", ""]
    (args.root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
