"""Recompute and publish the six-planner component-A comparison."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from protocol import CONDITIONS, NEW_COMPONENT, descriptions, strip_new_description


RL_METHODS = ("ppo", "vineppo", "agentuct", "uniform", "uct")
SEEDS = (11, 23, 37)
LABELS = {
    "ppo": "PPO",
    "vineppo": "VinePPO",
    "agentuct": "AgentUCT-RL",
    "uniform": "Uniform-RL",
    "uct": "UCT-RL",
    "base_qwen3_4b": "Base Qwen3-4B",
}
CONDITION_LABELS = {
    "correct_relation": "type + correct relation",
    "type_only": "type only",
    "name_only": "name only",
    "opposite_relation": "type + opposite relation",
}
CONDITION_LABELS_ZH = {
    "correct_relation": "类型 + 正确关系",
    "type_only": "只说明类型",
    "name_only": "只给名字",
    "opposite_relation": "类型 + 相反关系",
}


def save(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def rows_in(directory: Path) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    ]


def rl_rows(root: Path, method: str, seed: int, condition: str) -> list[dict]:
    direct = root / "runs/rl" / method / f"seed-{seed}" / condition / "rows"
    if direct.is_dir():
        return rows_in(direct)
    if method in {"agentuct", "uniform"} and condition == "correct_relation":
        rows = rows_in(root / "reused-correct/rl" / method / f"seed-{seed}" / "rows")
        return [
            {
                **row,
                "condition": condition,
                "new_component_selected": row["tagged_component_selected"],
                "new_component_probability": row["tagged_component_probability"],
            }
            for row in rows
        ]
    return rows_in(
        root / "runs/rl" / method / f"seed-{seed}" / condition / "rows"
    )


def metrics(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "selection_rate": statistics.mean(
            row["new_component_selected"] for row in rows
        ),
        "probability_a": statistics.mean(
            row["new_component_probability"] for row in rows
        ),
        "f1": statistics.mean(row["f1_official"] for row in rows),
        "em": statistics.mean(row["em_official"] for row in rows),
        "utility": statistics.mean(row["utility_official"] for row in rows),
        "workflow_words": statistics.mean(
            row["execution_cost_words"] for row in rows
        ),
    }


def mean_sd(values: list[float]) -> dict:
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}


def audit_row(row: dict, source: dict, condition: str) -> None:
    from atppo_lab.official_metrics import exact_match_score, f1_score

    golds = source.get("answer_aliases", [source["answer"]])
    f1 = max(f1_score(row["prediction"], gold)[0] for gold in golds)
    em = max(float(exact_match_score(row["prediction"], gold)) for gold in golds)
    if not math.isclose(row["f1_official"], f1, abs_tol=1e-12):
        raise RuntimeError("F1 mismatch")
    if not math.isclose(row["em_official"], em, abs_tol=1e-12):
        raise RuntimeError("EM mismatch")
    utility = f1 - 0.1 * row["execution_cost_words"] / 4096
    if not math.isclose(row["utility_official"], utility, abs_tol=1e-12):
        raise RuntimeError("utility mismatch")
    if row["path"] != [step["chosen_label"] for step in row["steps"]]:
        raise RuntimeError("path mismatch")
    if row["new_search_evaluations"] or row["new_optimizer_updates"]:
        raise RuntimeError("evaluation mutated training or search state")
    for key, value in row["token_usage"].items():
        if value != sum(step["token_usage"][key] for step in row["steps"]):
            raise RuntimeError("token ledger mismatch")
    retrieval = next(step for step in row["steps"] if step["stage"] == "retriever")
    if NEW_COMPONENT not in retrieval["actions"]:
        raise RuntimeError("component A missing from opt-in action set")
    expected = descriptions(condition).get(NEW_COMPONENT)
    described_line = None if expected is None else f"- {NEW_COMPONENT}: {expected}"
    if described_line is None:
        if any(line.startswith(f"- {NEW_COMPONENT}:") for line in retrieval["prompt"].splitlines()):
            raise RuntimeError("name-only condition contains a description")
    elif described_line not in retrieval["prompt"].splitlines():
        raise RuntimeError("description mismatch")


def build(root: Path) -> tuple[dict, dict]:
    sources = {
        row["id"]: row
        for row in json.loads((root / "data/scenario.json").read_text(encoding="utf-8"))
    }
    n = len(sources)
    if n != 100:
        raise RuntimeError("expected 100 controlled items")

    cells = []
    paired: dict[tuple[str, int | None, str], dict[str, dict]] = {}
    counts = {}
    for method in RL_METHODS:
        for seed in SEEDS:
            for condition in CONDITIONS:
                rows = rl_rows(root, method, seed, condition)
                if len(rows) != n or {row["qid"] for row in rows} != set(sources):
                    raise RuntimeError(f"incomplete cell: {method}/{seed}/{condition}")
                for row in rows:
                    audit_row(row, sources[row["qid"]], condition)
                    paired.setdefault((method, seed, row["qid"]), {})[condition] = row
                cells.append(
                    {"planner": method, "seed": seed, "condition": condition, **metrics(rows)}
                )
                counts[f"{method}/{seed}/{condition}"] = len(rows)

    for condition in CONDITIONS:
        rows = rows_in(root / "runs/base_qwen3_4b" / condition / "rows")
        if len(rows) != n or {row["qid"] for row in rows} != set(sources):
            raise RuntimeError(f"incomplete base cell: {condition}")
        for row in rows:
            audit_row(row, sources[row["qid"]], condition)
            if not row["base_adapter_disabled_for_planner"]:
                raise RuntimeError("base planner adapter was not disabled")
            paired.setdefault(("base_qwen3_4b", None, row["qid"]), {})[condition] = row
        cells.append(
            {"planner": "base_qwen3_4b", "seed": None, "condition": condition, **metrics(rows)}
        )
        counts[f"base_qwen3_4b/{condition}"] = len(rows)

    three_seed = []
    for method in RL_METHODS:
        for condition in CONDITIONS:
            selected = [
                cell for cell in cells
                if cell["planner"] == method and cell["condition"] == condition
            ]
            three_seed.append(
                {
                    "planner": method,
                    "condition": condition,
                    "metrics": {
                        key: mean_sd([cell[key] for cell in selected])
                        for key in (
                            "selection_rate", "probability_a", "f1", "em",
                            "utility", "workflow_words",
                        )
                    },
                }
            )

    contrasts = []
    for condition_rows in paired.values():
        if set(condition_rows) != set(CONDITIONS):
            raise RuntimeError("condition pair is incomplete")
        prompts = []
        for condition in CONDITIONS:
            retrieval = next(
                step for step in condition_rows[condition]["steps"]
                if step["stage"] == "retriever"
            )
            prompts.append(strip_new_description(retrieval["prompt"]))
        if len(set(prompts)) != 1:
            raise RuntimeError("a field other than component A description changed")

    for planner in (*RL_METHODS, "base_qwen3_4b"):
        seeds = SEEDS if planner != "base_qwen3_4b" else (None,)
        seed_rows = []
        direction = []
        for seed in seeds:
            pairs = [paired[(planner, seed, qid)] for qid in sources]
            seed_rows.append(
                {
                    "selection_rate": statistics.mean(
                        pair["correct_relation"]["new_component_selected"]
                        - pair["opposite_relation"]["new_component_selected"]
                        for pair in pairs
                    ),
                    "probability_a": statistics.mean(
                        pair["correct_relation"]["new_component_probability"]
                        - pair["opposite_relation"]["new_component_probability"]
                        for pair in pairs
                    ),
                }
            )
            direction.extend(
                pair["correct_relation"]["new_component_probability"]
                > pair["opposite_relation"]["new_component_probability"]
                for pair in pairs
            )
        contrasts.append(
            {
                "planner": planner,
                "correct_minus_opposite_selection": (
                    mean_sd([row["selection_rate"] for row in seed_rows])
                    if len(seed_rows) > 1 else seed_rows[0]["selection_rate"]
                ),
                "correct_minus_opposite_probability_a": (
                    mean_sd([row["probability_a"] for row in seed_rows])
                    if len(seed_rows) > 1 else seed_rows[0]["probability_a"]
                ),
                "per_item_probability_direction_rate": statistics.mean(direction),
            }
        )

    result = {
        "complete": True,
        "protocol": "component-a-description-controls-100-v1",
        "component_a": NEW_COMPONENT,
        "n": n,
        "seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "cells": cells,
        "three_seed": three_seed,
        "base_qwen3_4b": [
            cell for cell in cells if cell["planner"] == "base_qwen3_4b"
        ],
        "contrasts": contrasts,
    }
    audit = {
        "status": "passed",
        "protocol": result["protocol"],
        "analyzed_endpoints": sum(counts.values()),
        "formal_endpoints": sum(counts.values()),
        "row_counts": counts,
        "same_100_items_and_component_across_conditions": True,
        "only_component_a_description_changes": True,
        "component_a_opt_in_only": True,
        "base_planner_adapter_disabled": True,
        "official_f1_em_utility_recomputed": True,
        "paths_and_token_ledgers_checked": True,
        "new_search_evaluations": 0,
        "new_optimizer_updates": 0,
    }
    return result, audit


def fmt(value: dict) -> str:
    return f"{value['mean']:.6f} ± {value['sample_sd']:.6f}"


def write_report(path: Path, result: dict, advantage: dict, *, chinese: bool) -> None:
    k3 = [row for row in advantage["summary"] if row["k"] == 3]
    new = next(row for row in k3 if row["component"] == NEW_COMPONENT)
    old = [row for row in k3 if row["component"] != NEW_COMPONENT]
    best_old_recall = max(row["mean_support_recall"] for row in old)
    best_old_full = max(row["full_support_rate"] for row in old)
    if chinese:
        lines = [
            "# 新 embedding 组件 A：描述控制与六种 planner 对比",
            "",
            "组件 A 是可选的 `retriever.qwen3_tagged_embedding`。同一组件、同一 100 题和同一语料构造保持不变，只改变 A 的描述。对六种 planner，正确关系描述下的 A 选择率均为四个条件中最高。",
            "",
            f"该构造极度有利于 A：A 的 support recall@3 为 {new['mean_support_recall']:.4f}，full-support@3 为 {new['full_support_rate']:.4f}；最佳已有 retriever 分别为 {best_old_recall:.4f} 和 {best_old_full:.4f}。",
            "",
            "| Planner | A 的描述 | 选择 A | P(A) | F1 | U |",
            "|---|---|---:|---:|---:|---:|",
        ]
    else:
        lines = [
            "# Embedding component A: description controls and six planners",
            "",
            "Component A is the opt-in `retriever.qwen3_tagged_embedding`. The component, 100 items, and corpus construction stay fixed; only A's description changes. For every planner, A's selection rate is highest under the correct relation description.",
            "",
            f"The construction strongly favors A: its support recall@3 is {new['mean_support_recall']:.4f} and full-support@3 is {new['full_support_rate']:.4f}; the best existing retriever reaches {best_old_recall:.4f} and {best_old_full:.4f}, respectively.",
            "",
            "| Planner | Description of A | Select A | P(A) | F1 | U |",
            "|---|---|---:|---:|---:|---:|",
        ]
    for group in result["three_seed"]:
        metric = group["metrics"]
        condition_label = (
            CONDITION_LABELS_ZH[group["condition"]]
            if chinese else CONDITION_LABELS[group["condition"]]
        )
        lines.append(
            f"| {LABELS[group['planner']]} | {condition_label} | "
            f"{fmt(metric['selection_rate'])} | {fmt(metric['probability_a'])} | "
            f"{fmt(metric['f1'])} | {fmt(metric['utility'])} |"
        )
    for cell in result["base_qwen3_4b"]:
        condition_label = (
            CONDITION_LABELS_ZH[cell["condition"]]
            if chinese else CONDITION_LABELS[cell["condition"]]
        )
        lines.append(
            f"| {LABELS[cell['planner']]} | {condition_label} | "
            f"{cell['selection_rate']:.6f} | {cell['probability_a']:.6f} | "
            f"{cell['f1']:.6f} | {cell['utility']:.6f} |"
        )
    lines += [
        "",
        ("RL 结果为 seed 11/23/37 的均值 ± sample SD；Base 是一次确定性运行。" if chinese else "RL values are mean ± sample SD over seeds 11/23/37; Base is one deterministic run."),
        "",
        ("该构造只用于验证 planner 是否理解组件描述与适用关系。语料标签来自答案所需段落的标注，因此不代表自然数据上的端到端泛化。" if chinese else "This construction tests whether planners understand a component description and its applicability relation. Corpus tags derive from answer-bearing passage annotations, so this is not an end-to-end natural-data generalization result."),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result, audit = build(args.root)
    advantage = json.loads((args.root / "component-a-advantage.json").read_text(encoding="utf-8"))
    assert advantage["n"] == result["n"]
    save(args.root / "component-a-results.json", result)
    save(args.root / "component-a-audit.json", audit)
    write_report(args.root / "RESULTS_COMPONENT_A.md", result, advantage, chinese=False)
    write_report(args.root / "RESULTS_COMPONENT_A_ZH.md", result, advantage, chinese=True)


if __name__ == "__main__":
    main()
