"""Render English and Chinese reports from the audited 100-item summary."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LABELS = {
    "base": "Base Qwen3-4B",
    "ppo": "PPO",
    "vineppo": "VinePPO",
    "agentuct": "AgentUCT-RL",
    "uniform": "Uniform-RL",
    "uct": "UCT-RL",
}


def fmt(value: float | dict) -> str:
    if isinstance(value, dict):
        return f"{value['mean']:.6f} ± {value['sample_sd']:.6f}"
    return f"{value:.6f}"


def render(*, chinese: bool) -> str:
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    if chinese:
        lines = [
            "# 2Wiki 与 MuSiQue 各 100 题迁移测试",
            "",
            "2Wiki 固定为 100 道未见的 bridge_comparison；MuSiQue 固定为 100 题，其中 2/3/4-hop 为 34/33/33。",
            "",
            "| 数据集 | Planner | F1 | EM | U | Workflow words |",
            "|---|---|---:|---:|---:|---:|",
        ]
    else:
        lines = [
            "# 100-item 2Wiki and MuSiQue transfer evaluation",
            "",
            "The fixed cohorts contain 100 unseen 2Wiki bridge_comparison items and 100 MuSiQue items split 34/33/33 across 2/3/4-hop.",
            "",
            "| Dataset | Planner | F1 | EM | U | Workflow words |",
            "|---|---|---:|---:|---:|---:|",
        ]
    dataset_labels = {
        "two-wiki-bridge-comparison": "2Wiki bridge_comparison",
        "musique": "MuSiQue",
    }
    for entry in summary["summaries"]:
        metric = entry["metrics"]
        lines.append(
            f"| {dataset_labels[entry['dataset']]} | {LABELS[entry['method']]} | "
            f"{fmt(metric['f1'])} | {fmt(metric['em'])} | {fmt(metric['utility'])} | "
            f"{fmt(metric['workflow_words'])} |"
        )

    lines += [
        "",
        ("## MuSiQue 分 hop" if chinese else "## MuSiQue by hop"),
        "",
        "| Planner | Hop | F1 | U |",
        "|---|---:|---:|---:|",
    ]
    for entry in summary["summaries"]:
        if entry["dataset"] != "musique":
            continue
        for hop in ("2", "3", "4"):
            metric = entry["by_hop"][hop]
            lines.append(
                f"| {LABELS[entry['method']]} | {hop} | {fmt(metric['f1'])} | {fmt(metric['utility'])} |"
            )

    lines += [
        "",
        (
            "训练方法为 seed 11/23/37 的均值 ± sample SD；Base 是一次确定性运行。所有 planner 使用同一冻结生成器、RAG 组件、QID 和评分协议。"
            if chinese
            else "Trained planners are reported as mean ± sample SD over seeds 11/23/37; Base is one deterministic run. Every planner uses the same frozen generator, RAG components, QIDs, and scoring protocol."
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    (ROOT / "RESULTS.md").write_text(render(chinese=False), encoding="utf-8")
    (ROOT / "RESULTS_ZH.md").write_text(render(chinese=True), encoding="utf-8")


if __name__ == "__main__":
    main()
