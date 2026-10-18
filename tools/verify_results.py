"""Verify the selectively migrated ICLR27 result package with stdlib only."""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]


def rows(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def compressed_rows(path: str) -> list[dict[str, str]]:
    with gzip.open(ROOT / path, mode="rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close(actual: float, expected: float, tolerance: float = 1e-9) -> None:
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance), (
        actual,
        expected,
    )


def verify_experiment_1() -> dict[str, float]:
    endpoint_rows = rows("results/experiment-1-main/endpoints.csv")
    item_rows = rows("results/experiment-1-main/evaluation_rows_official.csv")
    assert len(endpoint_rows) == 32
    assert len(item_rows) == 12_800
    assert {row["setting"] for row in endpoint_rows + item_rows} == {"matched"}

    by_cell: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in item_rows:
        by_cell[(row["method"], row["seed"], row["split"])].append(row)
        expected_utility = float(row["f1_official"]) - 0.1 * float(
            row["execution_words"]
        ) / 4096.0
        close(float(row["utility_official"]), expected_utility)
    assert len(by_cell) == 32
    assert all(len(cell) == 400 for cell in by_cell.values())
    assert all(len({row["qid"] for row in cell}) == 400 for cell in by_cell.values())

    for split in ("valid", "test"):
        qid_sets = [
            {row["qid"] for row in cell}
            for (method, seed, cell_split), cell in by_cell.items()
            if cell_split == split
        ]
        assert qid_sets and all(qids == qid_sets[0] for qids in qid_sets[1:])

    endpoint_index = {
        (row["method"], row["seed"], row["split"]): row for row in endpoint_rows
    }
    for key, cell in by_cell.items():
        endpoint = endpoint_index[key]
        close(mean(float(row["f1_official"]) for row in cell), float(endpoint["f1_official"]))
        close(
            mean(float(row["utility_official"]) for row in cell),
            float(endpoint["utility_official"]),
        )

    test_by_method: dict[str, list[float]] = defaultdict(list)
    for row in endpoint_rows:
        if row["split"] == "test":
            test_by_method[row["method"]].append(float(row["utility_official"]))
    means = {method: mean(values) for method, values in test_by_method.items()}
    assert means["AgentUCT"] > means["VinePPO"] > means["PPO"] > means["Base"]
    assert min(means[m] for m in ("Uniform", "UCT", "AgentUCT")) > means["VinePPO"]
    return means


def verify_experiment_2() -> dict[str, dict[str, float]]:
    summary = {
        row["method"]: {
            "rollout": float(row["rollout_utility_mean"]),
            "j_pull": float(row["j_pull_mean"]),
            "endpoint": float(row["endpoint_utility_mean"]),
        }
        for row in rows("results/experiment-2-search-attention/search_summary.csv")
        if row["method"] in {"Uniform", "UCT", "AgentUCT"}
    }
    assert summary["AgentUCT"]["rollout"] > summary["UCT"]["rollout"] > summary["Uniform"]["rollout"]
    assert summary["AgentUCT"]["j_pull"] > summary["UCT"]["j_pull"] > summary["Uniform"]["j_pull"]
    assert summary["Uniform"]["endpoint"] > summary["AgentUCT"]["endpoint"] > summary["UCT"]["endpoint"]
    assert max(v["endpoint"] for v in summary.values()) - min(
        v["endpoint"] for v in summary.values()
    ) < 0.001
    return summary


def verify_experiment_3() -> dict[str, float]:
    totals = rows("results/experiment-3-meta-token/cf_token_totals.csv")
    savings = rows("results/experiment-3-meta-token/cf_token_savings.csv")
    by_seed: dict[int, dict[str, int]] = defaultdict(dict)
    for row in totals:
        by_seed[int(row["seed"])][row["method"]] = int(row["qwen_tokens"])
    assert set(by_seed) == {11, 23, 37}
    assert all(values["AgentUCT"] < values["Uniform"] for values in by_seed.values())
    assert all(values["AgentUCT"] < values["UCT"] for values in by_seed.values())
    for row in savings:
        expected = int(row["control_tokens"]) - int(row["agentuct_tokens"])
        assert expected == int(row["saved_tokens"])
    return {
        method: mean(values[method] for values in by_seed.values())
        for method in ("Uniform", "UCT", "AgentUCT")
    }


def verify_experiment_4() -> None:
    root = ROOT / "results/experiment-4-component-generalization"
    audit = json.loads((root / "component-a-audit.json").read_text(encoding="utf-8"))
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    selection = json.loads(
        (root / "data/component-a-selection.json").read_text(encoding="utf-8")
    )
    assert audit["status"] == "passed"
    assert audit["formal_endpoints"] == sum(audit["row_counts"].values()) == 6400
    assert status["status"] == "complete"
    assert selection["n"] == len(selection["qids"]) == 100
    assert len(set(selection["qids"])) == 100
    endpoint_rows = compressed_rows(
        "results/experiment-4-component-generalization/endpoint_rows.csv.gz"
    )
    assert len(endpoint_rows) == 6400
    assert not any("static" in value.casefold() for row in endpoint_rows for value in row.values())
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in endpoint_rows:
        grouped[(row["planner"], row["seed"], row["condition"])].append(row)
        close(
            float(row["utility_official"]),
            float(row["f1_official"]) - 0.1 * float(row["execution_cost_words"]) / 4096.0,
        )
        assert int(row["new_search_evaluations"]) == 0
        assert int(row["new_optimizer_updates"]) == 0
    assert len(grouped) == 64 and all(len(cell) == 100 for cell in grouped.values())
    result = json.loads((root / "component-a-results.json").read_text(encoding="utf-8"))
    for cell in result["cells"]:
        seed = "" if cell["seed"] is None else str(cell["seed"])
        selected = grouped[(cell["planner"], seed, cell["condition"])]
        checks = {
            "selection_rate": mean(row["new_component_selected"].casefold() == "true" for row in selected),
            "probability_a": mean(float(row["new_component_probability"]) for row in selected),
            "f1": mean(float(row["f1_official"]) for row in selected),
            "em": mean(float(row["em_official"]) for row in selected),
            "utility": mean(float(row["utility_official"]) for row in selected),
            "workflow_words": mean(float(row["execution_cost_words"]) for row in selected),
        }
        for metric, value in checks.items():
            close(value, float(cell[metric]))


def verify_experiment_5() -> None:
    root = ROOT / "results/experiment-5-scenario-transfer"
    audit = json.loads((root / "audit-final.json").read_text(encoding="utf-8"))
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    wiki = json.loads((root / "data/two-wiki-selection.json").read_text(encoding="utf-8"))
    musique = json.loads((root / "data/musique-selection.json").read_text(encoding="utf-8"))
    assert audit["status"] == "passed"
    assert audit["formal_endpoints"] == sum(audit["row_counts"].values()) == 3200
    assert status["phase"] == "complete"
    assert summary["complete"] is True
    assert wiki["n"] == len(wiki["qids"]) == 100
    assert musique["n"] == len(musique["qids"]) == 100
    assert musique["selected_by_hop"] == {"2": 34, "3": 33, "4": 33}
    endpoint_rows = compressed_rows(
        "results/experiment-5-scenario-transfer/endpoint_rows.csv.gz"
    )
    assert len(endpoint_rows) == 3200
    assert not any("static" in value.casefold() for row in endpoint_rows for value in row.values())
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in endpoint_rows:
        grouped[(row["dataset"], row["method"], row["seed"])].append(row)
        close(
            float(row["utility_official"]),
            float(row["f1_official"]) - 0.1 * float(row["execution_cost_words"]) / 4096.0,
        )
        assert int(row["new_search_evaluations"]) == 0
        assert int(row["new_optimizer_updates"]) == 0
    assert len(grouped) == 32 and all(len(cell) == 100 for cell in grouped.values())
    for cell in summary["cells"]:
        selected = grouped[(cell["dataset"], cell["method"], str(cell["seed"]))]
        checks = {
            "f1": mean(float(row["f1_official"]) for row in selected),
            "em": mean(float(row["em_official"]) for row in selected),
            "utility": mean(float(row["utility_official"]) for row in selected),
            "workflow_words": mean(float(row["execution_cost_words"]) for row in selected),
        }
        for metric, value in checks.items():
            close(value, float(cell[metric]))
    base_musique = grouped[("musique", "base", "20260915")]
    hop_counts = {hop: sum(row["hop"] == hop for row in base_musique) for hop in ("2", "3", "4")}
    assert hop_counts == {"2": 34, "3": 33, "4": 33}


def main() -> None:
    tracked_artifacts = [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts]
    assert not any(path.suffix.casefold() in {".sqlite", ".db"} for path in tracked_artifacts)
    main_means = verify_experiment_1()
    verify_experiment_2()
    token_means = verify_experiment_3()
    verify_experiment_4()
    verify_experiment_5()
    print("migration verification: PASS")
    print("test utility means:", {k: round(v, 6) for k, v in main_means.items()})
    print("CF token means:", {k: round(v / 1e6, 3) for k, v in token_means.items()})


if __name__ == "__main__":
    main()
