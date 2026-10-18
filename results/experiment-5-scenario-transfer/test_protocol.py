"""Artifact-level checks for the fixed 100-item transfer protocol."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_fixed_cohorts() -> None:
    wiki = load("data/two-wiki-selection.json")
    musique = load("data/musique-selection.json")
    assert wiki["n"] == len(wiki["qids"]) == 100
    assert len(set(wiki["qids"])) == 100
    assert wiki["existing_50_preserved"] is True
    assert musique["n"] == len(musique["qids"]) == 100
    assert len(set(musique["qids"])) == 100
    assert musique["selected_by_hop"] == {"2": 34, "3": 33, "4": 33}
    assert musique["existing_51_preserved"] is True


def test_formal_result_contract() -> None:
    status = load("status.json")
    audit = load("audit-final.json")
    summary = load("summary.json")
    assert status["phase"] == "complete"
    assert status["formal_rows"] == 3200
    assert audit["status"] == "passed"
    assert audit["formal_endpoints"] == sum(audit["row_counts"].values()) == 3200
    assert audit["official_f1_em_utility_recomputed"] is True
    assert audit["paths_and_token_ledgers_checked"] is True
    assert audit["new_search_evaluations"] == audit["new_optimizer_updates"] == 0
    assert summary["complete"] is True
    assert len(summary["cells"]) == 32
    assert len(summary["summaries"]) == 12


def test_scope_is_transfer_only() -> None:
    forbidden = ("dis" + "tractor", "component" + "-a", "component" + "_a")
    for path in ROOT.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path == Path(__file__).resolve():
            continue
        if path.suffix not in {".py", ".json", ".md"}:
            continue
        text = path.read_text(encoding="utf-8").casefold()
        assert not any(term in text for term in forbidden), path
