"""Recompute endpoint, paired-QID, workflow, and auxiliary-ledger summaries.

Reads only files retained in this package. No inference or training is run.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_new_controls"
SCORER_PATH = ROOT / "tools/scoring/hotpot_evaluate_v1.py"
spec = importlib.util.spec_from_file_location("official_hotpotqa", SCORER_PATH)
assert spec and spec.loader
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
SEEDS = (11, 23, 37)
SPLITS = ("valid", "test")
TOKEN_FIELDS = ("generator_4b_input_tokens", "generator_4b_output_tokens",
                "query_embedding_06b_input_tokens", "reranker_06b_input_tokens",
                "e5_input_tokens", "bge_m3_input_tokens", "actor_scoring_tokens")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    values = list(values)
    return {"mean": mean(values), "sd": stdev(values) if len(values) > 1 else 0.0}


def official_values(rows):
    for row in rows:
        f1 = scorer.f1_score(row["prediction"], row["answer"])[0]
        words = float(row["execution_cost_words"])
        yield row["qid"], {"f1": f1, "utility": f1 - 0.1 * words / 4096,
                           "execution_words": words, "workflow_path": row["workflow_path"]}


def pattern(path):
    actions = list(path)
    query = next((x for x in actions if x.startswith("query.")), "missing")
    retrieval = next((x for x in actions if x.startswith("retriever.")), "missing")
    reranker = next((x for x in actions if x.startswith("reranker.")), "missing")
    return {"query": query, "retriever": retrieval, "reranker": reranker,
            "continuations": actions.count("control.continue"),
            "steps": len(actions)}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    source = ROOT / "new_controls"
    endpoints = []
    workflow_rows = []
    differences = []
    cost_rows = []
    all_rows = {}
    at2po_attestations = []
    paper_rows = list(csv.DictReader(
        (ROOT / "results/experiment-1-main/evaluation_rows_official.csv")
        .open(encoding="utf-8-sig", newline="")))
    paper_by_key = defaultdict(dict)
    for row in paper_rows:
        if row["split"] == "test" and row["method"] in ("Uniform", "AgentUCT", "VinePPO", "PPO"):
            paper_by_key[(row["method"], int(row["seed"]))][row["qid"]] = {
                "f1": float(row["f1_official"]), "utility": float(row["utility_official"]),
                "execution_words": float(row["execution_words"])}
    for method in ("actor_cf", "at2po"):
        for seed in SEEDS:
            metrics = read(source / method / f"seed-{seed}" / "metrics.json")
            assert metrics["status"] == "complete" and metrics["completed_iterations"] == 3
            tokens = Counter()
            searches = evaluations = 0
            cf_aux_actor = 0
            training_utility_sum = 0.0
            for iteration in metrics["iterations"]:
                search = iteration["search_tokens"]
                searches += int(search["searches"])
                evaluations += int(search["logical_evaluations"])
                training_utility_sum += float(search["cumulative_evaluation_utility"])
                tokens.update({k: int(search["token_usage"][k]) for k in TOKEN_FIELDS})
                if method == "actor_cf":
                    cf_aux_actor += int(search["cf_actor_auxiliary_actor_scoring_tokens"])
            auxiliary_actor = cf_aux_actor if method == "actor_cf" else tokens["actor_scoring_tokens"]
            environment = sum(tokens[k] for k in TOKEN_FIELDS if k != "actor_scoring_tokens")
            cost_rows.append({"method": method, "seed": seed, "searches": searches,
                              "logical_evaluations": evaluations,
                              **{k: tokens[k] for k in TOKEN_FIELDS},
                              "environment_tokens": environment,
                              "auxiliary_actor_tokens": auxiliary_actor,
                              "main_actor_tokens_within_search_summary":
                                  tokens["actor_scoring_tokens"] - auxiliary_actor,
                              "auxiliary_total_tokens": environment + auxiliary_actor,
                              "training_rollout_utility_sum": training_utility_sum,
                              "optimizer_actor_scoring_tokens": sum(
                                  int(x["optimizer"]["actor_scoring_tokens"])
                                  for x in metrics["iterations"])})
            for split in SPLITS:
                if method == "actor_cf":
                    payload = read(source / method / "official" / f"cf_actor-seed-{seed}-{split}.json")
                    rows = payload["rows"]
                    recorded = payload["summary"]
                else:
                    payload = read(source / method / "independent_eval" / f"seed-{seed}" / f"{split}.json")
                    attestation = payload["attestation"]
                    passed = (payload["trained"] is True and payload["n_rows"] == 400 and
                              attestation["trained"] is True and
                              attestation["checkpoint_method"] == "at2po" and
                              attestation["checkpoint_seed"] == seed and
                              attestation["checkpoint_iteration"] == 3 and
                              attestation["delta_w_is_exactly_zero"] is False)
                    at2po_attestations.append(passed)
                    assert passed, (seed, split)
                    rows = payload["summary"]["rows"]
                    recorded = payload["summary"]
                actual = dict(official_values(rows))
                assert len(rows) == len(actual) == 400
                assert all(math.isclose(v["utility"],
                    scorer.f1_score(r["prediction"],r["answer"])[0] -
                    0.1 * float(r["execution_cost_words"])/4096, abs_tol=1e-12)
                           for r in rows for v in [actual[r["qid"]]])
                all_rows[(method, seed, split)] = actual
                f1, utility, words = (mean(x[k] for x in actual.values())
                                      for k in ("f1", "utility", "execution_words"))
                if method == "actor_cf":
                    assert math.isclose(f1, recorded["f1_official"], abs_tol=1e-10)
                    assert math.isclose(utility, recorded["utility_official"], abs_tol=1e-10)
                endpoints.append({"method": method, "seed": seed, "split": split,
                                  "n_qids": len(actual), "f1_official": f1,
                                  "utility_official": utility, "execution_words": words,
                                  "reported_repo_f1": recorded.get("mean_f1", ""),
                                  "reported_repo_utility": recorded.get("mean_utility", "")})
                features = [pattern(v["workflow_path"]) for v in actual.values()]
                for label in ("query.decomposition", "retriever.bm25", "reranker.qwen3"):
                    key = "query" if label.startswith("query") else "retriever" if label.startswith("retriever") else "reranker"
                    workflow_rows.append({"method": method, "seed": seed, "split": split,
                                          "measure": label + "_rate",
                                          "value": mean(x[key] == label for x in features)})
                for label in ("continuations", "steps"):
                    workflow_rows.append({"method": method, "seed": seed, "split": split,
                                          "measure": "mean_" + label,
                                          "value": mean(x[label] for x in features)})

    for seed in SEEDS:
        for comparator, left, right in (
            ("actor_cf_minus_Uniform", all_rows[("actor_cf", seed, "test")], paper_by_key[("Uniform", seed)]),
            ("actor_cf_minus_AgentUCT", all_rows[("actor_cf", seed, "test")], paper_by_key[("AgentUCT", seed)]),
            ("actor_cf_minus_VinePPO", all_rows[("actor_cf", seed, "test")], paper_by_key[("VinePPO", seed)]),
            ("at2po_minus_AgentUCT", all_rows[("at2po", seed, "test")], paper_by_key[("AgentUCT", seed)]),
            ("at2po_minus_PPO", all_rows[("at2po", seed, "test")], paper_by_key[("PPO", seed)]),
            ("at2po_minus_actor_cf", all_rows[("at2po", seed, "test")], all_rows[("actor_cf", seed, "test")]),
        ):
            assert set(left) == set(right), (comparator, seed, len(set(left) & set(right)))
            for qid in sorted(left):
                differences.append({"comparison": comparator, "seed": seed, "qid": qid,
                                    "delta_f1": left[qid]["f1"]-right[qid]["f1"],
                                    "delta_utility": left[qid]["utility"]-right[qid]["utility"],
                                    "delta_execution_words": left[qid]["execution_words"]-right[qid]["execution_words"]})

    endpoint_index = {(r["method"], r["seed"], r["split"]): r for r in endpoints}
    for row in cost_rows:
        method, seed = row["method"], row["seed"]
        row["mean_rollout_utility"] = (row["training_rollout_utility_sum"] /
                                       row["logical_evaluations"])
        row["j_search"] = (row["training_rollout_utility_sum"] -
                            1e-4 * row["auxiliary_total_tokens"]) / row["logical_evaluations"]
        row["j_deploy_100k"] = (endpoint_index[(method, seed, "test")]["utility_official"] -
                                1e-4 * row["auxiliary_total_tokens"] / 100000)
    write_csv(OUT / "new_endpoint_by_seed.csv", endpoints)
    write_csv(OUT / "workflow_patterns_by_seed.csv", workflow_rows)
    write_csv(OUT / "paired_qid_differences.csv", differences)
    write_csv(OUT / "auxiliary_ledger_by_seed.csv", cost_rows)
    cf_splits = read(source / "actor_cf/metadata/splits.json")
    at_splits = read(source / "at2po/splits.json")
    paper_splits = read(ROOT / "results/experiment-1-main/qid_splits.json")
    protocol_audit = {
        "train_qid_order_identical": cf_splits["train"] == at_splits["train"],
        "train_qid_count": len(cf_splits["train"]),
        "actor_cf_matches_paper_matched_splits": {
            s: cf_splits[s] == paper_splits["matched"][s]
            for s in ("train", *SPLITS)},
        "independent_at2po_matches_actor_cf_qids": {
            s: all(set(all_rows[("at2po", seed, s)]) ==
                   set(all_rows[("actor_cf", seed, s)]) for seed in SEEDS)
            for s in SPLITS},
        "independent_at2po_matches_paper_test_qids": all(
            set(all_rows[("at2po", seed, "test")]) == set(paper_by_key[("AgentUCT", seed)])
            for seed in SEEDS),
        "all_six_at2po_checkpoint_attestations_passed": all(at2po_attestations),
    }
    assert protocol_audit["train_qid_order_identical"]
    assert all(protocol_audit["actor_cf_matches_paper_matched_splits"].values())
    assert all(protocol_audit["independent_at2po_matches_actor_cf_qids"].values())
    (OUT / "protocol_audit.json").write_text(
        json.dumps(protocol_audit, indent=2) + "\n", encoding="utf-8")
    workflow_agreement = {}
    for method in ("actor_cf", "at2po"):
        cells = {seed: all_rows[(method, seed, "test")] for seed in SEEDS}
        workflow_agreement[method] = {
            "unique_test_workflow_paths_by_seed": {
                str(seed): len({tuple(v["workflow_path"]) for v in cells[seed].values()})
                for seed in SEEDS},
            "exact_path_agreement_on_same_qids": {
                f"{left}-{right}": mean(
                    cells[left][qid]["workflow_path"] == cells[right][qid]["workflow_path"]
                    for qid in cells[left])
                for left, right in ((11, 23), (11, 37), (23, 37))}}
    (OUT / "workflow_agreement.json").write_text(
        json.dumps(workflow_agreement, indent=2) + "\n", encoding="utf-8")
    summary = {m: {split: {k: stats(x[k] for x in endpoints if x["method"] == m and x["split"] == split)
                           for k in ("f1_official", "utility_official", "execution_words")}
                   for split in SPLITS} for m in ("actor_cf", "at2po")}
    summary["paired_differences"] = {name: {
        "seed_means": {str(seed): mean(float(r["delta_utility"]) for r in differences
                                      if r["comparison"] == name and r["seed"] == seed)
                       for seed in SEEDS},
        "mean_and_sample_sd_over_seed_means": stats(
            mean(float(r["delta_utility"]) for r in differences
                 if r["comparison"] == name and r["seed"] == seed) for seed in SEEDS)}
        for name in {r["comparison"] for r in differences}}
    summary["auxiliary_total_tokens"] = {m: stats(r["auxiliary_total_tokens"] for r in cost_rows if r["method"] == m)
                                        for m in ("actor_cf", "at2po")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
