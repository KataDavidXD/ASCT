from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from uct_engine import empty_token_usage

from atppo_lab.evaluation import (
    endj_static,
    endj_test_search,
    evaluate_run,
    observed_utility_token_diagnostics,
)


class EndJEvaluationTests(unittest.TestCase):
    def test_observed_auc_and_cumulative_ucb_diagnostics(self) -> None:
        def row(
            utility: float,
            best: float,
            field: str,
            tokens: int,
            cumulative: int,
        ) -> dict[str, object]:
            usage = empty_token_usage()
            usage[field] = tokens
            return {
                "utility": utility,
                "best_utility_so_far": best,
                "cumulative_qwen_tokens": cumulative,
                "token_usage": usage,
            }

        diagnostics = observed_utility_token_diagnostics(
            [
                [
                    row(0.2, 0.2, "generator_4b_input_tokens", 10, 10),
                    row(0.5, 0.5, "query_embedding_06b_input_tokens", 5, 15),
                    row(0.4, 0.5, "reranker_06b_input_tokens", 5, 20),
                ],
                [
                    row(0.7, 0.7, "generator_4b_input_tokens", 0, 0),
                    row(0.6, 0.7, "generator_4b_input_tokens", 0, 0),
                ],
            ],
            beta=0.01,
        )
        self.assertEqual(diagnostics["logical_evaluations"], 5)
        self.assertEqual(diagnostics["qwen_tokens"], 20)
        self.assertAlmostEqual(diagnostics["utility_sum"], 2.4)
        self.assertAlmostEqual(diagnostics["mean_utility"], 0.48)
        self.assertAlmostEqual(diagnostics["mean_rollout_utility"], 0.48)
        self.assertAlmostEqual(diagnostics["cumulative_rollout_net_utility"], 2.2)
        self.assertAlmostEqual(diagnostics["mean_rollout_net_utility"], 0.44)
        self.assertAlmostEqual(diagnostics["observed_utility_token_area"], 7.0)
        self.assertAlmostEqual(diagnostics["observed_utility_token_auc"], 0.35)
        self.assertAlmostEqual(
            diagnostics["mean_search_observed_utility_token_auc"], 0.35
        )
        self.assertEqual(diagnostics["observed_utility_token_auc_searches"], 1)
        self.assertEqual(diagnostics["zero_qwen_token_searches"], 1)

    def test_endj_formulas_keep_beta_unchanged(self) -> None:
        static = endj_static(
            endpoint_utility_sum=8.0,
            search_qwen_tokens=2_000,
            searches=10,
            beta=1e-4,
        )
        deploy = endj_test_search(
            test_mean_utility=0.2,
            train_search_qwen_tokens=40_000_000,
            deployment_horizon=100_000,
            beta=1e-4,
        )
        self.assertAlmostEqual(static["value"], 0.78)
        self.assertAlmostEqual(deploy["value"], 0.16)
        self.assertEqual(static["beta"], 1e-4)
        self.assertEqual(deploy["beta"], 1e-4)

    def test_completed_run_uses_test_utility_and_online_qwen_cost(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            usage = empty_token_usage()
            usage.update(
                generator_4b_input_tokens=800,
                generator_4b_output_tokens=200,
                query_embedding_06b_input_tokens=9_000,
                reranker_06b_input_tokens=5_000,
                e5_input_tokens=200,
                actor_scoring_tokens=7_000,
            )
            self._write_result(
                root,
                "agentuct_atppo",
                search_summary={
                    "searches": 10,
                    "logical_evaluations": 120,
                    "cumulative_evaluation_utility": 60.0,
                    "cumulative_endpoint_utility": 7.0,
                    "wtb_modes": {"full_miss": 1},
                    "token_usage": usage,
                    **usage,
                },
                test_utility=0.3,
            )
            self._write_result(
                root,
                "ppo",
                search_summary={
                    "searches": 0,
                    "logical_evaluations": 0,
                    "cumulative_evaluation_utility": 0.0,
                    "cumulative_endpoint_utility": 0.0,
                    "wtb_modes": {},
                    "token_usage": empty_token_usage(),
                    **empty_token_usage(),
                },
                test_utility=0.25,
            )
            evaluation = evaluate_run(
                root,
                methods=("agentuct_atppo", "ppo"),
                seeds=(11,),
                deployment_horizon=100,
            )
            agent = evaluation["method_aggregates"][0]
            self.assertEqual(agent["train_search_generator_tokens"], 1_000)
            self.assertEqual(agent["train_search_qwen_tokens"], 15_000)
            self.assertEqual(agent["train_online_search_raw_tokens"], 15_200)
            self.assertAlmostEqual(agent["endj_static"]["value"], 0.55)
            self.assertAlmostEqual(agent["endj_test_search"]["value"], 0.285)
            self.assertAlmostEqual(
                evaluation["deployment_vs_ppo_comparisons"][0]["left_minus_right"],
                0.035,
            )
            self.assertEqual(
                evaluation["deployment_vs_ppo_comparisons"][0][
                    "left_minus_right_train_search_qwen_tokens"
                ],
                15_000,
            )

    def test_completed_run_reconstructs_observed_curve_from_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_usage = empty_token_usage()
            first_usage["generator_4b_input_tokens"] = 10
            second_usage = empty_token_usage()
            second_usage["generator_4b_output_tokens"] = 5
            summary_usage = empty_token_usage()
            summary_usage["generator_4b_input_tokens"] = 10
            summary_usage["generator_4b_output_tokens"] = 5
            self._write_result(
                root,
                "agentuct_atppo",
                search_summary={
                    "searches": 1,
                    "logical_evaluations": 2,
                    "cumulative_evaluation_utility": 0.8,
                    "cumulative_endpoint_utility": 0.6,
                    "wtb_modes": {"full_miss": 2},
                    "token_usage": summary_usage,
                    **summary_usage,
                },
                test_utility=0.3,
            )
            trace = [
                {
                    "utility": 0.2,
                    "best_utility_so_far": 0.2,
                    "cumulative_qwen_tokens": 10,
                    "token_usage": first_usage,
                },
                {
                    "utility": 0.6,
                    "best_utility_so_far": 0.6,
                    "cumulative_qwen_tokens": 15,
                    "token_usage": second_usage,
                },
            ]
            cell_dir = root / "agentuct_atppo-seed-11"
            (cell_dir / "trajectories.jsonl").write_text(
                json.dumps({"steps": [{"search_trace": trace}]}) + "\n",
                encoding="utf-8",
            )

            evaluation = evaluate_run(
                root,
                methods=("agentuct_atppo",),
                seeds=(11,),
                deployment_horizon=100,
            )
            cell = evaluation["cells"][0]["internal_rollout_diagnostics"]
            aggregate = evaluation["method_aggregates"][0][
                "internal_rollout_diagnostics"
            ]
            self.assertAlmostEqual(cell["observed_utility_token_area"], 5.0)
            self.assertAlmostEqual(cell["observed_utility_token_auc"], 1.0 / 3.0)
            self.assertAlmostEqual(cell["cumulative_rollout_net_utility"], 0.7985)
            self.assertTrue(cell["summary_search_count_matches"])
            self.assertTrue(cell["summary_evaluation_count_matches"])
            self.assertTrue(cell["summary_qwen_tokens_match"])
            self.assertTrue(cell["summary_utility_sum_matches"])
            self.assertAlmostEqual(aggregate["observed_utility_token_auc"], 1.0 / 3.0)

    @staticmethod
    def _write_result(
        root: Path,
        method: str,
        *,
        search_summary: dict[str, object],
        test_utility: float,
    ) -> None:
        cell = root / f"{method}-seed-11"
        cell.mkdir()
        test_usage = empty_token_usage()
        test_usage["generator_4b_input_tokens"] = 10
        payload = {
            "method": method,
            "seed": 11,
            "status": "complete",
            "token_beta": 1e-4,
            "search_cost_mode": "agentuct-Bellman",
            "iterations": [
                {
                    "search_tokens": search_summary,
                    "optimizer": {"actor_scoring_tokens": 3},
                }
            ],
            "test": {
                "mean_utility": test_utility,
                "mean_f1": test_utility,
                "token_usage": test_usage,
                **test_usage,
            },
        }
        (cell / "run_result.json").write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
