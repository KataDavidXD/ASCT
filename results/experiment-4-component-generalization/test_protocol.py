import json
import unittest
from pathlib import Path

from protocol import (
    COMMON_DESCRIPTIONS,
    CONDITIONS,
    NEW_COMPONENT,
    TAG,
    descriptions,
)


ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


class ProtocolTests(unittest.TestCase):
    def test_only_new_component_description_changes(self):
        values = {condition: descriptions(condition) for condition in CONDITIONS}
        for condition, mapping in values.items():
            old = {key: value for key, value in mapping.items() if key != NEW_COMPONENT}
            self.assertEqual(old, COMMON_DESCRIPTIONS, condition)
        self.assertNotIn(NEW_COMPONENT, values["name_only"])

    def test_four_semantic_conditions(self):
        correct = descriptions("correct_relation")[NEW_COMPONENT]
        type_only = descriptions("type_only")[NEW_COMPONENT]
        opposite = descriptions("opposite_relation")[NEW_COMPONENT]
        self.assertIn("Qwen3 query-embedding", correct)
        self.assertIn(TAG, correct)
        self.assertIn("tagged passages before untagged", correct)
        self.assertIn("Qwen3 query-embedding", type_only)
        self.assertNotIn(TAG, type_only)
        self.assertNotIn("tagged passages before untagged", type_only)
        self.assertIn(TAG, opposite)
        self.assertIn("untagged", opposite)
        self.assertIn("before passages marked", opposite)

    def test_fixed_100_item_cohort(self):
        selection = load("data/component-a-selection.json")
        self.assertEqual(selection["n"], 100)
        self.assertEqual(len(selection["qids"]), 100)
        self.assertEqual(len(set(selection["qids"])), 100)
        self.assertTrue(selection["existing_40_preserved"])

    def test_complete_six_planner_results(self):
        result = load("component-a-results.json")
        audit = load("component-a-audit.json")
        status = load("status.json")
        self.assertTrue(result["complete"])
        self.assertEqual(result["n"], 100)
        self.assertEqual(len(result["cells"]), 64)
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(audit["formal_endpoints"], 6400)
        self.assertEqual(sum(audit["row_counts"].values()), 6400)
        self.assertTrue(audit["only_component_a_description_changes"])
        self.assertTrue(audit["component_a_opt_in_only"])
        self.assertEqual(status["formal_endpoints"], 6400)
        self.assertEqual(
            status["planners"],
            ["ppo", "vineppo", "agentuct", "uniform", "uct", "base_qwen3_4b"],
        )


if __name__ == "__main__":
    unittest.main()
