"""Protocol tests for test-time descriptions and an unseen tagged retriever."""

from __future__ import annotations

import unittest
from atppo_lab.dataset import HotpotItem
from atppo_lab.rag import RAGComponents, RAGWorkflowState
from atppo_lab.ragspace import (
    BM25,
    QWEN3_TAGGED_EMBEDDING,
    RETRIEVER_ACTIONS,
    TRUSTED_EVIDENCE_TAG,
    _tag_boosted_scores,
)
from uct_engine.wtb import RETRIEVER_ACTIONS as WTB_RETRIEVER_ACTIONS


class _Generator:
    def estimate_query_tokens(self, *_args):
        return 0


class _RAG:
    pass


def _item() -> HotpotItem:
    return HotpotItem(
        "q1",
        "Which passage contains the needed fact?",
        "gold-answer-must-not-leak",
        ("distractor", "target", "other"),
        (
            ("question-like distractor text",),
            (f"{TRUSTED_EVIDENCE_TAG} decisive target fact",),
            ("unrelated text",),
        ),
        ("target",),
    )


class ComponentGeneralizationTests(unittest.TestCase):
    def test_default_action_grammar_is_unchanged(self):
        state = RAGWorkflowState(
            _item(), 0.1, 4096.0, RAGComponents(_Generator(), _RAG())
        ).preview_child("query.identity")
        self.assertEqual(state.available_actions(), RETRIEVER_ACTIONS)
        self.assertNotIn(QWEN3_TAGGED_EMBEDDING, state.prompt())
        self.assertNotIn("Component descriptions", state.prompt())
        self.assertEqual(WTB_RETRIEVER_ACTIONS, RETRIEVER_ACTIONS)
        self.assertNotIn(QWEN3_TAGGED_EMBEDDING, WTB_RETRIEVER_ACTIONS)

    def test_test_time_description_exposes_unseen_component_without_gold(self):
        descriptions = {
            BM25: "Lexical matching for ordinary untagged corpora.",
            QWEN3_TAGGED_EMBEDDING: (
                f"Qwen query embedding with a metadata boost for {TRUSTED_EVIDENCE_TAG}."
            ),
        }
        state = RAGWorkflowState(
            _item(),
            0.1,
            4096.0,
            RAGComponents(
                _Generator(),
                _RAG(),
                retriever_actions=(*RETRIEVER_ACTIONS, QWEN3_TAGGED_EMBEDDING),
                action_descriptions=descriptions,
            ),
        ).preview_child("query.identity")
        prompt = state.prompt()
        self.assertIn(QWEN3_TAGGED_EMBEDDING, state.available_actions())
        self.assertIn("Component descriptions", prompt)
        self.assertIn(TRUSTED_EVIDENCE_TAG, prompt)
        self.assertNotIn(_item().answer, prompt)
        self.assertNotIn("supporting_titles", prompt)

    def test_tagged_query_embedding_metadata_boost_is_decisive(self):
        scores = _tag_boosted_scores(
            [0.99, -0.50, -0.75],
            (
                "distractor\nquestion-like distractor text",
                f"target\n{TRUSTED_EVIDENCE_TAG} decisive target fact",
                f"other\n{TRUSTED_EVIDENCE_TAG} second tagged fact",
            ),
        )
        self.assertGreater(scores[1], scores[0])
        self.assertGreater(scores[2], scores[0])


if __name__ == "__main__":
    unittest.main()
