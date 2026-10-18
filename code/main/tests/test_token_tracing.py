from __future__ import annotations

import unittest
from typing import ClassVar

from uct_engine import (
    TOKEN_FIELDS,
    add_token_usage,
    generator_token_count,
    online_search_raw_token_count,
    search_qwen_token_count,
    subtract_token_usage,
)

from atppo_lab.dataset import HotpotItem
from atppo_lab.rag import JSONTemplateGenerator
from atppo_lab.ragspace import DENSE_E5, QWEN3_EMBEDDING, RAGSpace


class _Tokenizer:
    def __call__(self, texts, **_kwargs):
        return {"input_ids": [[0] * len(text.split()) for text in texts]}


class _Embedding:
    tokenizer = _Tokenizer()
    max_seq_length = 512
    prompts: ClassVar[dict[str, str]] = {"query": "query: "}


class _BGE:
    tokenizer = _Tokenizer()


class _UnusedGeneratorModel:
    def generate_grounded_answer(self, *_args, **_kwargs):
        raise AssertionError("empty evidence must not call the generator")

    def estimate_grounded_answer_tokens(self, *_args, **_kwargs):
        raise AssertionError("empty evidence must not call the estimator")


class TokenTracingTests(unittest.TestCase):
    def test_usage_arithmetic_preserves_all_declared_fields(self) -> None:
        left = {
            "generator_4b_input_tokens": 10,
            "query_embedding_06b_input_tokens": 5,
            "e5_input_tokens": 3,
        }
        right = {
            "generator_4b_output_tokens": 4,
            "reranker_06b_input_tokens": 6,
            "actor_scoring_tokens": 9,
        }
        total = add_token_usage(left, right)
        self.assertEqual(set(total), set(TOKEN_FIELDS))
        self.assertEqual(generator_token_count(total), 14)
        self.assertEqual(search_qwen_token_count(total), 25)
        self.assertEqual(online_search_raw_token_count(total), 28)
        self.assertEqual(subtract_token_usage(total, left), add_token_usage(right))

    def test_retriever_token_counts_are_separate_and_cache_aware(self) -> None:
        ragspace = object.__new__(RAGSpace)
        ragspace.e5 = _Embedding()
        ragspace.qwen_embedding = _Embedding()
        ragspace.bge_m3 = _BGE()
        ragspace._ranking_cache = {}
        item = HotpotItem("q", "question", "answer", (), (), ())

        e5 = ragspace.estimate_rank_token_usage(
            item, ("alpha beta",), DENSE_E5
        )
        qwen = ragspace.estimate_rank_token_usage(
            item, ("alpha beta",), QWEN3_EMBEDDING
        )
        self.assertGreater(e5["e5_input_tokens"], 0)
        self.assertEqual(e5["query_embedding_06b_input_tokens"], 0)
        self.assertGreater(qwen["query_embedding_06b_input_tokens"], 0)

        ragspace._ranking_cache[(item.qid, ("alpha beta",), DENSE_E5)] = ()
        self.assertEqual(
            ragspace.estimate_rank_token_usage(item, ("alpha beta",), DENSE_E5)[
                "e5_input_tokens"
            ],
            0,
        )

    def test_empty_evidence_skips_grounded_generator(self) -> None:
        generator = JSONTemplateGenerator(_UnusedGeneratorModel())
        generated = generator.generate("question", ())
        self.assertEqual(generated.text, "UNKNOWN")
        self.assertEqual(generated.input_tokens + generated.output_tokens, 0)
        self.assertEqual(generator.estimate_tokens("question", ()), 0)

    def test_empty_rerank_skips_model_access(self) -> None:
        ragspace = object.__new__(RAGSpace)
        item = HotpotItem("q", "question", "answer", (), (), ())
        result = ragspace.rerank(item, "question", ())
        self.assertEqual(result.documents, ())
        self.assertEqual(sum(result.token_usage.values()), 0)


if __name__ == "__main__":
    unittest.main()
