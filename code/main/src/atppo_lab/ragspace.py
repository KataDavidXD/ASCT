"""Real retrieval and reranking components used by the RAG workflow."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from uct_engine import (
    add_token_usage,
    empty_token_usage,
    qwen_input_token_count,
)

from .dataset import HotpotItem

BM25 = "retriever.bm25"
DENSE_E5 = "retriever.dense_e5"
BGE_M3_HYBRID = "retriever.bge_m3_hybrid"
QWEN3_EMBEDDING = "retriever.qwen3_embedding"
RRF = "retriever.rrf"
RETRIEVER_ACTIONS = (BM25, DENSE_E5, BGE_M3_HYBRID, QWEN3_EMBEDDING, RRF)

# Test-time capability probe. This action is intentionally excluded from
# RETRIEVER_ACTIONS so existing training and evaluation grammars do not change.
QWEN3_TAGGED_EMBEDDING = "retriever.qwen3_tagged_embedding"
TRUSTED_EVIDENCE_TAG = "[[trusted-evidence]]"
_TRUSTED_TAG_BOOST = 1000.0

SELECT_RELEVANCE = "selection.relevance"
SELECT_MMR = "selection.mmr"
SELECTION_ACTIONS = (SELECT_RELEVANCE, SELECT_MMR)


@dataclass(frozen=True)
class RankedDocument:
    index: int
    score: float


@dataclass(frozen=True)
class RankingResult:
    documents: tuple[RankedDocument, ...]
    token_usage: dict[str, int] = field(default_factory=empty_token_usage)

    @property
    def qwen_input_tokens(self) -> int:
        return qwen_input_token_count(self.token_usage)


@dataclass(frozen=True)
class RerankResult:
    documents: tuple[RankedDocument, ...]
    token_usage: dict[str, int] = field(default_factory=empty_token_usage)

    @property
    def qwen_input_tokens(self) -> int:
        return qwen_input_token_count(self.token_usage)


@dataclass(frozen=True)
class RAGModelPaths:
    e5: Path
    bge_m3: Path
    qwen_embedding: Path
    qwen_reranker: Path

    @classmethod
    def from_root(cls, root: Path) -> "RAGModelPaths":
        root = root.resolve()

        def one(prefix: str) -> Path:
            matches = sorted(path for path in root.glob(f"{prefix}*") if path.is_dir())
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"expected one {prefix!r} model directory under {root}, found {matches}"
                )
            return matches[0]

        return cls(
            e5=one("e5-base-v2"),
            bge_m3=one("bge-m3"),
            qwen_embedding=one("Qwen3-Embedding-0.6B"),
            qwen_reranker=one("Qwen3-Reranker-0.6B"),
        )


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", text.casefold()))


def _documents(item: HotpotItem) -> list[str]:
    return [
        f"{title}\n{' '.join(item.context_sentences[index])}"
        for index, title in enumerate(item.context_titles)
    ]


def _bm25_index(documents: tuple[str, ...]) -> dict[str, Any]:
    tokenized = tuple(_tokens(document) for document in documents)
    document_frequency: Counter[str] = Counter()
    for document in tokenized:
        document_frequency.update(set(document))
    return {
        "tokenized": tokenized,
        "average_length": sum(map(len, tokenized)) / max(1, len(tokenized)),
        "document_frequency": dict(document_frequency),
    }


def _bm25_scores(query: str, index: dict[str, Any]) -> list[float]:
    query_tokens = _tokens(query)
    tokenized = index["tokenized"]
    average_length = float(index["average_length"])
    document_frequency = index["document_frequency"]
    scores: list[float] = []
    for document in tokenized:
        frequencies = Counter(document)
        score = 0.0
        for token in set(query_tokens):
            frequency = frequencies.get(token, 0)
            if not frequency:
                continue
            idf = math.log(
                1.0
                + (len(tokenized) - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
            )
            denominator = frequency + 1.5 * (
                1.0 - 0.75 + 0.75 * len(document) / max(1.0, average_length)
            )
            score += idf * frequency * 2.5 / denominator
        scores.append(score)
    return scores


def _ordered(scores: list[float]) -> tuple[RankedDocument, ...]:
    return tuple(
        RankedDocument(index=index, score=float(score))
        for index, score in sorted(
            enumerate(scores), key=lambda row: (-float(row[1]), row[0])
        )
    )


def _tag_boosted_scores(
    semantic_scores: list[float], documents: tuple[str, ...]
) -> list[float]:
    """Combine Qwen embedding similarity with the probe's trusted metadata."""

    if len(semantic_scores) != len(documents):
        raise ValueError("one semantic score is required per document")
    return [
        float(score)
        + _TRUSTED_TAG_BOOST
        * float(document.casefold().count(TRUSTED_EVIDENCE_TAG))
        for score, document in zip(semantic_scores, documents, strict=True)
    ]


def _input_tokens(tokenizer: Any, texts: list[str], *, max_length: int) -> int:
    if not texts:
        return 0
    encoded = tokenizer(
        texts,
        padding=False,
        truncation=True,
        max_length=max_length,
    )
    return sum(len(values) for values in encoded["input_ids"])


class RAGSpace:
    """One shared, replaceable model bundle per GPU worker."""

    def __init__(
        self, model_root: Path, device: str, *, index_root: Path | None = None
    ) -> None:
        import torch
        from FlagEmbedding import BGEM3FlagModel
        from sentence_transformers import CrossEncoder, SentenceTransformer

        paths = RAGModelPaths.from_root(model_root)
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        model_kwargs = {"dtype": dtype, "attn_implementation": "sdpa"}
        self.device = device
        self.e5 = SentenceTransformer(
            str(paths.e5),
            device=device,
            local_files_only=True,
            model_kwargs=model_kwargs,
        )
        self.bge_m3 = BGEM3FlagModel(
            str(paths.bge_m3),
            use_fp16=False,
            use_bf16=str(device).startswith("cuda"),
            devices=device,
            trust_remote_code=True,
            batch_size=16,
            query_max_length=256,
            passage_max_length=512,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        self.qwen_embedding = SentenceTransformer(
            str(paths.qwen_embedding),
            device=device,
            local_files_only=True,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
        )
        self.qwen_reranker = CrossEncoder(
            str(paths.qwen_reranker),
            device=device,
            local_files_only=True,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
            max_length=1024,
        )
        self.index_root = index_root.resolve() if index_root is not None else None
        if self.index_root is not None:
            self.index_root.mkdir(parents=True, exist_ok=True)
        self._document_index_cache: dict[str, dict[str, Any]] = {}
        self._ranking_cache: dict[
            tuple[str, tuple[str, ...], str], tuple[RankedDocument, ...]
        ] = {}
        self._rerank_cache: dict[
            tuple[str, str, tuple[int, ...]], tuple[RankedDocument, ...]
        ] = {}
        self.index_build_tokens = 0

    def _bge_tokenizer(self) -> Any:
        tokenizer = getattr(self.bge_m3, "tokenizer", None)
        if tokenizer is None:
            tokenizer = getattr(getattr(self.bge_m3, "model", None), "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("BGE-M3 tokenizer is unavailable for token tracing")
        return tokenizer

    def _index_build_token_count(self, documents: list[str]) -> int:
        return (
            _input_tokens(
                self.e5.tokenizer,
                [f"passage: {document}" for document in documents],
                max_length=int(getattr(self.e5, "max_seq_length", 512) or 512),
            )
            + _input_tokens(
                self.qwen_embedding.tokenizer,
                documents,
                max_length=int(
                    getattr(self.qwen_embedding, "max_seq_length", 8192) or 8192
                ),
            )
            + _input_tokens(self._bge_tokenizer(), documents, max_length=512)
        )

    def offload_models(self) -> None:
        """Free GPU headroom while the Actor performs its PPO update."""
        if not str(self.device).startswith("cuda"):
            return
        import gc
        import torch

        self.e5.to("cpu")
        self.bge_m3.model.to("cpu")
        self.qwen_embedding.to("cpu")
        self.qwen_reranker.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def restore_models(self) -> None:
        """Move the RAG models back for evaluation and the next collection."""
        if not str(self.device).startswith("cuda"):
            return
        self.e5.to(self.device)
        self.bge_m3.model.to(self.device)
        self.qwen_embedding.to(self.device)
        self.qwen_reranker.to(self.device)

    def _build_document_index(self, item: HotpotItem) -> dict[str, Any]:
        import numpy as np

        documents = tuple(_documents(item))
        self.index_build_tokens += self._index_build_token_count(list(documents))
        e5_vectors = self.e5.encode(
            [f"passage: {document}" for document in documents],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        qwen_vectors = self.qwen_embedding.encode(
            list(documents),
            prompt="",
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        bge = self.bge_m3.encode(
            list(documents),
            batch_size=16,
            max_length=512,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return {
            "documents": documents,
            "bm25": _bm25_index(documents),
            "e5_vectors": np.asarray(e5_vectors),
            "qwen_vectors": np.asarray(qwen_vectors),
            "bge_dense_vectors": np.asarray(bge["dense_vecs"]),
            "bge_lexical_weights": bge["lexical_weights"],
        }

    def _document_index(self, item: HotpotItem) -> dict[str, Any]:
        cached = self._document_index_cache.get(item.qid)
        if cached is not None:
            return cached
        path = (
            self.index_root / f"{item.qid}.pt" if self.index_root is not None else None
        )
        if path is not None and path.exists():
            import torch

            cached = torch.load(path, map_location="cpu", weights_only=False)
        else:
            cached = self._build_document_index(item)
            if path is not None:
                import torch

                torch.save(cached, path)
        self._document_index_cache[item.qid] = cached
        return cached

    def prepare(self, items: list[HotpotItem]) -> dict[str, int]:
        """Cold-build every missing document index in three batched model calls."""

        import numpy as np
        import torch

        pending = [
            item
            for item in items
            if item.qid not in self._document_index_cache
            and (
                self.index_root is None
                or not (self.index_root / f"{item.qid}.pt").exists()
            )
        ]
        if not pending:
            return empty_token_usage()
        document_groups = [tuple(_documents(item)) for item in pending]
        documents = [document for group in document_groups for document in group]
        build_tokens = self._index_build_token_count(documents)
        self.index_build_tokens += build_tokens
        e5_vectors = self.e5.encode(
            [f"passage: {document}" for document in documents],
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        qwen_vectors = self.qwen_embedding.encode(
            documents,
            batch_size=32,
            prompt="",
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        bge = self.bge_m3.encode(
            documents,
            batch_size=16,
            max_length=512,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        offset = 0
        for item, group in zip(pending, document_groups, strict=True):
            end = offset + len(group)
            index = {
                "documents": group,
                "bm25": _bm25_index(group),
                "e5_vectors": np.asarray(e5_vectors[offset:end]),
                "qwen_vectors": np.asarray(qwen_vectors[offset:end]),
                "bge_dense_vectors": np.asarray(bge["dense_vecs"][offset:end]),
                "bge_lexical_weights": bge["lexical_weights"][offset:end],
            }
            self._document_index_cache[item.qid] = index
            if self.index_root is not None:
                torch.save(index, self.index_root / f"{item.qid}.pt")
            offset = end
        usage = empty_token_usage()
        usage["index_build_tokens"] = build_tokens
        return usage

    @staticmethod
    def _fuse_query_rankings(
        rankings: list[tuple[RankedDocument, ...]], *, rrf_k: int = 60
    ) -> tuple[RankedDocument, ...]:
        scores: Counter[int] = Counter()
        for ranking in rankings:
            for rank, document in enumerate(ranking, start=1):
                scores[document.index] += 1.0 / (rrf_k + rank)
        return _ordered([scores[index] for index in range(len(scores))])

    def _rank_one(
        self, item: HotpotItem, query: str, mode: str
    ) -> tuple[RankedDocument, ...]:
        index = self._document_index(item)
        import numpy as np

        if mode == BM25:
            return _ordered(_bm25_scores(query, index["bm25"]))
        if mode == DENSE_E5:
            query_vector = self.e5.encode(
                [f"query: {query}"],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
            return _ordered((index["e5_vectors"] @ query_vector).tolist())
        if mode == QWEN3_EMBEDDING:
            query_vector = self.qwen_embedding.encode(
                [query],
                prompt_name="query",
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
            return _ordered((index["qwen_vectors"] @ query_vector).tolist())
        if mode == QWEN3_TAGGED_EMBEDDING:
            tagged_query = (
                f"Retrieve authoritative passages marked {TRUSTED_EVIDENCE_TAG}. "
                f"Question: {query}"
            )
            query_vector = self.qwen_embedding.encode(
                [tagged_query],
                prompt_name="query",
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
            semantic_scores = (index["qwen_vectors"] @ query_vector).tolist()
            return _ordered(
                _tag_boosted_scores(semantic_scores, index["documents"])
            )
        if mode == BGE_M3_HYBRID:
            query_values = self.bge_m3.encode(
                [query],
                batch_size=1,
                max_length=256,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            dense = (
                np.asarray(index["bge_dense_vectors"], dtype=float)
                @ np.asarray(query_values["dense_vecs"], dtype=float)[0]
            )
            sparse = np.asarray(
                self.bge_m3.compute_lexical_matching_score(
                    query_values["lexical_weights"],
                    index["bge_lexical_weights"],
                ),
                dtype=float,
            )[0]
            return _ordered(((dense + sparse) * 0.5).tolist())
        raise ValueError(f"unknown base retriever: {mode}")

    def rank(
        self, item: HotpotItem, queries: tuple[str, ...], mode: str
    ) -> RankingResult:
        if not queries or any(not query.strip() for query in queries):
            raise ValueError("retrieval requires non-empty queries")
        key = (item.qid, queries, mode)
        cached = self._ranking_cache.get(key)
        if cached is not None:
            return RankingResult(cached)
        usage = empty_token_usage()
        if mode == RRF:
            results = [
                self.rank(item, queries, base)
                for base in (BM25, DENSE_E5, BGE_M3_HYBRID, QWEN3_EMBEDDING)
            ]
            ranking = self._fuse_query_rankings(
                [result.documents for result in results], rrf_k=60
            )
            usage = add_token_usage(*(result.token_usage for result in results))
        else:
            per_query = [self._rank_one(item, query, mode) for query in queries]
            ranking = (
                per_query[0]
                if len(per_query) == 1
                else self._fuse_query_rankings(per_query, rrf_k=60)
            )
            usage = self.estimate_rank_token_usage(
                item, queries, mode, ignore_cache=True
            )
        self._ranking_cache[key] = ranking
        return RankingResult(ranking, usage)

    def estimate_rank_token_usage(
        self,
        item: HotpotItem,
        queries: tuple[str, ...],
        mode: str,
        *,
        ignore_cache: bool = False,
    ) -> dict[str, int]:
        usage = empty_token_usage()
        if mode == BM25:
            return usage
        if mode == RRF:
            return add_token_usage(
                *(
                    self.estimate_rank_token_usage(
                        item, queries, base, ignore_cache=ignore_cache
                    )
                    for base in (BM25, DENSE_E5, BGE_M3_HYBRID, QWEN3_EMBEDDING)
                )
            )
        key = (item.qid, queries, mode)
        if not ignore_cache and key in self._ranking_cache:
            return usage
        if mode == DENSE_E5:
            usage["e5_input_tokens"] = _input_tokens(
                self.e5.tokenizer,
                [f"query: {query}" for query in queries],
                max_length=int(getattr(self.e5, "max_seq_length", 512) or 512),
            )
        elif mode == BGE_M3_HYBRID:
            usage["bge_m3_input_tokens"] = _input_tokens(
                self._bge_tokenizer(), list(queries), max_length=256
            )
        elif mode in (QWEN3_EMBEDDING, QWEN3_TAGGED_EMBEDDING):
            prompt = self.qwen_embedding.prompts.get("query", "")
            values = (
                [
                    f"Retrieve authoritative passages marked {TRUSTED_EVIDENCE_TAG}. "
                    f"Question: {query}"
                    for query in queries
                ]
                if mode == QWEN3_TAGGED_EMBEDDING
                else list(queries)
            )
            usage["query_embedding_06b_input_tokens"] = _input_tokens(
                self.qwen_embedding.tokenizer,
                [f"{prompt}{query}" for query in values],
                max_length=8192,
            )
        else:
            raise ValueError(f"unknown retriever: {mode}")
        return usage

    def estimate_rank_tokens(
        self, item: HotpotItem, queries: tuple[str, ...], mode: str
    ) -> int:
        # Keep beta_tok in its historical Qwen-token unit. The exact E5/BGE
        # counts are traced but do not silently change the allocation policy.
        return qwen_input_token_count(
            self.estimate_rank_token_usage(item, queries, mode)
        )

    def _e5_document_vectors(self, item: HotpotItem) -> Any:
        return self._document_index(item)["e5_vectors"]

    def select(
        self,
        item: HotpotItem,
        ranking: tuple[RankedDocument, ...],
        *,
        mode: str,
        top_k: int,
    ) -> tuple[RankedDocument, ...]:
        if mode == SELECT_RELEVANCE:
            return ranking[:top_k]
        if mode != SELECT_MMR:
            raise ValueError(f"unknown retrieval selection mode: {mode}")
        import numpy as np

        # MMR selects from every candidate supplied by the FullWiki row.
        candidates = list(ranking)
        scores = np.asarray([document.score for document in candidates], dtype=float)
        span = float(scores.max() - scores.min()) if len(scores) else 0.0
        relevance = (scores - scores.min()) / span if span > 0 else np.ones_like(scores)
        vectors = self._e5_document_vectors(item)
        chosen: list[int] = []
        remaining = list(range(len(candidates)))
        while remaining and len(chosen) < top_k:
            best_position = max(
                remaining,
                key=lambda position: (
                    0.5 * float(relevance[position])
                    - 0.5
                    * max(
                        (
                            float(
                                vectors[candidates[position].index]
                                @ vectors[candidates[previous].index]
                            )
                            for previous in chosen
                        ),
                        default=0.0,
                    ),
                    -candidates[position].index,
                ),
            )
            chosen.append(best_position)
            remaining.remove(best_position)
        return tuple(candidates[position] for position in chosen)

    def rerank(
        self, item: HotpotItem, question: str, document_indices: tuple[int, ...]
    ) -> RerankResult:
        key = (item.qid, question, document_indices)
        if not document_indices:
            return RerankResult(())
        cached = self._rerank_cache.get(key)
        if cached is not None:
            return RerankResult(cached)
        documents = self._document_index(item)["documents"]
        pairs = [(question, documents[index]) for index in document_indices]
        scores = self.qwen_reranker.predict(
            pairs,
            batch_size=8,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        ranking = tuple(
            RankedDocument(index=index, score=float(score))
            for index, score in sorted(
                zip(document_indices, scores, strict=True),
                key=lambda row: (-float(row[1]), row[0]),
            )
        )
        self._rerank_cache[key] = ranking
        usage = empty_token_usage()
        usage["reranker_06b_input_tokens"] = self.estimate_rerank_tokens(
            item, question, document_indices, ignore_cache=True
        )
        return RerankResult(ranking, usage)

    def estimate_rerank_tokens(
        self,
        item: HotpotItem,
        question: str,
        document_indices: tuple[int, ...],
        *,
        ignore_cache: bool = False,
    ) -> int:
        if not document_indices:
            return 0
        key = (item.qid, question, document_indices)
        if not ignore_cache and key in self._rerank_cache:
            return 0
        documents = self._document_index(item)["documents"]
        prompt_name = self.qwen_reranker.default_prompt_name
        prompt = (
            self.qwen_reranker.prompts.get(prompt_name, "")
            if prompt_name is not None
            else ""
        )
        encoded = self.qwen_reranker.tokenizer(
            [f"{prompt}{question}" for _ in document_indices],
            [documents[index] for index in document_indices],
            padding=False,
            truncation=True,
            max_length=1024,
        )
        return sum(len(values) for values in encoded["input_ids"])


__all__ = [
    "BGE_M3_HYBRID",
    "BM25",
    "DENSE_E5",
    "QWEN3_EMBEDDING",
    "RAGModelPaths",
    "RAGSpace",
    "RankingResult",
    "RerankResult",
    "RRF",
    "RETRIEVER_ACTIONS",
    "QWEN3_TAGGED_EMBEDDING",
    "TRUSTED_EVIDENCE_TAG",
    "_tag_boosted_scores",
    "RankedDocument",
    "SELECT_MMR",
    "SELECT_RELEVANCE",
    "SELECTION_ACTIONS",
]
