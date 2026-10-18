"""Canonical component-level token tracing shared by search and RAG code."""

from __future__ import annotations

from collections.abc import Mapping

TOKEN_FIELDS = (
    "generator_4b_input_tokens",
    "generator_4b_output_tokens",
    "query_embedding_06b_input_tokens",
    "reranker_06b_input_tokens",
    "e5_input_tokens",
    "bge_m3_input_tokens",
    "actor_scoring_tokens",
    "index_build_tokens",
)
ONLINE_SEARCH_TOKEN_FIELDS = TOKEN_FIELDS[:6]
SEARCH_QWEN_TOKEN_FIELDS = (
    "generator_4b_input_tokens",
    "generator_4b_output_tokens",
    "query_embedding_06b_input_tokens",
    "reranker_06b_input_tokens",
)


def empty_token_usage() -> dict[str, int]:
    return {field: 0 for field in TOKEN_FIELDS}


def normalize_token_usage(value: Mapping[str, object] | None) -> dict[str, int]:
    value = value or {}
    output = empty_token_usage()
    for field in TOKEN_FIELDS:
        count = int(value.get(field, 0))
        if count < 0:
            raise ValueError(f"token count must be non-negative: {field}={count}")
        output[field] = count
    return output


def add_token_usage(*values: Mapping[str, object] | None) -> dict[str, int]:
    output = empty_token_usage()
    for value in values:
        normalized = normalize_token_usage(value)
        for field in TOKEN_FIELDS:
            output[field] += normalized[field]
    return output


def subtract_token_usage(
    total: Mapping[str, object], reused: Mapping[str, object]
) -> dict[str, int]:
    left = normalize_token_usage(total)
    right = normalize_token_usage(reused)
    output = {field: left[field] - right[field] for field in TOKEN_FIELDS}
    negative = {field: count for field, count in output.items() if count < 0}
    if negative:
        raise RuntimeError(f"reused token counters exceed terminal counters: {negative}")
    return output


def generator_token_count(value: Mapping[str, object] | None) -> int:
    normalized = normalize_token_usage(value)
    return (
        normalized["generator_4b_input_tokens"]
        + normalized["generator_4b_output_tokens"]
    )


def search_qwen_token_count(value: Mapping[str, object] | None) -> int:
    """Online Qwen cost shared by AgentUCT estimates and realized EndJ."""

    normalized = normalize_token_usage(value)
    return sum(normalized[field] for field in SEARCH_QWEN_TOKEN_FIELDS)


def online_search_raw_token_count(value: Mapping[str, object] | None) -> int:
    """Unweighted audit total; not a cross-model compute-equivalent quantity."""

    normalized = normalize_token_usage(value)
    return sum(normalized[field] for field in ONLINE_SEARCH_TOKEN_FIELDS)


def qwen_input_token_count(value: Mapping[str, object] | None) -> int:
    """Backward-compatible aggregate for generator, Qwen embed, and reranker."""

    normalized = normalize_token_usage(value)
    return sum(
        normalized[field]
        for field in (
            "generator_4b_input_tokens",
            "query_embedding_06b_input_tokens",
            "reranker_06b_input_tokens",
        )
    )


def legacy_token_usage(*, qwen_input_tokens: int, qwen_output_tokens: int) -> dict[str, int]:
    """Load an old collapsed ledger without claiming a reliable component split."""

    usage = empty_token_usage()
    usage["generator_4b_input_tokens"] = int(qwen_input_tokens)
    usage["generator_4b_output_tokens"] = int(qwen_output_tokens)
    return usage


__all__ = [
    "ONLINE_SEARCH_TOKEN_FIELDS",
    "SEARCH_QWEN_TOKEN_FIELDS",
    "TOKEN_FIELDS",
    "add_token_usage",
    "empty_token_usage",
    "generator_token_count",
    "legacy_token_usage",
    "normalize_token_usage",
    "online_search_raw_token_count",
    "qwen_input_token_count",
    "search_qwen_token_count",
    "subtract_token_usage",
]
