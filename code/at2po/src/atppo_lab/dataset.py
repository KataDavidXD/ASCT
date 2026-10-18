"""Small, readable HotpotQA FullWiki split loader."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HotpotItem:
    qid: str
    question: str
    answer: str
    context_titles: tuple[str, ...]
    context_sentences: tuple[tuple[str, ...], ...]
    supporting_titles: tuple[str, ...]


def _rows(paths: list[Path], indices: list[int]) -> list[dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    requested = set(indices)
    selected: dict[int, dict[str, Any]] = {}
    offset = 0
    for path in paths:
        metadata = pq.read_metadata(path)
        local = sorted(index - offset for index in requested if offset <= index < offset + metadata.num_rows)
        if local:
            table = pq.read_table(path)
            taken = table.take(pa.array(local, type=pa.int64())).to_pylist()
            selected.update({offset + index: row for index, row in zip(local, taken, strict=True)})
        offset += metadata.num_rows
    return [selected[index] for index in indices]


def _item(row: dict[str, Any]) -> HotpotItem:
    context = row["context"]
    support = row["supporting_facts"]
    return HotpotItem(
        qid=str(row["id"]),
        question=str(row["question"]),
        answer=str(row["answer"]),
        context_titles=tuple(str(value) for value in context["title"]),
        context_sentences=tuple(
            tuple(str(sentence) for sentence in sentences)
            for sentences in context["sentences"]
        ),
        supporting_titles=tuple(dict.fromkeys(str(value) for value in support["title"])),
    )


def load_experiment_splits(
    dataset_root: Path,
    *,
    train_size: int,
    valid_size: int,
    test_size: int,
    seed: int = 2027,
) -> dict[str, list[HotpotItem]]:
    """Return fixed disjoint subsets without content hashes or manifests."""

    root = dataset_root.resolve()
    train_paths = sorted((root / "fullwiki").glob("train-*.parquet"))
    validation_paths = sorted((root / "fullwiki").glob("validation-*.parquet"))
    if not train_paths or not validation_paths:
        raise FileNotFoundError(f"FullWiki parquet shards were not found under {root}")
    if min(train_size, valid_size, test_size) <= 0:
        raise ValueError("train/valid/test sizes must be positive")

    import pyarrow.parquet as pq

    train_count = sum(pq.read_metadata(path).num_rows for path in train_paths)
    validation_count = sum(pq.read_metadata(path).num_rows for path in validation_paths)
    if train_size > train_count or valid_size + test_size > validation_count:
        raise ValueError("requested split is larger than the available FullWiki rows")

    train_indices = list(range(train_count))
    validation_indices = list(range(validation_count))
    random.Random(seed).shuffle(train_indices)
    random.Random(seed + 1).shuffle(validation_indices)

    chosen_train = train_indices[:train_size]
    chosen_valid = validation_indices[:valid_size]
    chosen_test = validation_indices[valid_size : valid_size + test_size]
    return {
        "train": [_item(row) for row in _rows(train_paths, chosen_train)],
        "valid": [_item(row) for row in _rows(validation_paths, chosen_valid)],
        "test": [_item(row) for row in _rows(validation_paths, chosen_test)],
    }


def load_manifest_splits(dataset_root: Path, manifest: Path) -> dict[str, list[HotpotItem]]:
    """Load exact ordered QIDs from {train, valid, test}; preserve source rows."""
    ids = json.loads(manifest.read_text(encoding="utf-8"))
    roles = ("train", "valid", "test")
    if not isinstance(ids, dict) or set(ids) != set(roles) or any(
        not isinstance(ids[role], list) or not ids[role]
        or any(not isinstance(qid, str) or not qid for qid in ids[role])
        for role in roles
    ):
        raise ValueError("split manifest must contain three nonempty ordered QID lists")
    ordered = [qid for role in roles for qid in ids[role]]
    if len(ordered) != len(set(ordered)):
        raise ValueError("manifest QIDs repeat within/across splits")
    import pyarrow.parquet as pq

    paths = sorted((dataset_root / "fullwiki").glob("train-*.parquet")) + sorted(
        (dataset_root / "fullwiki").glob("validation-*.parquet")
    )
    requested = set(ordered)
    indices: dict[str, int] = {}
    offset = 0
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(columns=["id"]):
            for local, value in enumerate(batch.column(0).to_pylist()):
                qid = str(value)
                if qid in requested:
                    if qid in indices:
                        raise ValueError(f"duplicate QID in source parquet: {qid}")
                    indices[qid] = offset + local
            offset += batch.num_rows
    missing = requested - indices.keys()
    if missing:
        raise ValueError(f"manifest QIDs missing from source: {sorted(missing)[:10]}")
    selected = _rows(paths, [indices[qid] for qid in ordered])
    items = {qid: _item(row) for qid, row in zip(ordered, selected, strict=True)}
    return {role: [items[qid] for qid in ids[role]] for role in roles}
