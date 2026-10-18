"""Build the fixed 100-item 2Wiki and MuSiQue transfer cohorts."""

from __future__ import annotations

import argparse
import json
import random
import re
import zipfile
from pathlib import Path


SEED = 20260915

def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.casefold()).split())


def convert_wiki(row: dict, aliases: dict[str, list[str]]) -> dict:
    golds = sorted(set([row["answer"], *aliases.get(row["answer_id"], [])]))
    return {
        "id": "2wiki-" + row["_id"],
        "original_id": row["_id"],
        "question": row["question"],
        "answer": row["answer"],
        "answer_aliases": golds,
        "type": row["type"],
        "context": {
            "title": [entry[0] for entry in row["context"]],
            "sentences": [entry[1] for entry in row["context"]],
        },
        "supporting_facts": {
            "title": [entry[0] for entry in row["supporting_facts"]],
            "sent_id": [entry[1] for entry in row["supporting_facts"]],
        },
    }


def expand_wiki(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq

    with zipfile.ZipFile(args.wiki_source) as bundle:
        dev_name = next(name for name in bundle.namelist() if Path(name).name == "dev.json")
        alias_name = next(name for name in bundle.namelist() if Path(name).name == "id_aliases.json")
        rows = json.loads(bundle.read(dev_name))
        aliases = {}
        for line in bundle.read(alias_name).decode("utf-8").splitlines():
            record = json.loads(line)
            aliases[record["Q_id"]] = sorted(set(record["aliases"] + record["demonyms"]))

    prior_ids = {
        row["original_id"]
        for row in json.loads(args.wiki_prior_400.read_text(encoding="utf-8"))
    }
    old = json.loads(args.wiki_existing.read_text(encoding="utf-8"))
    old_by_id = {row["original_id"]: row for row in old}
    hotpot_questions = set()
    for path in args.hotpot_parquets.glob("*.parquet"):
        for batch in pq.ParquetFile(path).iter_batches(columns=["question"]):
            hotpot_questions.update(norm(value) for value in batch.column(0).to_pylist())

    eligible = []
    seen_questions = set()
    for row in rows:
        question = norm(row.get("question", ""))
        if (
            row.get("type") != "bridge_comparison"
            or row["_id"] in prior_ids
            or question in hotpot_questions
            or question in seen_questions
            or not row.get("answer")
            or not row.get("context")
            or "answer_id" not in row
        ):
            continue
        seen_questions.add(question)
        eligible.append(row)
    values = sorted(eligible, key=lambda row: row["_id"])
    random.Random(f"{SEED}:2wiki:bridge_comparison:unseen").shuffle(values)
    selected = sorted(values[:100], key=lambda row: row["_id"])
    converted = [convert_wiki(row, aliases) for row in selected]
    if len(converted) != 100 or not set(old_by_id) <= {row["original_id"] for row in converted}:
        raise RuntimeError("2Wiki expansion did not preserve the original cohort")
    for row in converted:
        if row["original_id"] in old_by_id and row != old_by_id[row["original_id"]]:
            raise RuntimeError("an existing 2Wiki item changed")
    save(args.output / "transfer/data/two-wiki-bridge-comparison.json", converted)
    save(
        args.output / "transfer/data/two-wiki-selection.json",
        {
            "dataset": "2WikiMultihopQA",
            "source_split": "dev",
            "sampling_seed": SEED,
            "selection": "100 deterministic bridge_comparison examples",
            "prior_evaluation_qids_excluded": len(prior_ids),
            "existing_50_preserved": True,
            "n": 100,
            "qids": [row["id"] for row in converted],
        },
    )


def convert_musique(row: dict) -> dict:
    found = re.fullmatch(r"([234])hop[^_]*__.+", row["id"])
    if not found:
        raise ValueError(row["id"])
    hop = int(found.group(1))
    supporting = [paragraph for paragraph in row["paragraphs"] if paragraph["is_supporting"]]
    aliases = list(dict.fromkeys([row["answer"], *row.get("answer_aliases", [])]))
    return {
        "id": "musique-" + row["id"],
        "original_id": row["id"],
        "question": row["question"],
        "answer": row["answer"],
        "answer_aliases": aliases,
        "type": f"{hop}hop",
        "hop": hop,
        "context": {
            "title": [paragraph["title"] for paragraph in row["paragraphs"]],
            "sentences": [[paragraph["paragraph_text"]] for paragraph in row["paragraphs"]],
        },
        "supporting_facts": {
            "title": [paragraph["title"] for paragraph in supporting],
            "sent_id": [0] * len(supporting),
        },
        "official_support_indices": [paragraph["idx"] for paragraph in supporting],
    }


def expand_musique(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in args.musique_source.read_text(encoding="utf-8").splitlines() if line]
    groups = {hop: [] for hop in (2, 3, 4)}
    for row in rows:
        found = re.fullmatch(r"([234])hop[^_]*__.+", row["id"])
        if not found:
            continue
        hop = int(found.group(1))
        if row.get("answerable") and row.get("answer") and len(row.get("question_decomposition", [])) == hop and row.get("paragraphs"):
            groups[hop].append(row)
    targets = {2: 34, 3: 33, 4: 33}
    chosen = []
    for hop in (2, 3, 4):
        values = sorted(groups[hop], key=lambda row: row["id"])
        random.Random(f"{SEED}:musique:{hop}hop").shuffle(values)
        chosen.extend(values[: targets[hop]])
    chosen.sort(key=lambda row: (int(row["id"][0]), row["id"]))
    converted = [convert_musique(row) for row in chosen]
    old = json.loads(args.musique_existing.read_text(encoding="utf-8"))
    old_by_id = {row["original_id"]: row for row in old}
    if len(converted) != 100 or not set(old_by_id) <= {row["original_id"] for row in converted}:
        raise RuntimeError("MuSiQue expansion did not preserve the original cohort")
    for row in converted:
        if row["original_id"] in old_by_id and row != old_by_id[row["original_id"]]:
            raise RuntimeError("an existing MuSiQue item changed")
    save(args.output / "transfer/data/musique.json", converted)
    save(
        args.output / "transfer/data/musique-selection.json",
        {
            "dataset": "MuSiQue-Answerable v1.0",
            "source_split": "dev",
            "sampling_seed": SEED,
            "selected_by_hop": {str(key): value for key, value in targets.items()},
            "existing_51_preserved": True,
            "n": 100,
            "qids": [row["id"] for row in converted],
        },
    )



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wiki-source", type=Path, required=True)
    parser.add_argument("--wiki-prior-400", type=Path, required=True)
    parser.add_argument("--wiki-existing", type=Path, required=True)
    parser.add_argument("--hotpot-parquets", type=Path, required=True)
    parser.add_argument("--musique-source", type=Path, required=True)
    parser.add_argument("--musique-existing", type=Path, required=True)
    args = parser.parse_args()
    expand_wiki(args)
    expand_musique(args)


if __name__ == "__main__":
    main()
