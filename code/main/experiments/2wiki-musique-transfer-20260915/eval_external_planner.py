"""Evaluate one frozen model or trained adapter as the workflow planner."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


class _BatchCap:
    def __init__(self, model, batch_size: int):
        self._model = model
        self._batch_size = batch_size

    def __getattr__(self, name):
        return getattr(self._model, name)

    def _call(self, method, *args, **kwargs):
        kwargs["batch_size"] = min(
            kwargs.get("batch_size", self._batch_size), self._batch_size
        )
        return getattr(self._model, method)(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self._call("encode", *args, **kwargs)

    def predict(self, *args, **kwargs):
        return self._call("predict", *args, **kwargs)


def cap_batches(ragspace, size: int = 4) -> None:
    for name in ("e5", "qwen_embedding", "bge_m3", "qwen_reranker"):
        setattr(ragspace, name, _BatchCap(getattr(ragspace, name), size))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def worker(job: dict) -> None:
    import torch
    from atppo_lab.dataset import _item
    from atppo_lab.official_metrics import exact_match_score, f1_score
    from atppo_lab.policy import PlannerPolicy
    from atppo_lab.rag import JSONTemplateGenerator, RAGComponents, RAGWorkflowState
    from atppo_lab.ragspace import RAGSpace
    from uct_engine import QIDPrefixCache, empty_token_usage, search_qwen_token_count

    output = Path(job["output"])
    source_rows = json.loads(Path(job["dataset"]).read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in source_rows}
    requested = list(job["qids"])
    assert len(source_rows) == len(by_id)
    assert len(requested) == len(set(requested)) and set(requested) <= set(by_id)
    remaining = []
    for qid in requested:
        path = output / "rows" / f"{qid}.json"
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            assert (
                previous["qid"],
                previous["method"],
                previous["seed"],
                previous["dataset"],
            ) == (qid, job["method"], job["seed"], job["dataset_name"])
            assert previous["answer"] == by_id[qid]["answer"]
            assert previous["candidate_documents"] == len(
                by_id[qid]["context"]["title"]
            )
        else:
            remaining.append(qid)
    if not remaining:
        print("RESUME_NO_NEW_WORK", job["dataset_name"], job["method"], job["seed"])
        return

    seed = int(job["seed"])
    policy = PlannerPolicy(Path(job["model"]), "cuda:0", seed=seed, learner=False)
    checkpoint_loaded = job["checkpoint"] is not None
    if checkpoint_loaded:
        payload = torch.load(
            job["checkpoint"], map_location="cpu", weights_only=True
        )
        assert payload["method"] == job["checkpoint_payload_method"]
        assert int(payload["seed"]) == seed
        assert payload["phase"] == "complete"
        assert int(payload["iteration"]) == 3
        assert len(payload["adapter"]) == 144
        policy.load_adapter_state(payload["adapter"])
        del payload
    else:
        assert job["method"] == "base"
        assert job["checkpoint_payload_method"] is None
        base_adapter = policy.adapter_state()
        base_b_tensors = [
            value for key, value in base_adapter.items() if "lora_B" in key
        ]
        assert base_b_tensors and all(
            not bool(torch.count_nonzero(value)) for value in base_b_tensors
        )
    policy.model.eval()
    assert policy.optimizer is None

    rag = RAGSpace(Path(job["rag_models"]), "cuda:0", index_root=output / "index")
    cap_batches(rag)
    components = RAGComponents(
        generator=JSONTemplateGenerator(policy),
        ragspace=rag,
    )
    cache = QIDPrefixCache(output / "wtb" / f"rank-{job['rank']}")
    try:
        for qid in remaining:
            started = time.monotonic()
            attempt = output / "attempts" / qid
            if attempt.exists():
                raise RuntimeError(
                    f"incomplete QID attempt needs cost reconciliation: {attempt}"
                )
            save(
                attempt / "start.json",
                {
                    "qid": qid,
                    "phase": "started",
                    "dataset": job["dataset_name"],
                    "method": job["method"],
                    "seed": seed,
                },
            )
            source = by_id[qid]
            item = _item(source)
            index_path = output / "index-work" / f"{qid}.json"
            index_usage = rag.prepare([item])
            if index_path.exists():
                assert not any(index_usage.values())
            else:
                save(index_path, index_usage)
            state = RAGWorkflowState(
                item=item,
                components=components,
                workflow_cost_lambda=0.1,
                execution_cost_reference=4096,
            )
            cache.begin_qid(state)
            usage = empty_token_usage()
            steps = []
            while not state.is_terminal():
                actions = list(state.available_actions())
                prompt = state.prompt()
                scoring_start = policy.actor_scoring_tokens
                probabilities, logprobs = policy.distribution(prompt, actions)
                chosen = max(range(len(actions)), key=probabilities.__getitem__)
                action = actions[chosen]
                actor_tokens = policy.actor_scoring_tokens - scoring_start
                next_state, reused = cache.advance(state, action)
                edge = {
                    key: (
                        0
                        if reused
                        else int(next_state.token_usage.get(key, 0))
                        - int(state.token_usage.get(key, 0))
                    )
                    for key in usage
                }
                edge["actor_scoring_tokens"] = actor_tokens
                assert all(value >= 0 for value in edge.values())
                for key in usage:
                    usage[key] += edge[key]
                step = {
                    "step": len(steps),
                    "stage": state.stage.value,
                    "prompt": prompt,
                    "actions": actions,
                    "probabilities": probabilities,
                    "logprobs": logprobs,
                    "chosen_label": action,
                    "token_usage": edge,
                    "wtb_reused": reused,
                    "observed_evidence_titles": [value.title for value in state.evidence],
                }
                steps.append(step)
                save(attempt / f"edge-{len(steps) - 1:02d}.json", step)
                state = next_state

            outcome = state.outcome()
            assert not any(step["wtb_reused"] for step in steps)
            for key in usage:
                if key != "actor_scoring_tokens":
                    assert usage[key] == int(outcome.token_usage.get(key, 0))
            golds = source.get("answer_aliases", [item.answer])
            assert item.answer in golds and golds
            f1 = max(f1_score(outcome.prediction, gold)[0] for gold in golds)
            em = max(
                float(exact_match_score(outcome.prediction, gold)) for gold in golds
            )
            result = {
                "protocol": "external-learned-planners-v2",
                "qid": qid,
                "dataset": job["dataset_name"],
                "method": job["method"],
                "role": job["role"],
                "seed": seed,
                "planner_mode": "greedy_tree_free",
                "checkpoint_loaded_as_planner": checkpoint_loaded,
                "base_adapter_effect_is_zero": not checkpoint_loaded,
                "question": item.question,
                "answer": item.answer,
                "answer_aliases": golds,
                "prediction": outcome.prediction,
                "path": list(state.path),
                "query_history": [list(values) for values in state.query_history],
                "selected_titles": [
                    item.context_titles[index]
                    for index in outcome.selected_document_indices
                ],
                "selected_document_indices": list(outcome.selected_document_indices),
                "f1_official": f1,
                "em_official": em,
                "utility_official": f1 - 0.1 * outcome.normalized_execution_cost,
                "execution_cost_words": outcome.execution_cost_words,
                "normalized_execution_cost": outcome.normalized_execution_cost,
                "support_recall": outcome.support_recall,
                "candidate_documents": len(item.context_titles),
                "token_usage": usage,
                **usage,
                "online_qwen_tokens": search_qwen_token_count(usage),
                "qwen_generation_attempts": outcome.qwen_generation_attempts,
                "qwen_retry_count": outcome.qwen_retry_count,
                "qwen_fallback_count": outcome.qwen_fallback_count,
                "qwen_length_retry_count": outcome.qwen_length_retry_count,
                "wtb_enabled": True,
                "new_search_evaluations": 0,
                "new_optimizer_updates": 0,
                "steps": steps,
                "elapsed_seconds": time.monotonic() - started,
            }
            save(output / "rows" / f"{qid}.json", result)
            save(
                attempt / "complete.json",
                {"qid": qid, "phase": "complete", "token_usage": usage},
            )
            print(
                json.dumps(
                    {
                        "qid": qid,
                        "dataset": job["dataset_name"],
                        "method": job["method"],
                        "seed": seed,
                        "f1": f1,
                        "utility": result["utility_official"],
                    }
                ),
                flush=True,
            )
    finally:
        cache.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    arguments = parser.parse_args()
    worker(json.loads(arguments.job.read_text(encoding="utf-8")))
