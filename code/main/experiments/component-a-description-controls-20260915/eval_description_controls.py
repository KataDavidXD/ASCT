"""Evaluate one frozen planner under single-component description ablations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from protocol import NEW_COMPONENT, descriptions


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
    from atppo_lab.ragspace import (
        QWEN3_TAGGED_EMBEDDING,
        RETRIEVER_ACTIONS,
        RAGSpace,
    )
    from uct_engine import QIDPrefixCache, empty_token_usage, search_qwen_token_count

    assert QWEN3_TAGGED_EMBEDDING == NEW_COMPONENT
    output_root = Path(job["output"])
    source_rows = json.loads(Path(job["dataset"]).read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in source_rows}
    seed = int(job["seed"])
    planner_kind = job["planner_kind"]
    policy = PlannerPolicy(Path(job["model"]), "cuda:0", seed=seed, learner=False)
    checkpoint_hash = None
    if planner_kind == "rl":
        checkpoint = torch.load(job["checkpoint"], map_location="cpu", weights_only=True)
        assert checkpoint["method"] == f"{job['method']}_atppo"
        assert int(checkpoint["seed"]) == seed and checkpoint["phase"] == "complete"
        policy.load_adapter_state(checkpoint["adapter"])
        del checkpoint
    elif planner_kind != "base":
        raise ValueError(f"unknown planner kind: {planner_kind}")
    policy.model.eval()

    for condition in job["conditions"]:
        output = output_root / condition
        requested = list(job["qids"])
        remaining = []
        for qid in requested:
            path = output / "rows" / f"{qid}.json"
            if path.exists():
                prior = json.loads(path.read_text(encoding="utf-8"))
                assert (
                    prior["qid"],
                    prior["method"],
                    prior["seed"],
                    prior["planner_kind"],
                    prior["condition"],
                ) == (qid, job["method"], seed, planner_kind, condition)
                assert prior["protocol"] == "component-description-controls-v1"
            else:
                remaining.append(qid)
        if not remaining:
            print("RESUME_NO_NEW_WORK", job["method"], seed, condition, flush=True)
            continue

        rag = RAGSpace(Path(job["rag_models"]), "cuda:0", index_root=output / "index")
        cap_batches(rag)
        components = RAGComponents(
            generator=JSONTemplateGenerator(policy),
            ragspace=rag,
            retriever_actions=(*RETRIEVER_ACTIONS, NEW_COMPONENT),
            action_descriptions=descriptions(condition),
        )
        cache = QIDPrefixCache(
            output / "wtb" / f"rank-{job['rank']}",
            extra_retriever_actions=(NEW_COMPONENT,),
        )
        try:
            for qid in remaining:
                started = time.monotonic()
                attempt = output / "attempts" / qid
                if attempt.exists():
                    raise RuntimeError(
                        f"incomplete attempt needs reconciliation: {attempt}"
                    )
                save(attempt / "start.json", {"qid": qid, "phase": "started"})
                source = by_id[qid]
                item = _item(source)
                index_usage = rag.prepare([item])
                save(output / "index-work" / f"{qid}.json", index_usage)
                state = RAGWorkflowState(item, 0.1, 4096.0, components)
                cache.begin_qid(state)
                usage = empty_token_usage()
                steps = []
                new_probability = None
                while not state.is_terminal():
                    actions = list(state.available_actions())
                    prompt = state.prompt()
                    scoring_start = policy.actor_scoring_tokens
                    if planner_kind == "base":
                        with policy.model.disable_adapter():
                            probabilities, logprobs = policy.distribution(prompt, actions)
                    else:
                        probabilities, logprobs = policy.distribution(prompt, actions)
                    chosen = max(range(len(actions)), key=probabilities.__getitem__)
                    action = actions[chosen]
                    if state.stage.value == "retriever":
                        new_probability = probabilities[actions.index(NEW_COMPONENT)]
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
                    }
                    steps.append(step)
                    save(attempt / f"edge-{len(steps) - 1:02d}.json", step)
                    state = next_state

                outcome = state.outcome()
                assert not any(step["wtb_reused"] for step in steps)
                golds = source.get("answer_aliases", [item.answer])
                f1 = max(f1_score(outcome.prediction, gold)[0] for gold in golds)
                em = max(
                    float(exact_match_score(outcome.prediction, gold)) for gold in golds
                )
                retriever_action = next(
                    step["chosen_label"]
                    for step in steps
                    if step["stage"] == "retriever"
                )
                result = {
                    "protocol": "component-description-controls-v1",
                    "qid": qid,
                    "original_qid": source["original_id"],
                    "method": job["method"],
                    "seed": seed,
                    "planner_kind": planner_kind,
                    "condition": condition,
                    "base_adapter_disabled_for_planner": planner_kind == "base",
                    "question": item.question,
                    "answer": item.answer,
                    "prediction": outcome.prediction,
                    "path": list(state.path),
                    "retriever_action": retriever_action,
                    "new_component_selected": retriever_action == NEW_COMPONENT,
                    "new_component_probability": new_probability,
                    "f1_official": f1,
                    "em_official": em,
                    "utility_official": f1 - 0.1 * outcome.normalized_execution_cost,
                    "execution_cost_words": outcome.execution_cost_words,
                    "normalized_execution_cost": outcome.normalized_execution_cost,
                    "support_recall": outcome.support_recall,
                    "selected_titles": [
                        item.context_titles[index]
                        for index in outcome.selected_document_indices
                    ],
                    "token_usage": usage,
                    "online_qwen_tokens": search_qwen_token_count(usage),
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
                            "method": job["method"],
                            "seed": seed,
                            "condition": condition,
                            "retriever": retriever_action,
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
