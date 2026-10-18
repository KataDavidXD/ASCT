"""CPU integration of the real collection/update/checkpoint loop with toy models."""

import json
import math
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from atppo_lab.at2po import AT2POConfig
from atppo_lab.experiment import CellSpec
from tests.test_vine import State


def test_collection_update_resume_and_tree_free_evaluation(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("wtb")
    from atppo_lab import experiment as exp

    @dataclass(frozen=True)
    class FullState(State):
        query_history = ()

        def restore(self, data):
            return FullState(data["qid"], tuple(data["path"]))

        def outcome(self):
            base = super().outcome()
            values = vars(base) | {
                "execution_cost_words": 4096.0,
                "support_recall": 1.0,
                "qwen_input_tokens": self.token_usage["generator_4b_input_tokens"],
                "qwen_output_tokens": self.token_usage["generator_4b_output_tokens"],
                "selected_document_indices": (),
                "prediction": "answer",
            }
            return SimpleNamespace(**values)

    class ToyPolicy:
        def __init__(self, model_path, device, seed, learning_rate=1e-5):
            self.torch = torch
            self.device = torch.device("cpu")
            self.model = torch.nn.Linear(1, 2, bias=False)
            self.model.weight.data.zero_()
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=learning_rate
            )
            self.actor_scoring_tokens = 0

        def batch_action_logprobs(self, prompts, labels, grad):
            self.actor_scoring_tokens += len(prompts)
            return [self.model(torch.ones(1))[: len(actions)] for actions in labels]

        def distribution(self, prompt, labels):
            with torch.no_grad():
                logs = torch.log_softmax(
                    self.batch_action_logprobs([prompt], [labels], False)[0], 0
                )
            return logs.exp().tolist(), logs.tolist()

        def adapter_state(self):
            return self.model.state_dict()

        def load_adapter_state(self, state):
            self.model.load_state_dict(state)

        def optimizer_state(self):
            return self.optimizer.state_dict()

        def load_optimizer_state(self, state):
            self.optimizer.load_state_dict(state)

    monkeypatch.setattr(exp, "PlannerPolicy", ToyPolicy)
    monkeypatch.setattr(
        exp, "RAGWorkflowState", lambda **kwargs: FullState(kwargs["item"].qid)
    )
    monkeypatch.setattr(exp, "JSONTemplateGenerator", lambda policy: None)
    monkeypatch.setattr(
        exp,
        "RAGSpace",
        lambda *args, **kwargs: SimpleNamespace(
            offload_models=lambda: None, restore_models=lambda: None
        ),
    )
    items = [
        SimpleNamespace(qid=f"q{i}", answer="answer", context_titles=())
        for i in range(4)
    ]
    cfg = {
        "spec": CellSpec("at2po", 11),
        "model_path": tmp_path,
        "rag_model_root": tmp_path,
        "rag_index_root": tmp_path,
        "output_dir": tmp_path / "paused",
        "train_items": items[:2],
        "valid_items": items[2:3],
        "test_items": items[3:],
        "device": "cpu",
        "resume": False,
        "max_new_train_items": 3,
        "search_budget": 12,
        "token_beta": 1e-4,
        "search_cost_mode": "agentuct-Bellman",
        "workflow_cost_lambda": 0.1,
        "execution_cost_reference": 4096,
        "iterations": 2,
        "epochs": 1,
        "ppo_batch_size": 8,
        "checkpoint_interval_qids": 1,
        "at2po_config": AT2POConfig(leaf_budget=8),
    }
    paused = exp.run_cell(**cfg)
    assert paused["status"] == "paused" and paused["iteration"] == 1
    checkpoint_path = cfg["output_dir"] / "at2po-seed-11" / "checkpoint.pt"
    before = torch.load(checkpoint_path, weights_only=False)
    assert before["optimizer"]["state"]
    with pytest.raises(ValueError, match="same algorithm settings"):
        exp.run_cell(
            **(cfg | {"resume": True, "at2po_config": AT2POConfig(leaf_budget=10)})
        )
    resumed = exp.run_cell(**(cfg | {"resume": True, "max_new_train_items": None}))
    after = torch.load(checkpoint_path, weights_only=False)
    full = exp.run_cell(
        **(cfg | {"output_dir": tmp_path / "full", "max_new_train_items": None})
    )
    uninterrupted = torch.load(
        tmp_path / "full/at2po-seed-11/checkpoint.pt", weights_only=False
    )
    for name in after["adapter"]:
        torch.testing.assert_close(
            after["adapter"][name], uninterrupted["adapter"][name], rtol=0, atol=0
        )
    assert resumed["status"] == full["status"] == "complete"
    assert (
        sum(i["search_tokens"]["logical_evaluations"] for i in resumed["iterations"])
        == 32
    )
    assert resumed["test"] == full["test"]
    rows = [
        json.loads(line)
        for line in (checkpoint_path.parent / "trajectories.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(rows) == 4 and all(
        "at2po_tree" in r and "actor_path" not in r for r in rows
    )
    assert all(math.isfinite(r["advantage"]) for row in rows for r in row["steps"])
    assert all("at2po_tree" not in row for row in resumed["test"]["rows"])
