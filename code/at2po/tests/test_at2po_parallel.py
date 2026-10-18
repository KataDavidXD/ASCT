"""Loss/gradient/AdamW parity, including real multi-process Gloo collectives."""

from copy import copy, deepcopy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from atppo_lab.at2po import (  # noqa: E402
    AT2POConfig,
    AT2POUpdateConfig,
    categorical_turn_loss,
    update_at2po,
)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor([[0.1, -0.1], [-0.2, 0.3], [0.4, 0.2]], dtype=torch.float64)
        )
        self.bias = torch.nn.Parameter(
            torch.tensor([0.01, -0.02, 0.03], dtype=torch.float32)
        )
        self.unused = torch.nn.Parameter(torch.tensor([0.7], dtype=torch.float64))
        self.frozen = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)

    def forward(self, inputs):
        return inputs @ self.weight.T + self.bias.to(torch.float64)


class Policy:
    def __init__(self, learner=True):
        self.torch, self.model = torch, Model()
        self.actor_scoring_tokens = 0
        self.calls = []
        self.optimizer = (
            torch.optim.AdamW(self.model.parameters(), lr=0.007) if learner else None
        )

    def batch_action_logprobs(self, prompts, labels, grad):
        self.calls.append(len(prompts))
        self.actor_scoring_tokens += sum(
            len(p) * len(a) for p, a in zip(prompts, labels, strict=True)
        )
        inputs = torch.tensor(
            [[1.0, int(p) / 10] for p in prompts], dtype=torch.float64
        )
        logits = self.model(inputs)
        return [
            row[: len(actions)] for row, actions in zip(logits, labels, strict=True)
        ]


def records():
    policy = Policy()
    result = []
    # Three global batches of eight leaves: regular, all-zero, and a short tail.
    # The last batch has one unique nonzero node, leaving seven ranks idle.
    for sequence in range(19):
        zero, tail = 8 <= sequence < 16, sequence >= 16
        nodes = (
            [1] if tail else [1, *(10 * sequence + j + 10 for j in range(sequence % 4))]
        )
        for node in nodes:
            labels = ["a", "b", "c"][: 1 + node % 3]
            chosen = labels[node % len(labels)]
            prompt = str(node)
            old_logprob = torch.log_softmax(
                policy.batch_action_logprobs([prompt], [labels], False)[0], 0
            )[labels.index(chosen)].item()
            result.append(
                {
                    "qid": "zero" if zero else "tail" if tail else "q",
                    "at2po_sequence": sequence,
                    "at2po_sequence_length": len(nodes),
                    "at2po_node": node,
                    "prompt": prompt,
                    "labels": labels,
                    "chosen_label": chosen,
                    "old_logprob": old_logprob,
                    "advantage": 0.0
                    if zero
                    else (-1 if node % 2 else 1) * (2 + node / 10),
                }
            )
    return result


def reference_update(policy, data, epochs=2):
    """Literal leaf/action mean, one occurrence at a time, with no folding."""
    sequences = {}
    for record in data:
        sequences.setdefault((record["qid"], record["at2po_sequence"]), []).append(
            record
        )
    leaves = list(sequences.values())
    for _ in range(epochs):
        for start in range(0, len(leaves), 8):
            active = leaves[start : start + 8]
            policy.optimizer.zero_grad(set_to_none=True)
            used = False
            for leaf in active:
                for r in leaf:
                    if abs(r["advantage"]) <= 1e-12:
                        continue
                    used = True
                    scores = policy.batch_action_logprobs(
                        [r["prompt"]], [r["labels"]], True
                    )[0]
                    logprob = torch.log_softmax(scores, 0)[
                        r["labels"].index(r["chosen_label"])
                    ]
                    (
                        categorical_turn_loss(
                            torch,
                            logprob,
                            r["old_logprob"],
                            r["advantage"],
                            AT2POConfig(),
                        )
                        / (len(active) * len(leaf))
                    ).backward()
            if used:
                torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)
                policy.optimizer.step()


def assert_state_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=3e-6, atol=2e-8)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_state_equal(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize("microbatch", [1, 2, 4, 100])
def test_microbatch_matches_literal_leaf_loss_and_optimizer(microbatch):
    policy, reference = Policy(), Policy()
    data = records()
    reference_update(reference, data)
    result = update_at2po(
        policy,
        data,
        epochs=2,
        batch_size=8,
        config=AT2POConfig(),
        execution=AT2POUpdateConfig(microbatch),
    )
    assert_state_equal(policy.model.state_dict(), reference.model.state_dict())
    assert_state_equal(policy.optimizer.state_dict(), reference.optimizer.state_dict())
    assert result["optimizer_steps"] == 4
    assert result["zero_advantage_batches_skipped"] == 2
    assert result["global_leaf_batch_size"] == 8
    assert max(policy.calls) <= microbatch
    assert policy.model.unused.grad is None
    assert policy.model.unused.item() == 0.7


def _worker(rank, world_size, rendezvous, output, microbatch):
    import torch.distributed as dist
    from atppo_lab import at2po_update

    torch.set_num_threads(1)
    at2po_update._TRANSFER_BATCHES = 2
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=90),
    )
    try:
        policy = Policy(learner=rank == 0)
        data = records() if rank == 0 else []
        telemetry = update_at2po(
            policy,
            data,
            epochs=2,
            batch_size=8,
            config=AT2POConfig(),
            execution=AT2POUpdateConfig(microbatch),
            dist=dist,
        )
        first_model = deepcopy(policy.model.state_dict())
        first_optimizer = deepcopy(policy.optimizer.state_dict()) if rank == 0 else None
        # Resume from a full committed update, switching execution microbatch.
        fresh = Policy(learner=rank == 0)
        fresh.model.load_state_dict(first_model)
        if rank == 0:
            fresh.optimizer.load_state_dict(deepcopy(first_optimizer))
        update_at2po(
            fresh,
            data,
            epochs=1,
            batch_size=8,
            config=AT2POConfig(),
            execution=AT2POUpdateConfig(1),
            dist=dist,
        )
        torch.save(
            {
                "model": first_model,
                "optimizer": first_optimizer,
                "telemetry": telemetry,
                "resumed_model": fresh.model.state_dict(),
                "local_tokens": policy.actor_scoring_tokens,
            },
            Path(output) / f"rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size,microbatch", [(2, 1), (2, 3), (8, 2)])
def test_distributed_sum_parameters_tokens_and_resume(
    tmp_path, world_size, microbatch, monkeypatch
):
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_gloo_available()
    ):
        pytest.skip("Gloo unavailable")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    rendezvous = (tmp_path / "rendezvous").as_uri()
    torch.multiprocessing.spawn(
        _worker,
        args=(world_size, rendezvous, str(tmp_path), microbatch),
        nprocs=world_size,
        join=True,
    )
    reference = Policy()
    telemetry = update_at2po(
        reference,
        records(),
        epochs=2,
        batch_size=8,
        config=AT2POConfig(),
        execution=AT2POUpdateConfig(1),
    )
    expected = deepcopy(reference.model.state_dict())
    optimizer = deepcopy(reference.optimizer.state_dict())
    update_at2po(
        reference,
        records(),
        epochs=1,
        batch_size=8,
        config=AT2POConfig(),
        execution=AT2POUpdateConfig(1),
    )
    results = [
        torch.load(tmp_path / f"rank-{r}.pt", weights_only=True)
        for r in range(world_size)
    ]
    assert sum(r["local_tokens"] for r in results) == telemetry["actor_scoring_tokens"]
    for result in results:
        assert_state_equal(result["model"], expected)
        assert_state_equal(result["resumed_model"], reference.model.state_dict())
        for key in (
            "mean_loss",
            "mean_gradient_norm",
            "max_gradient_norm",
            "scored_node_approx_kl",
            "scored_node_ratio_outside_clip_fraction",
        ):
            assert result["telemetry"][key] == pytest.approx(
                telemetry[key], rel=3e-6, abs=2e-8
            )
        for key in (
            "optimizer_steps",
            "samples",
            "leaf_sequences",
            "actor_scoring_tokens",
            "zero_advantage_batches_skipped",
            "scored_nodes",
        ):
            assert result["telemetry"][key] == telemetry[key]
        assert result["telemetry"]["update_world_size"] == world_size
    assert_state_equal(results[0]["optimizer"], optimizer)
    # Replica weights are synchronized exactly after each rank-zero update.
    for result in results[1:]:
        for key in expected:
            assert torch.equal(results[0]["model"][key], result["model"][key])


def test_execution_options_do_not_change_algorithm_resume_contract():
    from atppo_lab.distributed_cell import _algorithm_settings
    from atppo_lab.main import _parser

    args = _parser().parse_args(
        [
            "--model-path",
            "m",
            "--dataset-root",
            "d",
            "--output-dir",
            "o",
            "--devices",
            "cuda:0",
        ]
    )
    assert args.at2po_update_backend == "parallel"
    assert args.at2po_node_microbatch_size == 2
    cfg = SimpleNamespace(
        search_budget=12,
        token_beta=1e-4,
        search_cost_mode="qwen",
        workflow_cost_lambda=0.1,
        execution_cost_reference=4096,
        iterations=3,
        epochs=2,
        ppo_batch_size=8,
        world_size=8,
        spec=SimpleNamespace(method="at2po"),
        at2po_config=AT2POConfig(leaf_budget=106),
    )
    original = _algorithm_settings(cfg)
    cfg.at2po_update_config = AT2POUpdateConfig(4, "parallel")
    assert _algorithm_settings(cfg) == original
    cfg.at2po_update_config = AT2POUpdateConfig(1, "rank0")
    assert _algorithm_settings(cfg) == original


def test_empty_zero_incomplete_and_inconsistent_records():
    policy = Policy()
    result = update_at2po(policy, [], epochs=2, batch_size=8, config=AT2POConfig())
    assert result["optimizer_steps"] == result["actor_scoring_tokens"] == 0
    data = records()
    for r in data:
        r["advantage"] = 0.0
    result = update_at2po(policy, data, epochs=2, batch_size=8, config=AT2POConfig())
    assert result["optimizer_steps"] == result["actor_scoring_tokens"] == 0
    assert result["zero_advantage_batches_skipped"] == 6
    data[1]["advantage"] = 1.0
    with pytest.raises(ValueError, match="inconsistent shared"):
        update_at2po(policy, data, epochs=1, batch_size=8, config=AT2POConfig())
    with pytest.raises(ValueError, match="complete leaf"):
        update_at2po(
            policy, records()[2:], epochs=1, batch_size=8, config=AT2POConfig()
        )
    for invalid in (0, -1, 1.2):
        with pytest.raises(ValueError):
            AT2POUpdateConfig(invalid)


def test_planner_qwen_lora_padded_scoring_and_update():
    """Real PlannerPolicy scorer + tiny random Qwen3/LoRA, no model download."""
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    tokenizers = pytest.importorskip("tokenizers")
    from atppo_lab.policy import PlannerPolicy

    torch.set_num_threads(1)
    torch.manual_seed(11)
    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    vocab = {token: i for i, token in enumerate(["[PAD]", "[UNK]", *alphabet])}
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.BPE(vocab, [], unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )
    policy = PlannerPolicy.__new__(PlannerPolicy)
    policy.torch, policy.device = torch, torch.device("cpu")
    policy.tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        padding_side="right",
    )
    policy.tokenizer.chat_template = "{% for m in messages %}{{ m['role'] }}:{{ m['content'] }}\n{% endfor %}assistant:"
    cfg = transformers.Qwen3Config(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        attention_dropout=0.0,
    )
    cfg._attn_implementation = "sdpa"
    policy.model = peft.get_peft_model(
        transformers.Qwen3ForCausalLM(cfg),
        peft.LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    with torch.no_grad():
        for name, p in policy.model.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.02)
    policy.model.eval()
    policy.actor_scoring_tokens = 0
    prompts = ["q", "A longer question with different padding", "middle"]
    labels = [
        ["query.direct", "query.rewrite"],
        ["retrieve.e5"],
        ["stop", "continue", "retry"],
    ]
    single = [
        policy.batch_action_logprobs([p], [a], grad=False)[0]
        for p, a in zip(prompts, labels, strict=True)
    ]
    token_count = policy.actor_scoring_tokens
    policy.actor_scoring_tokens = 0
    batched = policy.batch_action_logprobs(prompts, labels, grad=False)
    assert policy.actor_scoring_tokens == token_count
    for a, b in zip(single, batched, strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)

    data = []
    for i, (p, a, scores) in enumerate(zip(prompts, labels, single, strict=True)):
        data.append(
            {
                "qid": f"q{i}",
                "at2po_node": 1,
                "at2po_sequence": i,
                "at2po_sequence_length": 1,
                "prompt": p,
                "labels": a,
                "chosen_label": a[0],
                "old_logprob": torch.log_softmax(scores, 0)[0].item(),
                "advantage": (-1) ** i * 0.5,
            }
        )
    baseline = copy(policy)
    baseline.model = deepcopy(policy.model)
    frozen = {
        n: p.detach().clone()
        for n, p in policy.model.named_parameters()
        if not p.requires_grad
    }
    results = []
    for current, size in ((baseline, 1), (policy, 2)):
        current.optimizer = torch.optim.AdamW(current.trainable_parameters(), lr=1e-6)
        results.append(
            update_at2po(
                current,
                data,
                epochs=2,
                batch_size=2,
                config=AT2POConfig(),
                execution=AT2POUpdateConfig(size),
            )
        )
    assert_state_equal(policy.model.state_dict(), baseline.model.state_dict())
    assert results[0]["actor_scoring_tokens"] == results[1]["actor_scoring_tokens"]
    assert results[0]["mean_loss"] == pytest.approx(results[1]["mean_loss"], abs=2e-6)
    for name, value in frozen.items():
        assert torch.equal(value, dict(policy.model.named_parameters())[name])
