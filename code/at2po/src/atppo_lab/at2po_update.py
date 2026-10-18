"""Globally weighted AT²PO updates with bounded node batches and gradient SUM.

Only rank zero owns the optimizer and full training record list. Each global
leaf minibatch is folded once, then its nodes are distributed without padding
or duplication. A small stream of batches bounds non-learner host memory.
"""

from __future__ import annotations

import math
from itertools import islice
from time import monotonic
from typing import Any

from .at2po import AT2POConfig, AT2POUpdateConfig, categorical_turn_loss

_NODE_FIELDS = ("prompt", "labels", "chosen_label", "old_logprob", "advantage")
_TRANSFER_BATCHES = 32


def _sequences(records):
    sequences = {}
    for record in records:
        if not math.isfinite(float(record["advantage"])):
            raise ValueError("non-finite AT2PO advantage")
        sequences.setdefault((record["qid"], record["at2po_sequence"]), []).append(
            record
        )
    for sequence in sequences.values():
        if any(r["at2po_sequence_length"] != len(sequence) for r in sequence):
            raise ValueError("AT2PO requires complete leaf trajectories in each update")
    return list(sequences.values())


def _weighted_nodes(active):
    weighted = {}
    for sequence in active:
        weight = 1.0 / (len(active) * len(sequence))
        for record in sequence:
            key = (record["qid"], record["at2po_node"])
            previous = weighted.get(key)
            if previous and any(previous[0][k] != record[k] for k in _NODE_FIELDS):
                raise ValueError("inconsistent shared AT2PO node")
            weighted[key] = (record, weight + (previous[1] if previous else 0))
    # Remove collection traces from the transferred payload. Keep original
    # leaf denominators, including zero-advantage and repeated occurrences.
    return [
        ({k: r[k] for k in _NODE_FIELDS}, weight)
        for r, weight in weighted.values()
        if abs(r["advantage"]) > 1e-12
    ]


def _batches(sequences, epochs, batch_size):
    for epoch in range(epochs):
        for start in range(0, len(sequences), batch_size):
            yield epoch, _weighted_nodes(sequences[start : start + batch_size])


def _broadcast(value, dist, device, rank):
    if dist is None:
        return value
    objects = [value if rank == 0 else None]
    dist.broadcast_object_list(objects, src=0, device=device)
    return objects[0]


def _backward(policy, nodes, config, microbatch_size):
    torch = policy.torch
    loss_sum, kl_sum, outside, count, forwards = 0.0, 0.0, 0, 0, 0
    for start in range(0, len(nodes), microbatch_size):
        chunk = nodes[start : start + microbatch_size]
        scores = policy.batch_action_logprobs(
            [r["prompt"] for r, _ in chunk],
            [r["labels"] for r, _ in chunk],
            grad=True,
        )
        weighted_losses, logprobs = [], []
        for (record, weight), action_scores in zip(chunk, scores, strict=True):
            logprob = torch.log_softmax(action_scores, dim=0)[
                record["labels"].index(record["chosen_label"])
            ]
            weighted_losses.append(
                weight
                * categorical_turn_loss(
                    torch,
                    logprob,
                    record["old_logprob"],
                    record["advantage"],
                    config,
                )
            )
            logprobs.append(logprob)
        loss = torch.stack(weighted_losses).sum()
        loss.backward()
        # One host copy per node microbatch, not two synchronizations per node.
        values = (
            torch.stack([loss.detach(), *(p.detach() for p in logprobs)]).cpu().tolist()
        )
        loss_sum += values[0]
        for (record, _), logprob in zip(chunk, values[1:], strict=True):
            delta = logprob - record["old_logprob"]
            kl_sum -= delta
            ratio = math.exp(min(delta, 10.0))
            outside += int(ratio < 1 - config.clip_low or ratio > 1 + config.clip_high)
        count += len(chunk)
        forwards += 1
        # Release the previous microbatch's logits/graph before the next forward.
        del scores, weighted_losses, logprobs, loss, logprob, action_scores
    return loss_sum, kl_sum, outside, count, forwards


class _GradientReducer:
    """Coalesce trainable tensors by dtype; communicate no frozen base weights."""

    def __init__(self, torch, parameters, dist, rank):
        self.torch, self.parameters, self.dist, self.rank = (
            torch,
            parameters,
            dist,
            rank,
        )
        groups = {}
        for index, parameter in enumerate(parameters):
            groups.setdefault((parameter.device, parameter.dtype), []).append(
                (index, parameter)
            )
        self.buckets = list(groups.values())

    def sum_to_learner(self):
        torch, dist = self.torch, self.dist
        # None and zero gradients differ for AdamW weight decay. Preserve None
        # for a parameter unused on every rank, including empty rank shards.
        present = torch.tensor(
            [p.grad is not None for p in self.parameters],
            device=self.parameters[0].device,
            dtype=torch.int32,
        )
        dist.reduce(present, dst=0, op=dist.ReduceOp.SUM)
        used = present.cpu().tolist() if self.rank == 0 else None
        for bucket in self.buckets:
            flat = torch.cat(
                [
                    (
                        p.grad.detach() if p.grad is not None else torch.zeros_like(p)
                    ).reshape(-1)
                    for _, p in bucket
                ]
            )
            # Weights already contain the GLOBAL leaf/action denominator.
            # No world-size average and no local renormalization here.
            dist.reduce(flat, dst=0, op=dist.ReduceOp.SUM)
            if self.rank == 0:
                offset = 0
                for index, parameter in bucket:
                    size = parameter.numel()
                    parameter.grad = (
                        flat[offset : offset + size].view_as(parameter)
                        if used[index]
                        else None
                    )
                    offset += size

    def broadcast_parameters(self):
        with self.torch.no_grad():
            for bucket in self.buckets:
                flat = self.torch.cat([p.detach().reshape(-1) for _, p in bucket])
                self.dist.broadcast(flat, src=0)
                if self.rank != 0:
                    offset = 0
                    for _, parameter in bucket:
                        size = parameter.numel()
                        parameter.copy_(flat[offset : offset + size].view_as(parameter))
                        offset += size


def update(
    policy: Any,
    records: list[dict[str, Any]],
    *,
    epochs: int,
    batch_size: int,
    config: AT2POConfig,
    execution: AT2POUpdateConfig,
    dist: Any = None,
    progress: Any = None,
) -> dict[str, float | int]:
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("AT2PO epochs and leaf batch size must be positive")
    if dist is not None and execution.backend != "parallel":
        raise ValueError("distributed AT2PO update requires the parallel backend")
    torch = policy.torch
    rank, world_size = (
        (dist.get_rank(), dist.get_world_size()) if dist is not None else (0, 1)
    )
    parameters = [p for p in policy.model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("AT2PO update requires trainable parameters")
    device = parameters[0].device
    if rank == 0 and policy.optimizer is None:
        raise RuntimeError("AT2PO requires a learner optimizer")
    policy.model.train()
    scoring_start, started = policy.actor_scoring_tokens, monotonic()
    header, batches = None, None
    if rank == 0:
        sequences = _sequences(records)
        advantages = [r["advantage"] for r in records]
        mean = math.fsum(advantages) / len(advantages) if advantages else 0.0
        header = {
            "samples": len(records),
            "leaf_sequences": len(sequences),
            "advantage_mean": mean,
            "advantage_std": math.sqrt(
                math.fsum((a - mean) ** 2 for a in advantages) / len(advantages)
            )
            if advantages
            else 0.0,
            "global_batches": epochs * math.ceil(len(sequences) / batch_size),
        }
        batches = iter(_batches(sequences, epochs, batch_size))
    header = _broadcast(header, dist, device, rank)
    reducer = (
        _GradientReducer(torch, parameters, dist, rank) if dist is not None else None
    )
    totals = [0.0] * 5  # weighted loss, KL, clip count, scored nodes, forwards
    norms, steps, zero_batches, processed = [], 0, 0, 0
    for _ in range(0, header["global_batches"], _TRANSFER_BATCHES):
        block = list(islice(batches, _TRANSFER_BATCHES)) if rank == 0 else None
        block = _broadcast(block, dist, device, rank)
        for epoch, nodes in block:
            processed += 1
            # This skip decision is global, including ranks with no local nodes.
            policy.model.zero_grad(set_to_none=True)
            if not nodes:
                zero_batches += 1
                continue
            stats = _backward(
                policy, nodes[rank::world_size], config, execution.node_microbatch_size
            )
            totals = [a + b for a, b in zip(totals, stats, strict=True)]
            if reducer is not None:
                reducer.sum_to_learner()
            if rank == 0:
                norm = torch.nn.utils.clip_grad_norm_(
                    parameters, 1.0, error_if_nonfinite=True
                )
                norms.append(float(norm))
                policy.optimizer.step()
            if reducer is not None:
                reducer.broadcast_parameters()
            steps += 1
        if rank == 0 and progress is not None:
            progress(
                {
                    "epoch": epoch,
                    "global_batches_completed": processed,
                    "global_batches_total": header["global_batches"],
                    "optimizer_steps": steps,
                    "update_world_size": world_size,
                    "node_microbatch_size": execution.node_microbatch_size,
                    "elapsed_seconds": monotonic() - started,
                }
            )
    totals.append(policy.actor_scoring_tokens - scoring_start)
    if dist is not None:
        total_tensor = torch.tensor(totals, dtype=torch.float64, device=device)
        dist.reduce(total_tensor, dst=0, op=dist.ReduceOp.SUM)
        if rank == 0:
            totals = total_tensor.cpu().tolist()
    loss_sum, kl_sum, clipped, scored, forwards, tokens = totals
    telemetry = {
        **header,
        "mean_loss": loss_sum / steps if steps else 0.0,
        "mean_gradient_norm": math.fsum(norms) / len(norms) if norms else 0.0,
        "max_gradient_norm": max(norms, default=0.0),
        "optimizer_steps": steps,
        "zero_advantage_batches_skipped": zero_batches,
        "scored_node_approx_kl": kl_sum / scored if scored else 0.0,
        "scored_node_ratio_outside_clip_fraction": clipped / scored if scored else 0.0,
        "actor_scoring_tokens": int(tokens),
        "scored_nodes": int(scored),
        "scoring_forward_calls": int(forwards),
        "update_world_size": world_size,
        "node_microbatch_size": execution.node_microbatch_size,
        "global_leaf_batch_size": batch_size,
        "elapsed_seconds": monotonic() - started,
    }
    return _broadcast(telemetry, dist, device, rank)
