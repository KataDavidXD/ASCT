"""Small tree-search data types used by Uniform, UCT, and AgentUCT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Protocol


class SearchState(Protocol):
    """State contract required by the three search methods."""

    qid: str
    path: tuple[Hashable, ...]

    def is_terminal(self) -> bool: ...

    def available_actions(self) -> tuple[Hashable, ...]: ...

    def child(self, action: Hashable) -> "SearchState": ...

    def preview_child(self, action: Hashable) -> "SearchState": ...

    def estimated_search_tokens(self, action: Hashable) -> float: ...

    def outcome(self) -> object: ...

    def path_costs(self) -> tuple[float, ...]: ...


@dataclass
class TreeNode:
    state: SearchState
    parent: "TreeNode | None" = None
    action_from_parent: Hashable | None = None
    children: dict[Hashable, "TreeNode"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0
    performance_sum: float = 0.0
    execution_cost_sum: float = 0.0

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    @property
    def q_performance(self) -> float:
        return self.performance_sum / self.visits if self.visits else 0.0

    @property
    def q_execution_cost(self) -> float:
        return self.execution_cost_sum / self.visits if self.visits else 0.0

    def unexpanded_actions(self) -> tuple[Hashable, ...]:
        return tuple(
            action
            for action in self.state.available_actions()
            if action not in self.children
        )

    def add_child(self, action: Hashable, state: SearchState) -> "TreeNode":
        child = TreeNode(
            state=state,
            parent=self,
            action_from_parent=action,
        )
        self.children[action] = child
        return child

    def update(
        self, *, utility: float, performance: float, execution_cost: float
    ) -> None:
        self.visits += 1
        self.value_sum += utility
        self.performance_sum += performance
        self.execution_cost_sum += execution_cost


__all__ = ["SearchState", "TreeNode"]
