"""Complete immutable policies independent of planner implementations."""
from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class Stop:
    path: int


@dataclass(frozen=True, slots=True)
class Test:
    path: int
    failure: 'Policy'
    success: 'Policy'


Policy: TypeAlias = Stop | Test


def validate_policy(policy: Policy, path_count: int, horizon: int) -> None:
    """Reject policies that cannot execute within the task's test budget."""
    def check(node, remaining):
        if isinstance(node, Stop):
            if node.path != -1 and not 0 <= node.path < path_count:
                raise ValueError('invalid terminal path')
            return
        if not isinstance(node, Test) or remaining <= 0 or not 0 <= node.path < path_count:
            raise ValueError('invalid test action or exhausted horizon')
        check(node.failure, remaining - 1)
        check(node.success, remaining - 1)
    check(policy, horizon)
