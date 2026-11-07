"""Shared budget and per-planning-call transition cache."""
from dataclasses import dataclass
from ..model import Belief, PlanningProblem, update
from ..policy import Policy


class BudgetExhausted(Exception):
    pass


@dataclass(slots=True)
class PlanningStats:
    evaluations: int = 0
    cache_hits: int = 0
    expansions: int = 0
    action_scans: int = 0


@dataclass(slots=True)
class Budget:
    limit: int | None
    stats: PlanningStats

    def can_consume(self, amount: int) -> bool:
        return self.limit is None or self.stats.evaluations + amount <= self.limit

    def consume(self) -> None:
        if not self.can_consume(1):
            raise BudgetExhausted
        self.stats.evaluations += 1


@dataclass(frozen=True, slots=True)
class PlanResult:
    policy: Policy
    model_value: float
    lower: float | None
    upper: float | None
    evaluations: int
    cache_hits: int
    expansions: int
    action_scans: int
    status: str = 'complete'


class TransitionCache:
    def __init__(self, problem: PlanningProblem, limit: int | None = None):
        if limit is not None and limit < 0:
            raise ValueError('budget must be nonnegative')
        self.problem = problem
        self.stats = PlanningStats()
        self.budget = Budget(limit, self.stats)
        self.cache: dict[tuple, Belief] = {}
        self.on_before_evaluation = None

    def key(self, state: tuple[int, ...], path: int, outcome: int):
        return state, path, outcome

    def needed(self, state, path, outcomes):
        return sum(self.key(state, path, o) not in self.cache for o in outcomes)

    def child(self, state: tuple[int, ...], belief: Belief, path: int, outcome: int):
        key = self.key(state, path, outcome)
        if key in self.cache:
            self.stats.cache_hits += 1
            return self.cache[key]
        if self.on_before_evaluation is not None:
            self.on_before_evaluation(self.stats.evaluations)
        if not self.budget.can_consume(1):
            raise BudgetExhausted
        result = update(belief, path, outcome, self.problem.spec.current)
        if result is None:
            raise ValueError('cannot calculate an impossible outcome')
        self.budget.consume()
        self.cache[key] = result
        return result

    def result(self, policy, value, lower=None, upper=None, status='complete'):
        s = self.stats
        return PlanResult(policy, value, lower, upper, s.evaluations, s.cache_hits,
                          s.expansions, s.action_scans, status)


def successor_state(state, path, outcome):
    counts = list(state)
    counts[2 * path + outcome] += 1
    return tuple(counts)
