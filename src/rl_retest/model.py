"""Model types and pure probability operations; no planning or I/O."""
from dataclasses import dataclass
from typing import TypeAlias

Probabilities: TypeAlias = tuple[float, float, float]
Belief: TypeAlias = tuple[Probabilities, ...]
FALLBACK = -1


@dataclass(frozen=True, slots=True)
class ModelSpec:
    reward: float
    cost: float
    horizon: int
    fallback: float
    current: Probabilities
    historical: tuple[Probabilities, ...]


@dataclass(frozen=True, slots=True)
class History:
    # Outer axis is capability, inner axis is path.
    successes: tuple[tuple[int, ...], ...]
    failures: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class PlanningProblem:
    belief: Belief
    spec: ModelSpec


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    belief: Belief
    spec: ModelSpec


def posterior(priors: Belief, history: History, tables: tuple[Probabilities, ...]) -> Belief | None:
    """Update independent path priors; return None for impossible history."""
    if len(history.successes) != len(tables) or len(history.failures) != len(tables):
        raise ValueError('history and capability tables must have the same length')
    result = []
    for i, prior in enumerate(priors):
        weights = list(prior)
        for successes, failures, probs in zip(history.successes, history.failures, tables):
            for z in range(3):
                weights[z] *= probs[z] ** successes[i] * (1.0 - probs[z]) ** failures[i]
        total = sum(weights)
        if total == 0:
            return None
        result.append(tuple(w / total for w in weights))
    return tuple(result)


def predictive(belief: Belief, path: int, probabilities: Probabilities) -> float:
    return sum(b * p for b, p in zip(belief[path], probabilities))


def outcome_probability(belief: Belief, path: int, outcome: int, probabilities: Probabilities) -> float:
    return sum(b * (p if outcome else 1.0 - p) for b, p in zip(belief[path], probabilities))


def update(belief: Belief, path: int, outcome: int, probabilities: Probabilities) -> Belief | None:
    weights = tuple(b * (p if outcome else 1.0 - p) for b, p in zip(belief[path], probabilities))
    total = sum(weights)
    if total <= 0:
        return None
    paths = list(belief)
    paths[path] = tuple(w / total for w in weights)
    return tuple(paths)


def best_stop(belief: Belief, spec: ModelSpec) -> tuple[float, int]:
    best, path = spec.fallback, FALLBACK
    for i in range(len(belief)):
        p = predictive(belief, i, spec.current)
        if p > best:
            best, path = p, i
    return spec.reward * best, path
