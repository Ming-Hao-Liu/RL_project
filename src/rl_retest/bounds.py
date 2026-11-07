"""Model-based value bounds for independent path types."""
from .model import Belief, ModelSpec, best_stop


def perfect_information(belief: Belief, spec: ModelSpec) -> float:
    levels = [spec.fallback] + sorted({p for p in spec.current if p > spec.fallback})
    value = spec.fallback
    for low, high in zip(levels, levels[1:]):
        product = 1.0
        for path in belief:
            product *= sum(b for b, p in zip(path, spec.current) if p <= low)
        value += (high - low) * (1.0 - product)
    return spec.reward * value


def initial_upper(belief: Belief, spec: ModelSpec, remaining: int) -> float:
    stop, _ = best_stop(belief, spec)
    if remaining == 0:
        return stop
    return max(stop, perfect_information(belief, spec) - spec.cost)
