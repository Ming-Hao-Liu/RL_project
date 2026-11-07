"""Evaluate a frozen policy under a separately supplied true model."""
from dataclasses import dataclass
from .model import FALLBACK, EvaluationContext, outcome_probability, predictive, update
from .policy import Policy, Stop, validate_policy


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    expected_utility: float
    expected_tests: float
    success_probability: float


def evaluate(policy: Policy, context: EvaluationContext) -> EvaluationResult:
    spec = context.spec
    validate_policy(policy, len(context.belief), spec.horizon)

    def visit(node, belief):
        if isinstance(node, Stop):
            p = spec.fallback if node.path == FALLBACK else predictive(belief, node.path, spec.current)
            return p, 0.0
        successes, tests = 0.0, 1.0
        for outcome, child in ((0, node.failure), (1, node.success)):
            probability = outcome_probability(belief, node.path, outcome, spec.current)
            if probability > 0:
                next_belief = update(belief, node.path, outcome, spec.current)
                success, count = visit(child, next_belief)
                successes += probability * success
                tests += probability * count
        return successes, tests

    success, tests = visit(policy, context.belief)
    return EvaluationResult(spec.reward * success - spec.cost * tests, tests, success)


def evaluate_resampling(policy: Policy, context: EvaluationContext, priors) -> EvaluationResult:
    """Exactly one uniformly selected path gets an independent prior draw."""
    results = []
    for i, prior in enumerate(priors):
        paths = list(context.belief)
        paths[i] = prior
        results.append(evaluate(policy, EvaluationContext(tuple(paths), context.spec)))
    n = len(results)
    return EvaluationResult(*(sum(getattr(r, field) for r in results) / n for field in
                              ('expected_utility', 'expected_tests', 'success_probability')))
