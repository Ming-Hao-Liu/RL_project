"""Blinkered planning, reselecting the path after each observation."""
from ..model import PlanningProblem, best_stop, outcome_probability
from ..policy import Stop, Test
from .common import TransitionCache, successor_state


def solve_blinkered(problem: PlanningProblem):
    cache = TransitionCache(problem)
    spec = problem.spec
    memo = {}

    def restricted(state, belief, remaining, target):
        key = (state, remaining, target)
        if key in memo:
            return memo[key]
        best, _ = best_stop(belief, spec)
        if remaining:
            value = -spec.cost
            for o in (0, 1):
                prob = outcome_probability(belief, target, o, spec.current)
                if prob > 0:
                    b2 = cache.child(state, belief, target, o)
                    value += prob * restricted(successor_state(state, target, o), b2, remaining - 1, target)
            best = max(best, value)
        memo[key] = best
        return best

    def build(state, belief, remaining):
        best, stop_path = best_stop(belief, spec)
        selected = None
        if remaining:
            cache.stats.expansions += 1
            for i in range(len(belief)):
                cache.stats.action_scans += 1
                value = -spec.cost
                for o in (0, 1):
                    prob = outcome_probability(belief, i, o, spec.current)
                    if prob > 0:
                        b2 = cache.child(state, belief, i, o)
                        value += prob * restricted(successor_state(state, i, o), b2, remaining - 1, i)
                if value > best:
                    best, selected = value, i
        if selected is None:
            return best_stop(belief, spec)[0], Stop(stop_path)
        value, children = -spec.cost, []
        for o in (0, 1):
            prob = outcome_probability(belief, selected, o, spec.current)
            if prob > 0:
                b2 = cache.child(state, belief, selected, o)
                v, child = build(successor_state(state, selected, o), b2, remaining - 1)
                value += prob * v
            else:
                child = Stop(stop_path)
            children.append(child)
        return value, Test(selected, *children)

    value, policy = build((0,) * (2 * len(problem.belief)), problem.belief, spec.horizon)
    return cache.result(policy, value, value)
