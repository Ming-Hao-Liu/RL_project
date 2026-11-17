"""Finite-horizon Bellman planning and deterministic baseline policies."""
from time import perf_counter
from ..model import PlanningProblem, best_stop, outcome_probability
from ..policy import Stop, Test
from .common import BudgetExhausted, TransitionCache, successor_state


class _Bellman:
    def __init__(self, problem, cache):
        self.problem = problem
        self.cache = cache
        self.memo = {}

    def solve(self, state, belief, depth):
        key = (state, depth)
        if key in self.memo:
            return self.memo[key]
        spec = self.problem.spec
        best, path = best_stop(belief, spec)
        policy = Stop(path)
        if depth:
            self.cache.stats.expansions += 1
            for i in range(len(belief)):
                self.cache.stats.action_scans += 1
                value = -spec.cost
                children = []
                for o in (0, 1):
                    prob = outcome_probability(belief, i, o, spec.current)
                    if prob > 0:
                        b2 = self.cache.child(state, belief, i, o)
                        s2 = successor_state(state, i, o)
                        child_value, child_policy = self.solve(s2, b2, depth - 1)
                        value += prob * child_value
                    else:
                        child_policy = Stop(path)
                    children.append(child_policy)
                if value > best:
                    best, policy = value, Test(i, *children)
        self.memo[key] = best, policy
        return best, policy


def solve_exact(problem: PlanningProblem):
    cache = TransitionCache(problem)
    state = (0,) * (2 * len(problem.belief))
    value, policy = _Bellman(problem, cache).solve(state, problem.belief, problem.spec.horizon)
    return cache.result(policy, value, value, value)


def solve_stop(problem: PlanningProblem):
    cache = TransitionCache(problem)
    value, path = best_stop(problem.belief, problem.spec)
    return cache.result(Stop(path), value, value)


def solve_lookahead(problem: PlanningProblem, depth: int, *, _cache=None):
    if depth < 1:
        raise ValueError('lookahead depth must be positive')
    cache = TransitionCache(problem) if _cache is None else _cache
    bellman = _Bellman(problem, cache)
    spec = problem.spec

    def build(state, belief, remaining):
        _, candidate = bellman.solve(state, belief, min(depth, remaining))
        if isinstance(candidate, Stop):
            return best_stop(belief, spec)[0], candidate
        value, children = -spec.cost, []
        _, fallback_path = best_stop(belief, spec)
        for o in (0, 1):
            prob = outcome_probability(belief, candidate.path, o, spec.current)
            if prob > 0:
                b2 = cache.child(state, belief, candidate.path, o)
                s2 = successor_state(state, candidate.path, o)
                v, child = build(s2, b2, remaining - 1)
                value += prob * v
            else:
                child = Stop(fallback_path)
            children.append(child)
        return value, Test(candidate.path, *children)

    value, policy = build((0,) * (2 * len(problem.belief)), problem.belief, spec.horizon)
    return cache.result(policy, value, value)


def solve_iterative_trace(problem: PlanningProblem, budgets):
    """Capture the last complete depth at each budget in a single search."""
    start = perf_counter()
    budgets = tuple(sorted(set(budgets)))
    if not budgets or min(budgets) < 0:
        raise ValueError('budgets must be nonempty and nonnegative')
    cache = TransitionCache(problem, max(budgets))
    bellman = _Bellman(problem, cache)
    state = (0,) * (2 * len(problem.belief))
    value, path = best_stop(problem.belief, problem.spec)
    policy = Stop(path)
    pending = set(budgets)
    snapshots = {}

    def before_evaluation(used):
        for budget in tuple(pending):
            if used >= budget:
                snapshots[budget] = (cache.result(policy, value, value, None, 'budget_exhausted'),
                                     perf_counter() - start)
                pending.remove(budget)

    cache.on_before_evaluation = before_evaluation
    for depth in range(1, problem.spec.horizon + 1):
        try:
            candidate_value, candidate = bellman.solve(state, problem.belief, depth)
        except BudgetExhausted:
            break
        value, policy = candidate_value, candidate
    for budget in pending:
        snapshots[budget] = (cache.result(policy, value, value, None, 'complete'), perf_counter() - start)
    return snapshots


def solve_iterative(problem: PlanningProblem, max_evaluations: int):
    return solve_iterative_trace(problem, [max_evaluations])[max_evaluations][0]
