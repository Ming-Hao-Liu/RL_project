"""Anytime AND/OR search with executable lower-bound policies."""
from dataclasses import dataclass, field
from time import perf_counter
from ..bounds import perfect_information
from ..model import PlanningProblem, best_stop, outcome_probability
from ..policy import Stop, Test
from .common import BudgetExhausted, TransitionCache, successor_state


@dataclass(slots=True)
class _Action:
    upper: float
    children: tuple | None = None
    probabilities: tuple[float, float] = (0.0, 0.0)
    pruned: bool = False


@dataclass(slots=True)
class _Node:
    state: tuple[int, ...]
    belief: tuple
    remaining: int
    lower: float
    upper: float
    policy: object
    actions: list[_Action] = field(default_factory=list)
    frontier_score: float = 0.0
    choice: int | None = None
    outcome: int | None = None


def solve_bounded_trace(problem: PlanningProblem, budgets, epsilons, *, prune: bool = True, _seed_info=None):
    """Capture immutable policy snapshots for all budget/tolerance combinations in one run."""
    start = perf_counter()
    budgets, epsilons = tuple(sorted(set(budgets))), tuple(sorted(set(epsilons)))
    if not budgets or not epsilons or min(budgets) < 0 or min(epsilons) < 0:
        raise ValueError('budgets and epsilons must be nonempty and nonnegative')
    cache = TransitionCache(problem, max(budgets))
    spec = problem.spec
    targets = {(budget, epsilon) for budget in budgets for epsilon in epsilons}
    snapshots = {}

    def node(state, belief, remaining):
        stop_value, path = best_stop(belief, spec)
        action_upper = perfect_information(belief, spec) - spec.cost
        upper = max(stop_value, action_upper) if remaining else stop_value
        actions = [_Action(action_upper) for _ in belief] if remaining else []
        result = _Node(state, belief, remaining, stop_value, upper, Stop(path), actions)
        refresh(result)
        return result

    def refresh(n):
        n.frontier_score, n.choice, n.outcome = 0.0, None, None
        if not n.actions or n.upper <= n.lower:
            return
        candidates = []
        for i, action in enumerate(n.actions):
            if prune and action.pruned:
                continue
            cache.stats.action_scans += 1
            candidates.append((action.upper, -i, i))
        if not candidates:
            return
        _, _, i = max(candidates)
        action = n.actions[i]
        if action.upper <= n.lower:
            return
        n.choice = i
        if action.children is None:
            n.frontier_score = n.upper - n.lower
            return
        scores = [(prob * child.frontier_score, -o, o) for o, (child, prob) in
                  enumerate(zip(action.children, action.probabilities)) if child is not None and prob > 0]
        if scores:
            score, _, outcome = max(scores)
            n.frontier_score, n.outcome = score, outcome

    def backup(n):
        stop, path = best_stop(n.belief, spec)
        lower, policy = n.lower, n.policy
        upper = max(stop, lower)
        for i, action in enumerate(n.actions):
            if prune and action.pruned:
                continue
            cache.stats.action_scans += 1
            if action.children is not None:
                qlo, qup = -spec.cost, -spec.cost
                policies = []
                for child, probability in zip(action.children, action.probabilities):
                    if child is None:
                        policies.append(Stop(path))
                    else:
                        qlo += probability * child.lower
                        qup += probability * child.upper
                        policies.append(child.policy)
                action.upper = qup
                if qlo > lower:
                    lower, policy = qlo, Test(i, *policies)
            upper = max(upper, action.upper)
        if prune:
            for action in n.actions:
                if action.upper <= lower:
                    action.pruned = True
        if lower + 1e-10 < n.lower or upper > n.upper + 1e-10 or upper + 1e-10 < lower:
            raise ArithmeticError('invalid or non-monotonic search bounds')
        n.lower, n.upper, n.policy = lower, upper, policy
        refresh(n)

    def frontier(n):
        if n.frontier_score <= 0 or n.choice is None:
            return None
        ancestors = []
        while n.actions[n.choice].children is not None:
            if n.outcome is None:
                return None
            ancestors.append(n)
            n = n.actions[n.choice].children[n.outcome]
        return n, n.choice, ancestors

    root = node((0,) * (2 * len(problem.belief)), problem.belief, spec.horizon)
    if _seed_info is not None:
        # A warm call has one total budget. Initialization shares the same cache
        # and retains STOP if interrupted; unfinished policies are never returned.
        if len(budgets) != 1:
            raise ValueError('warm initialization requires a single total budget')
        seed_start = perf_counter()
        seed_status = 'analytic_stop' if root.upper <= root.lower else 'low_budget_cold'
        if root.upper > root.lower and max(budgets) >= 1000:
            from .dynamic import solve_lookahead
            try:
                seed = solve_lookahead(problem, 1, _cache=cache)
                if seed.model_value > root.lower:
                    root.lower, root.policy = seed.model_value, seed.policy
                seed_status = 'complete'
            except BudgetExhausted:
                seed_status = 'incomplete'
            refresh(root)
        _seed_info.update(status=seed_status, evaluations=cache.stats.evaluations,
                          lower=root.lower, upper=root.upper,
                          seconds=perf_counter() - seed_start)

    def capture(target, status):
        snapshots[target] = (cache.result(root.policy, root.lower, root.lower, root.upper, status),
                             perf_counter() - start)
        targets.remove(target)

    while targets:
        width = root.upper - root.lower
        for target in tuple(targets):
            if width <= target[1] * spec.reward:
                capture(target, 'complete' if width <= 0 else 'tolerance')
        if not targets:
            break
        selected = frontier(root)
        if selected is None:
            for target in tuple(targets):
                capture(target, 'numerical_stall')
            break
        leaf, index, ancestors = selected
        probs = tuple(outcome_probability(leaf.belief, index, o, spec.current) for o in (0, 1))
        outcomes = [o for o, p in enumerate(probs) if p > 0]
        needed = cache.needed(leaf.state, index, outcomes)
        for target in tuple(targets):
            if cache.stats.evaluations + needed > target[0]:
                capture(target, 'budget_exhausted')
        if not targets:
            break
        children = []
        for outcome, probability in enumerate(probs):
            if probability > 0:
                belief = cache.child(leaf.state, leaf.belief, index, outcome)
                children.append(node(successor_state(leaf.state, index, outcome), belief, leaf.remaining - 1))
            else:
                children.append(None)
        leaf.actions[index].children = tuple(children)
        leaf.actions[index].probabilities = probs
        cache.stats.expansions += 1
        backup(leaf)
        for ancestor in reversed(ancestors):
            backup(ancestor)
    return snapshots


def solve_bounded(problem: PlanningProblem, max_evaluations: int, epsilon: float, *, prune: bool = True):
    return solve_bounded_trace(problem, [max_evaluations], [epsilon], prune=prune)[(max_evaluations, epsilon)][0]


def solve_bounded_warm(problem: PlanningProblem, max_evaluations: int, epsilon: float):
    """Root myopic incumbent, fully billed; budgets below 1000 use cold search.

    Returns a normal PlanResult and the initialization checkpoint. Only a
    completed myopic construction guarantees dominance over full myopic.
    """
    initialization = {}
    result = solve_bounded_trace(problem, [max_evaluations], [epsilon],
                                 _seed_info=initialization)[(max_evaluations, epsilon)][0]
    return result, initialization
