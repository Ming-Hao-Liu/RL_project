"""Executable root incumbents, shared accounting and bounded interruption."""
import random
import pytest
from rl_retest.model import ModelSpec, PlanningProblem, EvaluationContext
from rl_retest.evaluation import evaluate
from rl_retest.planners.dynamic import solve_exact, solve_lookahead
from rl_retest.planners.bounded import solve_bounded, solve_bounded_warm
from rl_retest.planners import common


def test_warm_certificate_and_incumbent():
    rng = random.Random(20261108)
    for horizon in (0, 1, 3):
        for cost in (0., .005, .1):
            belief = tuple(tuple(v / sum(a) for v in a) for a in
                           ([rng.random() for _ in range(3)] for _ in range(3)))
            p = PlanningProblem(belief, ModelSpec(1, cost, horizon, .5, (0,.5,.9), ()))
            optimum = solve_exact(p).model_value
            myopic = solve_lookahead(p, 1).model_value
            for epsilon in (0, .01):
                plan, seed = solve_bounded_warm(p, 1000, epsilon)
                actual = evaluate(plan.policy, EvaluationContext(p.belief,p.spec)).expected_utility
                assert actual == pytest.approx(plan.lower, abs=1e-10)
                assert plan.lower <= optimum + 1e-10 <= plan.upper + 2e-10
                if seed['status'] == 'complete':
                    assert plan.lower >= myopic - 1e-10
                assert plan.evaluations <= 1000


def test_warm_low_budget_and_analytic_stop():
    for probabilities in ((0,.5,.9), (0,0,1)):
        p = PlanningProblem(((.2,.6,.2),), ModelSpec(1,.005,3,.5,probabilities,()))
        for budget in (0,1,100):
            warm, seed = solve_bounded_warm(p,budget,0)
            assert warm == solve_bounded(p,budget,0)
    p = PlanningProblem(((0,0,1),), ModelSpec(1,.02,6,.5,(0,.5,.9),()))
    plan, seed = solve_bounded_warm(p,10000,0)
    assert seed['status']=='analytic_stop' and plan.evaluations==0


def test_warm_actual_update_accounting(monkeypatch):
    p = PlanningProblem(((.2,.6,.2),)*4, ModelSpec(1,.005,5,.5,(0,.5,.9),()))
    observed = []
    original = common.update
    def counted(*args, **kwargs):
        observed.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(common,'update',counted)
    plan, seed = solve_bounded_warm(p,1000,0)
    assert len(observed)==plan.evaluations and seed['evaluations']>0


def test_interrupted_initialization_retains_executable_stop():
    p = PlanningProblem(((.2,.6,.2),)*40, ModelSpec(1,0,12,.5,(0,.5,.9),()))
    plan, seed = solve_bounded_warm(p,1000,0)
    assert seed['status']=='incomplete'
    assert plan.evaluations==1000
    assert evaluate(plan.policy, EvaluationContext(p.belief,p.spec)).expected_utility == plan.lower
